import os
from collections import defaultdict

import apex.amp as amp
import numpy
import pscp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.distributions import Bernoulli
from torch.utils.checkpoint import checkpoint
from torchvision.utils import save_image


def nat2bit(val):
    return torch.log2(torch.exp(val))


class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


class LearnedPadding2d(nn.Module):
    def __init__(self, c_size, h_size, w_size, x1, x2=0, y1=0, y2=0, requires_grad=True):
        super().__init__()
        self.args = x1, x2, y1, y2
        self.requires_grad = requires_grad
        self.x1 = nn.Parameter(torch.zeros(1, c_size, h_size, x1), requires_grad)
        self.x2 = nn.Parameter(torch.zeros(1, c_size, h_size, x2), requires_grad)
        w_size = x1 + w_size + x2
        self.y1 = nn.Parameter(torch.zeros(1, c_size, y1, w_size), requires_grad)
        self.y2 = nn.Parameter(torch.zeros(1, c_size, y2, w_size), requires_grad)

    def forward(self, x):
        x1, x2, y1, y2 = self.args
        x = F.pad(x, self.args)

        if not self.requires_grad:
            return x

        neg_x2 = -x2 if x2 else None
        neg_y2 = -y2 if y2 else None

        if x1:
            x[..., y1:neg_y2, :x1] = self.x1

        if x2:
            x[..., y1:neg_y2, neg_x2:] = self.x2

        if y1:
            x[..., :y1, :] = self.y1

        if y2:
            x[..., neg_y2:, :] = self.y2

        return x


class GatedActivation2d(nn.Module):
    def forward(self, x):
        a = x[:, ::2]
        b = x[:, 1::2]
        return torch.tanh(a) * torch.sigmoid(b)


class ConcatReLU(nn.Module):
    def forward(self, x):
        """
        Expects x with shape [N, C, *]
        Simple concatenation (e.g. [R, G, B] -> [R', G', B', -R', -G', -B'])
        and reads [R', G'] for R, [B', -R'] for G, [-G', -B'] for B, causing leaks.
        Should interleave for PixelCNN
        """
        sizes = x.shape
        return torch.stack((F.relu(x), F.relu(-x)), dim=2).view(sizes[0], -1, *sizes[2:])


class ConcatMinMax(nn.Module):
    def __init__(self, pool_size=2):
        super().__init__()
        self.pool_size = pool_size

    def forward(self, x):
        """
        Expects x with shape [N, C, *]
        Simple concatenation (e.g. [R, G, B] -> [R', G', B', -R', -G', -B'])
        and reads [R', G'] for R, [B', -R'] for G, [-G', -B'] for B, causing leaks.
        Should interleave for PixelCNN
        """
        sizes = x.shape
        x = x.view(sizes[0], -1, self.pool_size, *sizes[2:])
        maxout = x.max(2)[0]
        minout = x.sum(2) - maxout
        return torch.stack((maxout, minout), dim=2).view(sizes[0], -1, *sizes[2:])


class MaskedConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, c_size, h_size, w_size, mask_type, **kwargs):
        super().__init__()
        dilation = kwargs.get('dilation', 1)
        assert mask_type in ('A', 'B')

        self.kernel_size = kernel_size
        self.c_size = c_size
        self.mask_type = mask_type
        self.in_subpixel_c_size = in_dim // c_size
        self.out_subpixel_c_size = out_dim // c_size
        self.out_dim = out_dim

        padding = (kernel_size - 1) * dilation
        self.conv = nn.Sequential(
            Lambda(lambda x: x[..., :-padding]),
            LearnedPadding2d(in_dim, h_size, w_size - padding, padding, requires_grad=False),
            nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size-1, dilation=dilation),
        ) if kernel_size > 1 else Lambda(lambda x: 0.)

        self.subpixel_convs = nn.ModuleList([
            nn.Conv2d(
                self.in_subpixel_c_size * (i + 1),
                self.out_subpixel_c_size,
                kernel_size=1,
            )
            for i in range(c_size - (mask_type == 'A'))
        ])
        self.base_subpixel_conv_out_index = self.out_subpixel_c_size if self.mask_type == 'A' else 0

    def forward(self, x):
        n_size, c_size, h_size, w_size = x.shape
        subpixel_conv_results = x.new_empty(n_size, self.out_dim, h_size, w_size)

        for i, subpixel_conv in enumerate(self.subpixel_convs):
            in_index = self.in_subpixel_c_size * (i + 1)
            out_index = self.base_subpixel_conv_out_index + self.out_subpixel_c_size * i
            subpixel_conv_results[:, out_index:out_index + self.out_subpixel_c_size] = \
                subpixel_conv(x[:, :in_index])

        x = self.conv(x)
        return x + subpixel_conv_results


class PixelSharp(nn.Module):
    def __init__(self, c_size, h_size, w_size):
        super().__init__()

        self.sizes = sizes = input_sizes = c_size, h_size, w_size
        num_blocks = 3
        num_layers_per_block = 5
        dim = 144  # TODO: * 3
        input_dim = c_size
        v_conv, v_act, h_conv, h_act, v2h, h2h = [], [], [], [], [], []

        # TODO: h0
        r = 0

        for i in range(num_blocks):
            dilation = 2 ** i

            for j in range(num_layers_per_block):
                r += dilation  # kernel_size == 3

                padding = dilation
                v_conv.append(nn.Sequential(
                    LearnedPadding2d(*input_sizes, padding, padding, padding, 0,
                                     requires_grad=False),
                    nn.Conv2d(
                        input_dim,
                        dim,
                        kernel_size=(2, 3),
                        dilation=dilation,
                    ),
                ))
                v_act.append(nn.Sequential(
                    ConcatMinMax(),
                    nn.BatchNorm2d(dim),
                    # nn.Dropout2d(),
                ))

                if (i, j) == (0, 0):
                    # NOTE: DO NOT USE RESIDUAL FOR THE FIRST H LAYER
                    h_conv.append(
                        MaskedConv2d(input_dim, dim, 2, *sizes, dilation=dilation, mask_type='A'),
                    )
                else:
                    h_conv.append(
                        MaskedConv2d(input_dim, dim, 2, *sizes, dilation=dilation, mask_type='B'),
                    )

                h_act.append(nn.Sequential(
                    ConcatMinMax(),
                    nn.BatchNorm2d(dim),
                    # nn.Dropout2d(),
                ))

                # TODO: Dense
                input_dim = dim
                v2h.append(nn.Conv2d(dim, dim, 1))
                h2h.append(MaskedConv2d(input_dim, input_dim, 1, *sizes, mask_type='B'))
                input_sizes = (input_dim, *input_sizes[1:])

        self.x2vx = nn.Sequential(
            Lambda(lambda x: x[..., :-1, :]),
            LearnedPadding2d(c_size, h_size - 1, w_size, 0, 0, 1, requires_grad=False),
        )
        self.vx2vx = nn.Conv2d(c_size, v_conv[0][-1].out_channels, 1)
        self.v_conv = nn.ModuleList(v_conv)
        self.v_act = nn.ModuleList(v_act)
        self.h_conv = nn.ModuleList(h_conv)
        self.h_act = nn.ModuleList(h_act)
        self.v2h = nn.ModuleList(v2h)
        self.h2h = nn.ModuleList(h2h)
        self.out = nn.Sequential(
            ConcatMinMax(),
            MaskedConv2d(input_dim, 384, 1, *sizes, mask_type='B'),
            # nn.ReLU(),
            ConcatMinMax(),
            MaskedConv2d(384, c_size, 1, *sizes, mask_type='B'),
            # nn.ReLU(),
        )
        self.r = r

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x = x / 127.5 - 1.
        x = x * 2. - 1.
        vx = self.x2vx(x)
        hx = x
        skip_connection = []

        for i, (v_conv, v_act, h_conv, h_act, v2h, h2h) in enumerate(
                zip(self.v_conv, self.v_act, self.h_conv, self.h_act, self.v2h, self.h2h)):
            if i == 0:
                res_hx = 0.
                res_vx = self.vx2vx(vx)
            else:
                res_hx = hx
                res_vx = vx

            v_conv_x = v_conv(vx)
            h_conv_x = h_conv(hx)

            vx = res_vx + v_act(v_conv_x)
            # vx = res_vx + checkpoint(v_act, v_conv_x)

            h_act_x = h_act(h_conv_x + v2h(v_conv_x))
            # h_act_x = checkpoint(h_act, h_conv_x + checkpoint(v2h, v_conv_x))

            hx = res_hx + h2h(h_act_x)
            # hx = res_hx + checkpoint(h2h, h_act_x)

            skip_connection.append(h_act_x)

        x = sum(skip_connection)
        # NOTE: should care behavior of masked cnn
        logits = self.out(x)
        return logits

    def prob_packbits(self, logits):
        num_bits = 8
        n_size, cb_size, h_size, w_size = logits.shape
        c_size = cb_size // num_bits
        probs = torch.sigmoid(logits).view(n_size, c_size, num_bits, 1, 1, h_size, w_size)
        out = probs.new_ones(n_size, c_size, 256, h_size, w_size)

        for i in range(num_bits):
            p = probs[:, :, i]
            a = 1 << i
            b = 256 >> i
            out = out.view(n_size, c_size, a, b, h_size, w_size)
            out[:, :, :, :b // 2] *= 1. - p
            out[:, :, :, b // 2:] *= p

        return out.view(n_size, c_size, 256, h_size, w_size)

    def sample(self, n=1):
        prev_device = next(self.parameters()).device
        device = torch.device('cpu')
        self.to(device)

        prev_training = self.training
        self.eval()
        c_size, h_size, w_size = self.sizes
        out = torch.zeros(n, c_size, h_size, w_size, dtype=torch.float, device=device)

        for y in range(h_size):
            for x in range(w_size):
                for c in range(c_size):
                    # TODO: Uncomment following line after lifting LearnedPadding2d
                    logits = self(out[..., :y+1, :])
                    # logits = self(out)
                    # probs = self.prob_packbits(logits)
                    # distribution = Categorical(probs=probs[:, c, :, y, x])
                    distribution = Bernoulli(logits=logits[:, c, y, x])
                    out[:, c, y, x] = distribution.sample()

        self.train(prev_training)
        self.to(prev_device)
        return out


def loop(ctx, phase):
    device = ctx['device']
    dataloader = ctx['dataloaders'][phase]
    optimizer = ctx['optimizer']
    # scheduler = ctx['scheduler']
    model = ctx['model']
    # logger = ctx['logger']
    epoch = ctx['epoch']
    out_dir = ctx['out_dir']
    train_step = ctx.get('train_step', 0)
    print(f'{ctx["pscp"]}/Epoch {epoch}/{phase}')

    is_train = phase == 'train'
    model.train(is_train)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    nll_loss_criterion = nn.NLLLoss()
    metrics = defaultdict(lambda: 0.)
    num_metrics = 0

    def log_metrics():
        nonlocal num_metrics
        mean_metrics = {k: v / num_metrics for k, v in metrics.items()}
        mean_metrics['loss'] = nat2bit(mean_metrics['loss'])

        for k, v in mean_metrics.items():
            print(f'[{epoch}/{train_step}] {k}: {v:.4f}')
            # logger.add_scalar(f'{phase}/{k}', v, train_step)

        metrics.clear()
        num_metrics = 0
        return mean_metrics

    os.makedirs(out_dir, exist_ok=True)

    for (x, y), _ in dataloader:
        x = x.to(device, torch.long, non_blocking=True).permute(0, 3, 1, 2)
        y = y.to(device, torch.float, non_blocking=True).permute(0, 3, 4, 1, 2)
        n_size, c_size, num_bits, h_size, w_size = y.shape
        reshaped_y = y.view(n_size, c_size * num_bits, h_size, w_size)

        normalized_x = x.float() / 127.5 - 1.
        logits = model(reshaped_y)

        if is_train:
            optimizer.zero_grad()
            loss = criterion(logits, reshaped_y)

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            # loss.backward()

            optimizer.step()
            # scheduler.step()
            train_step += 1

        with torch.no_grad():
            probs = model.prob_packbits(logits)
            nll_loss = nll_loss_criterion(probs.transpose(1, 2).log(), x)
            num_metrics += 1
            metrics['loss'] += nll_loss

            if is_train:
                metrics['bce_loss'] += loss

            if is_train and train_step % 100 == 99:
                loss = log_metrics()['loss']
                postfix = f'_{loss:.3f}_{ctx["pscp"]}'

                sample = [normalized_x, probs.argmax(2) / 127.5 - 1.]

                if train_step % 1000 == 999 and False:
                    postfix += '_s'
                    sample = model.sample(1).cpu().numpy()
                    breakpoint()
                    sample = numpy.packbits(sample)
                    sample.append(sample / 127.5 - 1.)

                # if i % (m * 10) == 0:
                #     resultsample.append(vae.decode(z))

                sample = torch.cat(sample, dim=0)
                sample = (sample * 0.5 + 0.5).cpu()
                save_image(
                    sample,
                    f'{out_dir}/sample_{epoch:02d}_{train_step:06d}{postfix}.png',
                )

    newctx = {**ctx, 'train_step': train_step}

    if not is_train:
        newctx['val_metrics'] = log_metrics()

    return newctx


def transform(image):
    image = numpy.array(image)
    # return image
    binary_image = numpy.unpackbits(numpy.expand_dims(image, axis=-1), axis=-1)
    return image, binary_image


def main():
    ctx = {'pscp': pscp.create()}
    ctx['device'] = device = torch.device('cuda')
    datasets = {
        # phase: torchvision.datasets.MNIST(
        phase: torchvision.datasets.CIFAR10(
            'datasets',
            train=phase == 'train',
            download=True,
            transform=transform,
        )
        for phase in ('train', 'val')
    }
    ctx['out_dir'] = f'results_{datasets["train"].__class__.__name__}'
    ctx['dataloaders'] = {
        phase: torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        for phase, dataset in datasets.items()
    }
    model = PixelSharp(3 * 8, 32, 32)
    print(f'r = {model.r}')
    model.to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=5e-3,
    )
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    ctx['model'], ctx['optimizer'] = model, optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        verbose=True,
        min_lr=1e-5,
    )
    '''
    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=5e-6,
        max_lr=5e-3,
        cycle_momentum=False,
        step_size_up=2500,
    )
    '''
    ctx['scheduler'] = scheduler

    for epoch in range(150):
        ctx['epoch'] = epoch
        print(f'lr: {optimizer.param_groups[0]["lr"]}')
        ctx = loop(ctx, 'train')

        with torch.no_grad():
            ctx = loop(ctx, 'val')

        val_loss = ctx['val_metrics']['loss'].item()
        print(f'val_loss: {val_loss:.4f} bits/dim')
        scheduler.step(val_loss)


if __name__ == '__main__':
    main()
