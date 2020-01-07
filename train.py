import math
import os
from collections import defaultdict

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
        self.x1 = nn.Parameter(torch.zeros(1, c_size, h_size, x1), requires_grad) if requires_grad else None
        self.x2 = nn.Parameter(torch.zeros(1, c_size, h_size, x2), requires_grad) if requires_grad else None
        w_size = x1 + w_size + x2
        self.y1 = nn.Parameter(torch.zeros(1, c_size, y1, w_size), requires_grad) if requires_grad else None
        self.y2 = nn.Parameter(torch.zeros(1, c_size, y2, w_size), requires_grad) if requires_grad else None

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
        sizes = x.shape
        x = x.view(sizes[0], -1, 2, *sizes[2:])
        a = x[:, :, 0]
        b = x[:, :, 1]
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


class MaskedConv2d(torch.jit.ScriptModule):
    """Accepts 4d (N, C, H, W) input but convolute horizontally only (1d)."""
    __constants__ = ['in_dim', 'out_dim', 'in_subpixel_c_size', 'out_subpixel_c_size',
                     'base_subpixel_conv_out_index', 'padding', 'kernel_size', 'mask_type']

    def __init__(self, in_dim, out_dim, kernel_size, c_size, mask_type, bias=True, **kwargs):
        super().__init__()
        dilation = kwargs.get('dilation', 1)
        assert mask_type in ('A', 'B')

        self.kernel_size = kernel_size
        self.c_size = c_size
        self.mask_type = mask_type
        self.in_subpixel_c_size = in_dim // c_size
        self.out_subpixel_c_size = out_dim // c_size
        self.in_dim = in_dim
        self.out_dim = out_dim

        # consider that self.conv is without current pixel
        self.padding = (kernel_size - 2) * dilation + 1
        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=(1, kernel_size - 1),
            dilation=dilation,
            padding=(0, self.padding),
            bias=False,
        ) if kernel_size > 1 else nn.Sequential()

        self.subpixel_convs = nn.ModuleList([
            nn.Conv2d(
                self.in_subpixel_c_size * (i + 1),
                self.out_subpixel_c_size,
                kernel_size=1,
                bias=False,
            )
            for i in range(c_size - (mask_type == 'A'))
        ])
        self.base_subpixel_conv_out_index = self.out_subpixel_c_size if self.mask_type == 'A' else 0

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim, 1, 1))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.jit.script_method
    def forward(self, x):
        assert x.size(1) == self.in_dim

        n_size, c_size, h_size, w_size = x.shape
        subpixel_conv_results = x.new_full((n_size, self.out_dim, h_size, w_size), math.nan)
        subpixel_conv_results[:, :self.base_subpixel_conv_out_index] = 0.

        i = 0

        for subpixel_conv in self.subpixel_convs:
            in_index = self.in_subpixel_c_size * (i + 1)
            out_index = self.base_subpixel_conv_out_index + self.out_subpixel_c_size * i
            subpixel_conv_results[:, out_index:out_index + self.out_subpixel_c_size] = \
                subpixel_conv(x[:, :in_index])
            i += 1

        if self.kernel_size >= 2:
            x = self.conv(x)[..., :-(self.padding+1)]
            x = x + subpixel_conv_results
        else:
            x = subpixel_conv_results

        return x + self.bias


class PixelCNN(nn.Module):
    def __init__(self, c_size, h_size, w_size, format='bit'):
        super().__init__()
        assert format in ('softmax', 'bit', 'byte')
        self.format = format

        self.sizes = input_sizes = c_size, h_size, w_size
        num_blocks = 1
        num_layers_per_block = 16
        dim = 192
        input_dim = c_size + 1  # + x0
        v_conv, v_act, h_conv, h_act, v2h = [], [], [], [], []

        r = 0

        for i in range(num_blocks):
            dilation = 2 ** i

            for j in range(num_layers_per_block):
                kernel_size = 7 if (i, j) == (0, 0) else 3
                r += (kernel_size - 1) * dilation

                padding_y = (kernel_size - 1) * dilation
                v_conv.append(
                    nn.Conv2d(
                        input_dim,
                        dim,
                        kernel_size,
                        dilation=dilation,
                        padding=(padding_y, kernel_size // 2),
                        # bias=False,
                    ),
                )
                h_conv.append(
                    MaskedConv2d(
                        input_dim,
                        dim,
                        kernel_size,
                        c_size,
                        dilation=dilation,
                        mask_type='A' if (i, j) == (0, 0) else 'B',
                        # bias=False,
                    ),
                )

                # TODO: Dense
                input_dim = dim // 2
                v_act.append(nn.Sequential(
                    # ConcatReLU(),
                    GatedActivation2d(),
                    # nn.Dropout2d(0.),
                    # nn.BatchNorm2d(input_dim),
                ))
                h_act.append(nn.Sequential(
                    # ConcatReLU(),
                    GatedActivation2d(),
                    # nn.Conv2d(input_dim, input_dim, 1),
                    MaskedConv2d(input_dim, input_dim, 1, c_size, mask_type='B'),
                    # nn.Dropout2d(0.),
                    # nn.BatchNorm2d(input_dim),
                ))
                v2h.append(nn.Conv2d(dim, dim, 1))
                input_sizes = (input_dim, *input_sizes[1:])

        self.x2vx = nn.Sequential(
            Lambda(lambda x: x[..., :-1, :]),
            LearnedPadding2d(c_size, h_size - 1, w_size, 0, 0, 1, requires_grad=False),
        )
        self.v_conv = nn.ModuleList(v_conv)
        self.v_act = nn.ModuleList(v_act)
        self.h_conv = nn.ModuleList(h_conv)
        self.h_act = nn.ModuleList(h_act)
        self.v2h = nn.ModuleList(v2h)
        self.out = nn.Sequential(
            MaskedConv2d(
                input_dim,
                input_dim,
                1,
                c_size,
                mask_type='B',
            ),
            nn.ReLU(),
            MaskedConv2d(
                input_dim,
                c_size * {'softmax': 256, 'byte': 8, 'bit': 1}[format],
                1,
                c_size,
                mask_type='B',
            ),
        )
        self.r = r

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.format == 'bit':
            x = x * 2. - 1.
        else:
            x = (x / 127.5 - 1.) * 2.

        # to distinguish zero paddings
        x = F.pad(x, (0, 0, 0, 0, 0, 1), value=1)
        vx = self.x2vx(x)
        hx = x
        # skip_connection = []
        # NOTE: DO NOT USE RESIDUAL FOR THE FIRST H LAYER
        res_hx = 0.

        for i, (v_conv, v_act, h_conv, h_act, v2h) in enumerate(
                zip(self.v_conv, self.v_act, self.h_conv, self.h_act, self.v2h)):
            v_conv_x = v_conv(vx)[..., :-v_conv.padding[0], :]
            vx = v_act(v_conv_x)
            # vx = checkpoint(v_act, v_conv_x)

            h_conv_x = h_conv(hx)
            hx = h_act(h_conv_x + v2h(v_conv_x))
            # h_act_x = checkpoint(h_act, h_conv_x + checkpoint(v2h, v_conv_x))

            if i % 1 == 0:
                res_hx = hx = hx + res_hx

        # x = sum(skip_connection) / len(skip_connection) + hx
        x = hx
        logits = self.out(x)

        if self.format == 'byte':
            return self.logprob_packbits(logits).transpose(1, 2)
        else:
            # NOTE: should care the behavior of masked cnn
            n_size, cb_size, h_size, w_size = logits.shape
            c_size = cb_size // 256
            return logits.view(n_size, c_size, 256, h_size, w_size).transpose(1, 2)

    def logprob_packbits(self, logits):
        num_bits = 8
        n_size, cb_size, h_size, w_size = logits.shape
        c_size = cb_size // num_bits
        logprobs = F.logsigmoid(logits).view(n_size, c_size, num_bits, 1, 1, h_size, w_size)
        log1mprobs = F.logsigmoid(-logits).view_as(logprobs)
        out = logprobs.new_zeros(n_size, c_size, 256, h_size, w_size)

        for i in range(num_bits):
            logp = logprobs[:, :, i]
            log1mp = log1mprobs[:, :, i]
            a = 1 << i
            b = 256 >> i
            out = out.view(n_size, c_size, a, b, h_size, w_size)
            out[:, :, :, :b // 2] += log1mp  # log(1 - p)
            out[:, :, :, b // 2:] += logp

        return out.view(n_size, c_size, 256, h_size, w_size)

    def sample(self, n=1):
        prev_deterministic = torch.backends.cudnn.deterministic
        torch.backends.cudnn.deterministic = True
        prev_device = next(self.parameters()).device
        device = torch.device('cuda')
        self.to(device)

        prev_training = self.training
        self.eval()
        c_size, h_size, w_size = self.sizes
        out = torch.zeros(n, c_size, h_size, w_size, dtype=torch.float, device=device)

        # assert (self(out) == self(out)).all(), breakpoint()

        for y in range(h_size):
            for x in range(w_size):
                for c in range(c_size):
                    # TODO: Uncomment following line after lifting LearnedPadding2d
                    logits = self(out[..., :y+1, :])
                    # self(torch.zeros_like(out))[:, 0, 0, 0]
                    # logits = self(out)
                    # probs = self.prob_packbits(logits)
                    # distribution = Categorical(probs=probs[:, c, :, y, x])
                    distribution = Bernoulli(logits=logits[:, c, y, x])
                    out[:, c, y, x] = distribution.sample()

        torch.backends.cudnn.deterministic = prev_deterministic
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

    if model.format == 'bit':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    nll_loss_criterion = nn.NLLLoss()
    metrics = defaultdict(lambda: 0.)
    num_metrics = 0

    def log_metrics():
        nonlocal num_metrics
        mean_metrics = {k: v / num_metrics for k, v in metrics.items()}
        # mean_metrics['loss'] = nat2bit(mean_metrics['loss'])
        # unit['loss'] = 'bpd'
        unit = defaultdict(str)

        for k, v in mean_metrics.items():
            print(f'[{epoch}/{train_step}] {k}: {v:.4f} {unit[k]}')
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

        if model.format == 'bit':
            logits = model(reshaped_y)
            loss = criterion(logits, reshaped_y)
        else:
            logits = model(x)
            nll_loss = loss = criterion(logits, x)

        if is_train:
            optimizer.zero_grad()

            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            loss.backward()

            optimizer.step()
            # scheduler.step()
            train_step += 1

        with torch.no_grad():
            if model.format == 'bit':
                log_probs = model.logprob_packbits(logits.flip(1))
                # log_probs = model.logprob_packbits(logits)
                nll_loss = nll_loss_criterion(log_probs.transpose(1, 2), x)
            else:
                log_probs = logits

            num_metrics += 1
            metrics['loss'] += nll_loss

            if is_train:
                metrics['bce_loss'] += loss

            if is_train and train_step % 100 == 99:
                loss = log_metrics()['loss']
                postfix = f'_{loss:.3f}_{ctx["pscp"]}'

                sample = [
                    x.float() / 127.5 - 1.,
                    log_probs.argmax(1) / 127.5 - 1.,
                ]

                if train_step % 10000 == 999 and False:
                    postfix += '_s'
                    sample_ = model.sample(1).view(-1, c_size, num_bits, h_size, w_size)
                    sample_ = sample_.cpu().numpy()[:, ::-1]
                    # sample_ = sample_.cpu().numpy()
                    sample_ = numpy.packbits(sample_ == 1., axis=2).squeeze(2)
                    sample.append(torch.from_numpy(sample_).to(device) / 127.5 - 1.)

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

    if len(image.shape) == 2:
        image = numpy.expand_dims(image, -1)

    binary_image = numpy.unpackbits(numpy.expand_dims(image, axis=-1), axis=-1)
    binary_image = numpy.ascontiguousarray(binary_image[..., ::-1])
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
    model = PixelCNN(3, 32, 32, format='softmax')
    # model = PixelCNN(3 * 8, 32, 32)
    print(f'r = {model.r}')
    model.to(device)
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=1e-3,
    )
    # model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    ctx['model'], ctx['optimizer'] = model, optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        verbose=True,
        min_lr=1e-4,
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

    for epoch in range(250):
        ctx['epoch'] = epoch
        print(f'lr: {optimizer.param_groups[0]["lr"]}')
        ctx = loop(ctx, 'train')

        with torch.no_grad():
            ctx = loop(ctx, 'val')
        """
        for _ in loop(ctx, 'train'):
            with torch.no_grad():
                list(loop(ctx, 'val'))

        """
        val_loss = ctx['val_metrics']['loss'].item()
        print(f'val_loss: {val_loss:.4f}')
        scheduler.step(val_loss)


if __name__ == '__main__':
    main()
