import os
from collections import defaultdict

import numpy
import pscp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
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
    def __init__(self, c_size, h_size, w_size, x1, x2=0, y1=0, y2=0):
        super().__init__()
        self.args = x1, x2, y1, y2
        self.x1 = nn.Parameter(torch.zeros(1, c_size, h_size, x1))
        self.x2 = nn.Parameter(torch.zeros(1, c_size, h_size, x2))
        w_size = x1 + w_size + x2
        self.y1 = nn.Parameter(torch.zeros(1, c_size, y1, w_size))
        self.y2 = nn.Parameter(torch.zeros(1, c_size, y2, w_size))

    def forward(self, x):
        x1, x2, y1, y2 = self.args
        x = F.pad(x, self.args)
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
        each_c_size = x.size(1) // 2
        a = x[:, :each_c_size]
        b = x[:, each_c_size:]
        return torch.tanh(a) * torch.sigmoid(b)


class ConcatReLU(nn.Module):
    def forward(self, x):
        """Expects x with shape [N, C, *]"""
        return torch.cat((F.relu(x), F.relu(-x)), dim=1)


class ConcatMinMax(nn.Module):
    def __init__(self, pool_size=2):
        super().__init__()
        self.pool_size = pool_size

    def forward(self, x):
        sizes = x.shape
        x = x.view(sizes[0], self.pool_size, -1, *sizes[2:])
        maxout = x.max(1)[0]
        minout = x.sum(1) - maxout
        return torch.cat((maxout, minout), dim=1)


# class MaskAConv2d(nn.Module):
#     def __init__(self, in_dim, out_dim, c_size, h_size, w_size, **kwargs):
#         super().__init__()
#         assert kwargs.get('dilation', 1) == 1
#
#         self.c_size = c_size
#         self.in_rg_index = in_subpixel_c_size = in_dim // c_size
#         self.in_gb_index = self.in_rg_index * 2
#         self.out_rg_index = out_subpixel_c_size = out_dim // c_size
#         self.out_gb_index = self.out_rg_index * 2
#
#         self.conv = nn.Sequential(
#             Lambda(lambda x: x[..., :-1]),
#             LearnedPadding2d(in_dim, h_size, w_size - 1, 1),
#             nn.Conv2d(in_dim, out_dim, kernel_size=1),
#         )
#         self.r_conv = nn.Conv2d(
#             in_subpixel_c_size,
#             out_subpixel_c_size,
#             kernel_size=1,
#         )
#         self.rg_conv = nn.Conv2d(
#             in_subpixel_c_size * 2,
#             out_subpixel_c_size,
#             kernel_size=1,
#         )
#
#     def forward(self, x):
#         r = x[:, :self.in_rg_index]
#         rg = x[:, :self.in_gb_index]
#         x = self.conv(x)
#         x[:, self.out_rg_index:self.out_gb_index] += self.r_conv(r)
#         x[:, self.out_gb_index:] += self.rg_conv(rg)
#         return x


class MaskedConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, c_size, h_size, w_size, mask_type, **kwargs):
        super().__init__()
        dilation = kwargs.get('dilation', 1)
        assert dilation >= 1

        self.c_size = c_size
        self.mask_type = mask_type
        self.in_rg_index = in_subpixel_c_size = in_dim // c_size
        self.in_gb_index = self.in_rg_index * 2
        self.out_rg_index = out_subpixel_c_size = out_dim // c_size
        self.out_gb_index = self.out_rg_index * 2

        padding = dilation
        self.conv = nn.Sequential(
            Lambda(lambda x: x[..., :-padding]),
            LearnedPadding2d(in_dim, h_size, w_size - padding, padding),
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
        )
        self.r_conv = nn.Conv2d(
            in_subpixel_c_size,
            out_subpixel_c_size,
            kernel_size=1,
        )
        self.rg_conv = nn.Conv2d(
            in_subpixel_c_size * 2,
            out_subpixel_c_size,
            kernel_size=1,
        )

        if mask_type == 'A':
            self.rgb_conv = nn.Conv2d(
                in_subpixel_c_size * 3,
                out_subpixel_c_size,
                kernel_size=1,
            )

    def forward(self, x):
        r = x[:, :self.in_rg_index]
        rg = x[:, :self.in_gb_index]
        rgb = x
        x = self.conv(x)

        if self.mask_type == 'A':
            x[:, self.out_rg_index:self.out_gb_index] += self.r_conv(r)
            x[:, self.out_gb_index:] += self.rg_conv(rg)
        elif self.mask_type == 'B':
            x[:, :self.out_rg_index] += self.r_conv(r)
            x[:, self.out_rg_index:self.out_gb_index] += self.rg_conv(rg)
            x[:, self.out_gb_index:] += self.rgb_conv(rgb)

        return x


class PixelSharp(nn.Module):
    def __init__(self, c_size, h_size, w_size):
        super().__init__()

        v_conv, v_act, h_conv, h_act, v2h, h2h = [], [], [], [], [], []
        num_blocks = 4
        num_layers_per_block = 4
        dim = 192
        input_dim = c_size
        sizes = input_sizes = c_size, h_size, w_size

        self.x2vx = nn.Sequential(
            Lambda(lambda x: x[..., :-1, :]),
            LearnedPadding2d(c_size, h_size - 1, w_size, 0, 0, 1),
        )
        r = 0

        for i in range(num_blocks):
            dilation = i + 1

            for j in range(num_layers_per_block):
                r += dilation  # kernel_size == 3

                padding = dilation
                v_conv.append(nn.Sequential(
                    LearnedPadding2d(*input_sizes, padding, padding, padding, 0),
                    nn.Conv2d(
                        input_dim,
                        dim,
                        kernel_size=(2, 3),
                        dilation=dilation,
                    ),
                ))
                v_act.append(nn.Sequential(
                    nn.BatchNorm2d(dim),
                    GatedActivation2d(),
                    # nn.Dropout2d(),
                ))

                if (i, j) == (0, 0):
                    # NOTE: DO NOT USE RESIDUAL FOR FIRST H LAYER
                    h_conv.append(
                        MaskedConv2d(input_dim, dim, *sizes, dilation=dilation, mask_type='A'),
                    )
                else:
                    h_conv.append(
                        MaskedConv2d(input_dim, dim, *sizes, dilation=dilation, mask_type='B'),
                    )

                h_act.append(nn.Sequential(
                    nn.BatchNorm2d(dim),
                    GatedActivation2d(),
                    # nn.Dropout2d(),
                ))

                input_dim = dim // 2
                v2h.append(nn.Conv2d(dim, dim, 1))
                h2h.append(nn.Conv2d(input_dim, input_dim, 1))
                input_sizes = (input_dim, *input_sizes[1:])

        self.vx2vx = nn.Conv2d(c_size, dim // 2, 1)
        self.v_conv = nn.ModuleList(v_conv)
        self.v_act = nn.ModuleList(v_act)
        self.h_conv = nn.ModuleList(h_conv)
        self.h_act = nn.ModuleList(h_act)
        self.v2h = nn.ModuleList(v2h)
        self.h2h = nn.ModuleList(h2h)
        self.out = nn.Sequential(
            ConcatReLU(),
            MaskedConv2d(input_dim * 2, 768, *sizes, mask_type='B'),
            ConcatReLU(),
            MaskedConv2d(768 * 2, 256 * c_size, *sizes, mask_type='B'),
        )
        self.r = r

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        n_size, c_size, h_size, w_size = x.shape
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
            vx = res_vx + v_act(v_conv_x)
            h_conv_x = h_conv(hx)
            h_act_x = h2h(h_act(h_conv_x + v2h(v_conv_x)))
            hx = res_hx + h_act_x
            skip_connection.append(h_act_x)

        x = sum(skip_connection)
        return self.out(x).view(n_size, 256, c_size, h_size, w_size)


def loop(ctx, phase):
    device = ctx['device']
    dataloader = ctx['dataloaders'][phase]
    optimizer = ctx['optimizer']
    model = ctx['model']
    # logger = ctx['logger']
    epoch = ctx['epoch']
    out_dir = ctx['out_dir']
    train_step = ctx.get('train_step', 0)
    print(f'{ctx["pscp"]}/Epoch {epoch}/{phase}')

    is_train = phase == 'train'
    model.train(is_train)
    criterion = nn.CrossEntropyLoss()
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

    for i, (x, _) in enumerate(dataloader):
        x = x.to(device, non_blocking=True).permute(0, 3, 1, 2)

        normalized_x = x.float() / 127.5 - 1.
        logits = model(normalized_x)
        loss = criterion(logits, x.long())

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_step += 1

        with torch.no_grad():
            num_metrics += 1
            metrics['loss'] += loss

            if is_train and num_metrics % 100 == 0:
                loss = log_metrics()['loss']
                postfix = f'{loss:.3f}'

                sample = [normalized_x, logits.argmax(1) / 127.5 - 1.]

                # if i % (m * 10) == 0:
                #     sample.append(vae.decode(z, init='normal'))

                # if i % (m * 10) == 0:
                #     resultsample.append(vae.decode(z))

                sample = torch.cat(sample, dim=0)
                sample = (sample * 0.5 + 0.5).cpu()
                save_image(
                    sample,
                    f'{out_dir}/sample_{epoch:02d}_{i:04d}_{postfix}.png',
                )

    newctx = {**ctx, 'train_step': train_step}

    if not is_train:
        newctx['val_metrics'] = log_metrics()

    return newctx


def transform(image):
    return numpy.array(image)


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
            num_workers=1,
            pin_memory=True,
        )
        for phase, dataset in datasets.items()
    }
    ctx['model'] = model = PixelSharp(3, 32, 32)
    print(f'r = {model.r}')
    model.to(device)
    ctx['optimizer'] = optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    for epoch in range(500):
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
