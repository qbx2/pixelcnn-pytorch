import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


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


class GatedActivation2d(nn.Module):
    def forward(self, x):
        sizes = x.shape
        x = x.view(sizes[0], -1, 2, *sizes[2:])
        a = x[:, :, 0]
        b = x[:, :, 1]
        return torch.tanh(a) * torch.sigmoid(b)


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
    def __init__(self, c_size, dim):
        super().__init__()
        num_blocks = 1
        num_layers_per_block = 14
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
                    ),
                )

                input_dim = dim // 2
                v_act.append(nn.Sequential(
                    GatedActivation2d(),
                ))
                h_act.append(nn.Sequential(
                    GatedActivation2d(),
                    MaskedConv2d(input_dim, input_dim, 1, c_size, mask_type='B'),
                ))
                v2h.append(nn.Conv2d(dim, dim, 1, bias=False))

        self.x2vx = nn.Sequential(
            Lambda(lambda x: x[..., :-1, :]),
            nn.ConstantPad2d((0, 0, 1, 0), value=0),
        )
        self.v_conv = nn.ModuleList(v_conv)
        self.v_act = nn.ModuleList(v_act)
        self.h_conv = nn.ModuleList(h_conv)
        self.h_act = nn.ModuleList(h_act)
        self.v2h = nn.ModuleList(v2h)
        self.out = nn.Sequential(
            ConcatReLU(),
            MaskedConv2d(
                input_dim * 2,
                input_dim,
                1,
                c_size,
                mask_type='B',
            ),
            ConcatReLU(),
            MaskedConv2d(
                input_dim * 2,
                c_size * 256,
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
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                assert m.bias is None

    def forward(self, x):
        x = x / 127.5 - 1.

        # to distinguish zero paddings
        x = F.pad(x, (0, 0, 0, 0, 0, 1), value=1.)
        vx = self.x2vx(x)
        hx = x
        # NOTE: DO NOT USE RESIDUAL FOR THE FIRST H LAYER
        res_hx = 0.

        for i, (v_conv, v_act, h_conv, h_act, v2h) in enumerate(
                zip(self.v_conv, self.v_act, self.h_conv, self.h_act, self.v2h)):
            v_conv_x = v_conv(vx)[..., :-v_conv.padding[0], :]
            vx = v_act(v_conv_x)

            h_conv_x = h_conv(hx)
            hx = h_act(h_conv_x + v2h(v_conv_x))

            res_hx = hx = hx + res_hx

        x = hx
        logits = self.out(x)

        # NOTE: should care the behavior of masked cnn
        n_size, cb_size, h_size, w_size = logits.shape
        c_size = cb_size // 256
        return logits.view(n_size, c_size, 256, h_size, w_size).transpose(1, 2)

    def sample(self, n=1):
        raise NotImplementedError
