import torch

from train import PixelSharp

m = PixelSharp(1, 4, 4)
x = torch.ones(1, 1, 4, 4)
y1 = m(x)
x[..., 0, 0] = 0.
y2 = m(x)
print(y1.max(1)[0][..., 0, 0].squeeze())
print(y2.max(1)[0][..., 0, 0].squeeze())
