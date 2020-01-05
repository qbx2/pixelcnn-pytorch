import torch
import torch.nn as nn

from train import PixelSharp

m = PixelSharp(8, 4, 4)
m.eval()

for _ in range(2):
    x = torch.ones(1, 8, 1, 1)
    y1 = m(x)
    x[..., 0, 0] = 0.
    y2 = m(x)
    print(y1.max(1)[0][..., 0, 0].squeeze())
    print(y2.max(1)[0][..., 0, 0].squeeze())
