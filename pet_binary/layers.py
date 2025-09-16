import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA(nn.Module):
    """
    Exponential Moving Average.
    """
    def __init__(self, in_chn=8, length=80):
        super().__init__()
        self.in_chn = in_chn
        self.length = length
        self.alpha = nn.Parameter(torch.rand(in_chn, 1, 1), requires_grad=True)
        self.x0 = nn.Parameter(torch.rand(1, in_chn), requires_grad=True)
        self.exponent = torch.arange(length, 0, -1, dtype=torch.float32)

    def forward(self, x):
        alpha = self.alpha
        with torch.no_grad():
            alpha.clip_(.01, .99)
        weight = (1 - alpha) * torch.pow(alpha, self.exponent)

        x = F.pad(x, (self.length-1, 0), value=0)
        with torch.no_grad():
            self.x0.clip_(0, 6)
        x[:, :, self.length-2] += self.x0
        y = F.conv1d(x, weight, groups=self.in_chn)
        return y

    def extra_repr(self):
        return f'in_chn={self.in_chn}, length={self.length}'


class Transpose(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = torch.transpose(x, 1, -1)
        return y
