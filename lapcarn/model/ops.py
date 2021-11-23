import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class MeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()

        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign

        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data   = torch.Tensor([r, g, b])

        # Freeze the mean shift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.body(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self,
                 n_channels):
        super(UpsampleBlock, self).__init__()

        self.up =  _UpsampleBlock(n_channels)

    def forward(self, x):
        return self.up(x)


class _UpsampleBlock(nn.Module):
    def __init__(self,
				 n_channels):
        super(_UpsampleBlock, self).__init__()

        modules = []
        modules += [nn.Conv2d(n_channels, 4*n_channels, 3, 1, 1), nn.ReLU(inplace=True)]
        modules += [nn.PixelShuffle(2)]

        self.body = nn.Sequential(*modules)

    def forward(self, x):
        out = self.body(x)
        return out
