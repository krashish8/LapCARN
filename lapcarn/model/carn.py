import torch
import torch.nn as nn
import model.ops as ops

class CascadingBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels):
        super(CascadingBlock, self).__init__()

        self.r1 = ops.ResidualBlock(64, 64)
        self.r2 = ops.ResidualBlock(64, 64)
        self.r3 = ops.ResidualBlock(64, 64)
        self.b1 = ops.BasicBlock(64*2, 64, 1, 1, 0)
        self.b2 = ops.BasicBlock(64*3, 64, 1, 1, 0)
        self.b3 = ops.BasicBlock(64*4, 64, 1, 1, 0)

    def forward(self, x):
        b0 = o0 = x

        r1 = self.r1(o0)
        b1 = torch.cat([b0, r1], dim=1)
        o1 = self.b1(b1)

        r2 = self.r2(o1)
        b2 = torch.cat([b1, r2], dim=1)
        o2 = self.b2(b2)

        r3 = self.r3(o2)
        b3 = torch.cat([b2, r3], dim=1)
        o3 = self.b3(b3)

        return o3


class CARN(nn.Module):
    def __init__(self):
        super(CARN, self).__init__()

        self.c1 = CascadingBlock(64, 64)
        self.c2 = CascadingBlock(64, 64)
        self.c3 = CascadingBlock(64, 64)
        self.b1 = ops.BasicBlock(64*2, 64, 1, 1, 0)
        self.b2 = ops.BasicBlock(64*3, 64, 1, 1, 0)
        self.b3 = ops.BasicBlock(64*4, 64, 1, 1, 0)

        self.upsample = ops.UpsampleBlock(64)

    def forward(self, x):
        b0 = o0 = x

        c1 = self.c1(o0)
        b1 = torch.cat([b0, c1], dim=1)
        o1 = self.b1(b1)

        c2 = self.c2(o1)
        b2 = torch.cat([b1, c2], dim=1)
        o2 = self.b2(b2)

        c3 = self.c3(o2)
        b3 = torch.cat([b2, c3], dim=1)
        o3 = self.b3(b3)

        out = self.upsample(o3)
        return out
