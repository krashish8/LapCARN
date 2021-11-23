import torch
import torch.nn as nn
import model.ops as ops
import model.carn as carn

class LapCARN(nn.Module):
    def __init__(self):
        super(LapCARN, self).__init__()

        # For upsampling image with 3 channels (Image Reconstuction)
        self.upsample_3 = ops.UpsampleBlock(3)
        # For upsampling image with 64 channels (Feature Extraction)
        self.upsample_64 = ops.UpsampleBlock(64)
        self.feat = self.make_layer(carn.CARN)

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)

        self.entry = nn.Conv2d(3, 64, 3, 1, 1)
        self.exit = nn.Conv2d(64, 3, 3, 1, 1)

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.sub_mean(x)
        out = self.entry(out)

        convt_F1 = self.feat(out)
        convt_R1 = self.add_mean(self.exit(convt_F1))
        convt_I1 = self.upsample_3(x)
        HR_2x = convt_I1 + convt_R1

        convt_F2 = self.feat(convt_F1)
        convt_R2 = self.add_mean(self.exit(convt_F2))
        convt_I2 = self.upsample_3(HR_2x)
        HR_4x = convt_I2 + convt_R2

        convt_F3 = self.feat(convt_F2)
        convt_R3 = self.add_mean(self.exit(convt_F3))
        convt_I3 = self.upsample_3(HR_4x)
        HR_8x = convt_I3 + convt_R3

        return HR_2x, HR_4x, HR_8x
