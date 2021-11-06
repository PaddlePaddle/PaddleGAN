# code was based on https://github.com/hellloxiaotian/LESRCNN
import math
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .builder import GENERATORS


class MeanShift(nn.Layer):
    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()

        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign

        self.shifter = nn.Conv2D(3, 3, 1, 1, 0)
        self.shifter.weight.set_value(paddle.eye(3).reshape([3, 3, 1, 1]))
        self.shifter.bias.set_value(np.array([r, g, b]).astype('float32'))
        # Freeze the mean shift layer
        for params in self.shifter.parameters():
            params.trainable = False

    def forward(self, x):
        x = self.shifter(x)
        return x


class UpsampleBlock(nn.Layer):
    def __init__(self, n_channels, scale, multi_scale, group=1):
        super(UpsampleBlock, self).__init__()

        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, group=group)
            self.up3 = _UpsampleBlock(n_channels, scale=3, group=group)
            self.up4 = _UpsampleBlock(n_channels, scale=4, group=group)
        else:
            self.up = _UpsampleBlock(n_channels, scale=scale, group=group)

        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)


class _UpsampleBlock(nn.Layer):
    def __init__(self, n_channels, scale, group=1):
        super(_UpsampleBlock, self).__init__()

        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [
                    nn.Conv2D(n_channels, 4 * n_channels, 3, 1, 1, groups=group)
                ]
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            modules += [
                nn.Conv2D(n_channels, 9 * n_channels, 3, 1, 1, groups=group)
            ]
            modules += [nn.PixelShuffle(3)]

        self.body = nn.Sequential(*modules)

    def forward(self, x):
        out = self.body(x)
        return out


@GENERATORS.register()
class LESRCNNGenerator(nn.Layer):
    """Construct a Resnet-based generator that consists of residual blocks
    between a few downsampling/upsampling operations.

    Args:
        scale (int): scale of upsample.
        multi_scale (bool): Whether to train multi scale model.
        group (int): group option for convolution.
    """
    def __init__(
        self,
        scale=4,
        multi_scale=False,
        group=1,
    ):
        super(LESRCNNGenerator, self).__init__()

        kernel_size = 3
        kernel_size1 = 1
        padding1 = 0
        padding = 1
        features = 64
        groups = 1
        channels = 3
        features1 = 64
        self.scale = scale
        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)

        self.conv1 = nn.Sequential(
            nn.Conv2D(in_channels=channels,
                      out_channels=features,
                      kernel_size=kernel_size,
                      padding=padding,
                      groups=1,
                      bias_attr=False))
        self.conv2 = nn.Sequential(
            nn.Conv2D(in_channels=features,
                      out_channels=features,
                      kernel_size=kernel_size,
                      padding=1,
                      groups=1,
                      bias_attr=False), nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2D(in_channels=features,
                      out_channels=features,
                      kernel_size=kernel_size1,
                      padding=0,
                      groups=groups,
                      bias_attr=False))
        self.conv4 = nn.Sequential(
            nn.Conv2D(in_channels=features,
                      out_channels=features,
                      kernel_size=kernel_size,
                      padding=1,
                      groups=1,
                      bias_attr=False), nn.ReLU())
        self.conv5 = nn.Sequential(
            nn.Conv2D(in_channels=features,
                      out_channels=features,
                      kernel_size=kernel_size1,
                      padding=0,
                      groups=groups,
                      bias_attr=False))
        self.conv6 = nn.Sequential(
            nn.Conv2D(in_channels=features,
                      out_channels=features,
                      kernel_size=kernel_size,
                      padding=1,
                      groups=1,
                      bias_attr=False), nn.ReLU())
        self.conv7 = nn.Sequential(
            nn.Conv2D(in_channels=features,
                      out_channels=features,
                      kernel_size=kernel_size1,
                      padding=0,
                      groups=groups,
                      bias_attr=False))
        self.conv8 = nn.Sequential(
            nn.Conv2D(in_channels=features,
                      out_channels=features,
                      kernel_size=kernel_size,
                      padding=1,
                      groups=1,
                      bias_attr=False), nn.ReLU())
        self.conv9 = nn.Sequential(
            nn.Conv2D(in_channels=features,
                      out_channels=features,
                      kernel_size=kernel_size1,
                      padding=0,
                      groups=groups,
                      bias_attr=False))
        self.conv10 = nn.Sequential(
            nn.Conv2D(in_channels=features,
                      out_channels=features,
                      kernel_size=kernel_size,
                      padding=1,
                      groups=1,
                      bias_attr=False), nn.ReLU())
        self.conv11 = nn.Sequential(
            nn.Conv2D(in_channels=features,
                      out_channels=features,
                      kernel_size=kernel_size1,
                      padding=0,
                      groups=groups,
                      bias_attr=False))
        self.conv12 = nn.Sequential(
            nn.Conv2D(in_channels=features,
                      out_channels=features,
                      kernel_size=kernel_size,
                      padding=1,
                      groups=1,
                      bias_attr=False), nn.ReLU())
        self.conv13 = nn.Sequential(
            nn.Conv2D(in_channels=features,
                      out_channels=features,
                      kernel_size=kernel_size1,
                      padding=0,
                      groups=groups,
                      bias_attr=False))
        self.conv14 = nn.Sequential(
            nn.Conv2D(in_channels=features,
                      out_channels=features,
                      kernel_size=kernel_size,
                      padding=1,
                      groups=1,
                      bias_attr=False), nn.ReLU())
        self.conv15 = nn.Sequential(
            nn.Conv2D(in_channels=features,
                      out_channels=features,
                      kernel_size=kernel_size1,
                      padding=0,
                      groups=groups,
                      bias_attr=False))
        self.conv16 = nn.Sequential(
            nn.Conv2D(in_channels=features,
                      out_channels=features,
                      kernel_size=kernel_size,
                      padding=1,
                      groups=1,
                      bias_attr=False), nn.ReLU())
        self.conv17 = nn.Sequential(
            nn.Conv2D(in_channels=features,
                      out_channels=features,
                      kernel_size=kernel_size1,
                      padding=0,
                      groups=groups,
                      bias_attr=False))
        self.conv17_1 = nn.Sequential(
            nn.Conv2D(in_channels=features,
                      out_channels=features,
                      kernel_size=kernel_size,
                      padding=1,
                      groups=1,
                      bias_attr=False), nn.ReLU())
        self.conv17_2 = nn.Sequential(
            nn.Conv2D(in_channels=features,
                      out_channels=features,
                      kernel_size=kernel_size,
                      padding=1,
                      groups=1,
                      bias_attr=False), nn.ReLU())
        self.conv17_3 = nn.Sequential(
            nn.Conv2D(in_channels=features,
                      out_channels=features,
                      kernel_size=kernel_size,
                      padding=1,
                      groups=1,
                      bias_attr=False), nn.ReLU())
        self.conv17_4 = nn.Sequential(
            nn.Conv2D(in_channels=features,
                      out_channels=features,
                      kernel_size=kernel_size,
                      padding=1,
                      groups=1,
                      bias_attr=False), nn.ReLU())
        self.conv18 = nn.Sequential(
            nn.Conv2D(in_channels=features,
                      out_channels=3,
                      kernel_size=kernel_size,
                      padding=padding,
                      groups=groups,
                      bias_attr=False))

        self.ReLU = nn.ReLU()
        self.upsample = UpsampleBlock(64,
                                      scale=scale,
                                      multi_scale=multi_scale,
                                      group=1)

    def forward(self, x, scale=None):
        if scale is None:
            scale = self.scale

        x = self.sub_mean(x)

        x1 = self.conv1(x)
        x1_1 = self.ReLU(x1)
        x2 = self.conv2(x1_1)
        x3 = self.conv3(x2)

        x2_3 = x1 + x3
        x2_4 = self.ReLU(x2_3)
        x4 = self.conv4(x2_4)
        x5 = self.conv5(x4)
        x3_5 = x2_3 + x5

        x3_6 = self.ReLU(x3_5)
        x6 = self.conv6(x3_6)
        x7 = self.conv7(x6)
        x7_1 = x3_5 + x7

        x7_2 = self.ReLU(x7_1)
        x8 = self.conv8(x7_2)
        x9 = self.conv9(x8)
        x9_2 = x7_1 + x9

        x9_1 = self.ReLU(x9_2)
        x10 = self.conv10(x9_1)
        x11 = self.conv11(x10)
        x11_1 = x9_2 + x11

        x11_2 = self.ReLU(x11_1)
        x12 = self.conv12(x11_2)
        x13 = self.conv13(x12)
        x13_1 = x11_1 + x13

        x13_2 = self.ReLU(x13_1)
        x14 = self.conv14(x13_2)
        x15 = self.conv15(x14)
        x15_1 = x15 + x13_1

        x15_2 = self.ReLU(x15_1)
        x16 = self.conv16(x15_2)
        x17 = self.conv17(x16)
        x17_2 = x17 + x15_1

        x17_3 = self.ReLU(x17_2)
        temp = self.upsample(x17_3, scale=scale)
        x1111 = self.upsample(x1_1, scale=scale)
        temp1 = x1111 + temp
        temp2 = self.ReLU(temp1)
        temp3 = self.conv17_1(temp2)
        temp4 = self.conv17_2(temp3)
        temp5 = self.conv17_3(temp4)
        temp6 = self.conv17_4(temp5)
        x18 = self.conv18(temp6)
        out = self.add_mean(x18)

        return out
