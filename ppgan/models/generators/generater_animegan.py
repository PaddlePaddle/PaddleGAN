# code was heavily based on https://github.com/TachibanaYoshino/AnimeGANv2
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/TachibanaYoshino/AnimeGANv2#license

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .builder import GENERATORS


class Conv2DNormLReLU(nn.Layer):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 bias_attr=False) -> None:
        super().__init__()
        self.conv = nn.Conv2D(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              padding,
                              bias_attr=bias_attr)
        # NOTE layer norm is crucial for animegan!
        self.norm = nn.GroupNorm(1, out_channels)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.lrelu(x)
        return x


class ResBlock(nn.Layer):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            Conv2DNormLReLU(in_channels, out_channels, 1, padding=0),
            Conv2DNormLReLU(out_channels, out_channels, 3),
            nn.Conv2D(out_channels, out_channels // 2, 1, bias_attr=False))

    def forward(self, x0):
        x = self.body(x0)
        return x0 + x


class InvertedresBlock(nn.Layer):
    def __init__(self,
                 in_channels: int,
                 expansion: float,
                 out_channels: int,
                 bias_attr=False):
        super().__init__()
        self.in_channels = in_channels
        self.expansion = expansion
        self.out_channels = out_channels
        self.bottle_channels = round(self.expansion * self.in_channels)
        self.body = nn.Sequential(
            # pw
            Conv2DNormLReLU(self.in_channels,
                            self.bottle_channels,
                            kernel_size=1,
                            bias_attr=bias_attr),
            # dw
            nn.Conv2D(self.bottle_channels,
                      self.bottle_channels,
                      kernel_size=3,
                      stride=1,
                      padding=0,
                      groups=self.bottle_channels,
                      bias_attr=True),
            nn.GroupNorm(1, self.bottle_channels),
            nn.LeakyReLU(0.2),
            # pw & linear
            nn.Conv2D(self.bottle_channels,
                      self.out_channels,
                      kernel_size=1,
                      padding=0,
                      bias_attr=False),
            nn.GroupNorm(1, self.out_channels),
        )

    def forward(self, x0):
        x = self.body(x0)
        if self.in_channels == self.out_channels:
            out = paddle.add(x0, x)
        else:
            out = x
        return x


@GENERATORS.register()
class AnimeGeneratorLite(nn.Layer):
    def __init__(self) -> None:
        super().__init__()
        self.A = nn.Sequential(Conv2DNormLReLU(3, 32, 7, padding=3),
                               Conv2DNormLReLU(32, 32, stride=2),
                               Conv2DNormLReLU(32, 32))

        self.B = nn.Sequential(Conv2DNormLReLU(32, 64, stride=2),
                               Conv2DNormLReLU(64, 64), Conv2DNormLReLU(64, 64))

        self.C = nn.Sequential(ResBlock(64, 128), ResBlock(64, 128),
                               ResBlock(64, 128), ResBlock(64, 128))

        self.D = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                               Conv2DNormLReLU(64, 64), Conv2DNormLReLU(64, 64),
                               Conv2DNormLReLU(64, 64))

        self.E = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                               Conv2DNormLReLU(64, 32), Conv2DNormLReLU(32, 32),
                               Conv2DNormLReLU(32, 32, 7, padding=3))

        self.out = nn.Sequential(nn.Conv2D(32, 3, 1, bias_attr=False),
                                 nn.Tanh())

    def forward(self, x):
        x = self.A(x)
        x = self.B(x)
        x = self.C(x)
        x = self.D(x)
        x = self.E(x)
        x = self.out(x)
        return x


@GENERATORS.register()
class AnimeGenerator(nn.Layer):
    def __init__(self) -> None:
        super().__init__()
        self.A = nn.Sequential(Conv2DNormLReLU(3, 32, 7, padding=3),
                               Conv2DNormLReLU(32, 64, stride=2),
                               Conv2DNormLReLU(64, 64))

        self.B = nn.Sequential(Conv2DNormLReLU(64, 128, stride=2),
                               Conv2DNormLReLU(128, 128),
                               Conv2DNormLReLU(128, 128))

        self.C = nn.Sequential(InvertedresBlock(128, 2, 256),
                               InvertedresBlock(256, 2, 256),
                               InvertedresBlock(256, 2, 256),
                               InvertedresBlock(256, 2, 256),
                               Conv2DNormLReLU(256, 128))

        self.D = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                               Conv2DNormLReLU(128, 128),
                               Conv2DNormLReLU(128, 128))

        self.E = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                               Conv2DNormLReLU(128, 64),
                               Conv2DNormLReLU(64, 64),
                               Conv2DNormLReLU(64, 32, 7, padding=3))

        self.out = nn.Sequential(nn.Conv2D(32, 3, 1, bias_attr=False),
                                 nn.Tanh())

    def forward(self, x):
        x = self.A(x)
        x = self.B(x)
        x = self.C(x)
        x = self.D(x)
        x = self.E(x)
        x = self.out(x)
        return x
