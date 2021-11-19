# MIT License
# Copyright (c) 2020 Yong Guo
# code was based on https://github.com/guoyongcs/DRN

import math
import paddle
import paddle.nn as nn

from .builder import GENERATORS


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2D(in_channels,
                     out_channels,
                     kernel_size,
                     padding=(kernel_size // 2),
                     bias_attr=bias)


class MeanShift(nn.Conv2D):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = paddle.to_tensor(rgb_std)
        self.weight.set_value(paddle.eye(3).reshape([3, 3, 1, 1]))
        self.weight.set_value(self.weight / (std.reshape([3, 1, 1, 1])))

        mean = paddle.to_tensor(rgb_mean)
        self.bias.set_value(sign * rgb_range * mean / std)

        self.weight.trainable = False
        self.bias.trainable = False


class DownBlock(nn.Layer):
    def __init__(self,
                 negval,
                 n_feats,
                 n_colors,
                 scale,
                 nFeat=None,
                 in_channels=None,
                 out_channels=None):
        super(DownBlock, self).__init__()

        if nFeat is None:
            nFeat = n_feats

        if in_channels is None:
            in_channels = n_colors

        if out_channels is None:
            out_channels = n_colors

        dual_block = [
            nn.Sequential(
                nn.Conv2D(in_channels,
                          nFeat,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          bias_attr=False), nn.LeakyReLU(negative_slope=negval))
        ]

        for _ in range(1, int(math.log2(scale))):
            dual_block.append(
                nn.Sequential(
                    nn.Conv2D(nFeat,
                              nFeat,
                              kernel_size=3,
                              stride=2,
                              padding=1,
                              bias_attr=False),
                    nn.LeakyReLU(negative_slope=negval)))

        dual_block.append(
            nn.Conv2D(nFeat,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias_attr=False))

        self.dual_module = nn.Sequential(*dual_block)

    def forward(self, x):
        x = self.dual_module(x)
        return x


## Channel Attention (CA) Layer
class CALayer(nn.Layer):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2D(channel,
                      channel // reduction,
                      1,
                      padding=0,
                      bias_attr=True), nn.ReLU(),
            nn.Conv2D(channel // reduction,
                      channel,
                      1,
                      padding=0,
                      bias_attr=True), nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Layer):
    def __init__(self,
                 conv,
                 n_feat,
                 kernel_size,
                 reduction=16,
                 bias=True,
                 bn=False,
                 act=nn.ReLU(),
                 res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2D(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2D(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU())
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2D(n_feats))

            if act == 'relu':
                m.append(nn.ReLU())
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


@GENERATORS.register()
class DRNGenerator(nn.Layer):
    """DRNGenerator"""
    def __init__(
        self,
        scale,
        n_blocks=30,
        n_feats=16,
        n_colors=3,
        rgb_range=255,
        negval=0.2,
        kernel_size=3,
        conv=default_conv,
    ):
        super(DRNGenerator, self).__init__()
        self.scale = scale
        self.phase = len(scale)
        act = nn.ReLU()

        self.upsample = nn.Upsample(scale_factor=max(scale),
                                    mode='bicubic',
                                    align_corners=False)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)

        self.head = conv(n_colors, n_feats, kernel_size)

        self.down = [
            DownBlock(negval, n_feats, n_colors, 2, n_feats * pow(2, p),
                      n_feats * pow(2, p), n_feats * pow(2, p + 1))
            for p in range(self.phase)
        ]

        self.down = nn.LayerList(self.down)

        up_body_blocks = [[
            RCAB(conv, n_feats * pow(2, p), kernel_size, act=act)
            for _ in range(n_blocks)
        ] for p in range(self.phase, 1, -1)]

        up_body_blocks.insert(0, [
            RCAB(conv, n_feats * pow(2, self.phase), kernel_size, act=act)
            for _ in range(n_blocks)
        ])

        # The fisrt upsample block
        up = [[
            Upsampler(conv, 2, n_feats * pow(2, self.phase), act=False),
            conv(n_feats * pow(2, self.phase),
                 n_feats * pow(2, self.phase - 1),
                 kernel_size=1)
        ]]

        # The rest upsample blocks
        for p in range(self.phase - 1, 0, -1):
            up.append([
                Upsampler(conv, 2, 2 * n_feats * pow(2, p), act=False),
                conv(2 * n_feats * pow(2, p),
                     n_feats * pow(2, p - 1),
                     kernel_size=1)
            ])

        self.up_blocks = nn.LayerList()
        for idx in range(self.phase):
            self.up_blocks.append(nn.Sequential(*up_body_blocks[idx], *up[idx]))

        # tail conv that output sr imgs
        tail = [conv(n_feats * pow(2, self.phase), n_colors, kernel_size)]
        for p in range(self.phase, 0, -1):
            tail.append(conv(n_feats * pow(2, p), n_colors, kernel_size))
        self.tail = nn.LayerList(tail)

        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # upsample x to target sr size
        x = self.upsample(x)

        # preprocess
        x = self.sub_mean(x)
        x = self.head(x)

        # down phases,
        copies = []
        for idx in range(self.phase):
            copies.append(x)
            x = self.down[idx](x)

        # up phases
        sr = self.tail[0](x)
        sr = self.add_mean(sr)
        results = [sr]
        for idx in range(self.phase):
            # upsample to SR features
            x = self.up_blocks[idx](x)
            # concat down features and upsample features
            x = paddle.concat((x, copies[self.phase - idx - 1]), 1)
            # output sr imgs
            sr = self.tail[idx + 1](x)
            sr = self.add_mean(sr)

            results.append(sr)

        return results
