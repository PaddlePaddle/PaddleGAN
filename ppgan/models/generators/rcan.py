# base on https://github.com/kongdebug/RCAN-Paddle
import math
import paddle
import paddle.nn as nn

from .builder import GENERATORS


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    weight_attr = paddle.ParamAttr(
        initializer=paddle.nn.initializer.XavierUniform(), need_clip=True)
    return nn.Conv2D(in_channels,
                     out_channels,
                     kernel_size,
                     padding=(kernel_size // 2),
                     weight_attr=weight_attr,
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


## Residual Group (RG)
class ResidualGroup(nn.Layer):

    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale,
                 n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

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
class RCAN(nn.Layer):

    def __init__(
        self,
        scale,
        n_resgroups,
        n_resblocks,
        n_feats=64,
        n_colors=3,
        rgb_range=255,
        kernel_size=3,
        reduction=16,
        conv=default_conv,
    ):
        super(RCAN, self).__init__()
        self.scale = scale
        act = nn.ReLU()

        n_resgroups = n_resgroups
        n_resblocks = n_resblocks
        n_feats = n_feats
        kernel_size = kernel_size
        reduction = reduction
        scale = scale
        act = nn.ReLU()

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale= 1, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x
