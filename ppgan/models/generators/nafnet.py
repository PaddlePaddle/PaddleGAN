# Copyright (c) 2022 megvii-model. All Rights Reserved.
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors

import paddle
from paddle import nn as nn
import paddle.nn.functional as F
from paddle.autograd import PyLayer

from .builder import GENERATORS


class LayerNormFunction(PyLayer):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.shape
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.reshape([1, C, 1, 1]) * y + bias.reshape([1, C, 1, 1])
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.shape
        y, var, weight = ctx.saved_tensor()
        g = grad_output * weight.reshape([1, C, 1, 1])
        mean_g = g.mean(axis=1, keepdim=True)

        mean_gy = (g * y).mean(axis=1, keepdim=True)
        gx = 1. / paddle.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(axis=3).sum(axis=2).sum(
            axis=0), grad_output.sum(axis=3).sum(axis=2).sum(axis=0)


class LayerNorm2D(nn.Layer):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2D, self).__init__()
        self.add_parameter(
            'weight',
            self.create_parameter(
                [channels],
                default_initializer=paddle.nn.initializer.Constant(value=1.0)))
        self.add_parameter(
            'bias',
            self.create_parameter(
                [channels],
                default_initializer=paddle.nn.initializer.Constant(value=0.0)))
        self.eps = eps

    def forward(self, x):
        if self.training:
            y = LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
        else:
            N, C, H, W = x.shape
            mu = x.mean(1, keepdim=True)
            var = (x - mu).pow(2).mean(1, keepdim=True)
            y = (x - mu) / (var + self.eps).sqrt()
            y = self.weight.reshape([1, C, 1, 1]) * y + self.bias.reshape(
                [1, C, 1, 1])

        return y


class AvgPool2D(nn.Layer):

    def __init__(self,
                 kernel_size=None,
                 base_size=None,
                 auto_pad=True,
                 fast_imp=False,
                 train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp)

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[
                0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[
                1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.shape[-2] and self.kernel_size[
                1] >= x.shape[-1]:
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(axis=-1).cumsum(axis=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(
                    w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] -
                       s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = paddle.nn.functional.interpolate(out,
                                                       scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(axis=-1).cumsum(axis=-2)
            s = paddle.nn.functional.pad(s,
                                         [1, 0, 1, 0])  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1,
                                                    k2:], s[:, :,
                                                            k1:, :-k2], s[:, :,
                                                                          k1:,
                                                                          k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            pad2d = [(w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2,
                     (h - _h + 1) // 2]
            out = paddle.nn.functional.pad(out, pad2d, mode='replicate')

        return out


def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2D):
            pool = AvgPool2D(base_size=base_size,
                             fast_imp=fast_imp,
                             train_size=train_size)
            assert m._output_size == 1
            setattr(model, n, pool)


'''
ref.
@article{chu2021tlsc,
  title={Revisiting Global Statistics Aggregation for Improving Image Restoration},
  author={Chu, Xiaojie and Chen, Liangyu and and Chen, Chengpeng and Lu, Xin},
  journal={arXiv preprint arXiv:2112.04491},
  year={2021}
}
'''


class Local_Base():

    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = paddle.rand(train_size)
        with paddle.no_grad():
            self.forward(imgs)


class SimpleGate(nn.Layer):

    def forward(self, x):
        x1, x2 = x.chunk(2, axis=1)
        return x1 * x2


class NAFBlock(nn.Layer):

    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2D(in_channels=c,
                               out_channels=dw_channel,
                               kernel_size=1,
                               padding=0,
                               stride=1,
                               groups=1,
                               bias_attr=True)
        self.conv2 = nn.Conv2D(in_channels=dw_channel,
                               out_channels=dw_channel,
                               kernel_size=3,
                               padding=1,
                               stride=1,
                               groups=dw_channel,
                               bias_attr=True)
        self.conv3 = nn.Conv2D(in_channels=dw_channel // 2,
                               out_channels=c,
                               kernel_size=1,
                               padding=0,
                               stride=1,
                               groups=1,
                               bias_attr=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(in_channels=dw_channel // 2,
                      out_channels=dw_channel // 2,
                      kernel_size=1,
                      padding=0,
                      stride=1,
                      groups=1,
                      bias_attr=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2D(in_channels=c,
                               out_channels=ffn_channel,
                               kernel_size=1,
                               padding=0,
                               stride=1,
                               groups=1,
                               bias_attr=True)
        self.conv5 = nn.Conv2D(in_channels=ffn_channel // 2,
                               out_channels=c,
                               kernel_size=1,
                               padding=0,
                               stride=1,
                               groups=1,
                               bias_attr=True)

        self.norm1 = LayerNorm2D(c)
        self.norm2 = LayerNorm2D(c)

        self.drop_out_rate = drop_out_rate

        self.dropout1 = nn.Dropout(
            drop_out_rate) if drop_out_rate > 0. else None
        self.dropout2 = nn.Dropout(
            drop_out_rate) if drop_out_rate > 0. else None

        self.add_parameter(
            "beta",
            self.create_parameter(
                [1, c, 1, 1],
                default_initializer=paddle.nn.initializer.Constant(value=0.0)))
        self.add_parameter(
            "gamma",
            self.create_parameter(
                [1, c, 1, 1],
                default_initializer=paddle.nn.initializer.Constant(value=0.0)))

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        if self.drop_out_rate > 0:
            x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        if self.drop_out_rate > 0:
            x = self.dropout2(x)

        return y + x * self.gamma


@GENERATORS.register()
class NAFNet(nn.Layer):

    def __init__(self,
                 img_channel=3,
                 width=16,
                 middle_blk_num=1,
                 enc_blk_nums=[],
                 dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2D(in_channels=img_channel,
                               out_channels=width,
                               kernel_size=3,
                               padding=1,
                               stride=1,
                               groups=1,
                               bias_attr=True)
        self.ending = nn.Conv2D(in_channels=width,
                                out_channels=img_channel,
                                kernel_size=3,
                                padding=1,
                                stride=1,
                                groups=1,
                                bias_attr=True)

        self.encoders = nn.LayerList()
        self.decoders = nn.LayerList()
        self.middle_blks = nn.LayerList()
        self.ups = nn.LayerList()
        self.downs = nn.LayerList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2D(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(nn.Conv2D(chan, chan * 2, 1, bias_attr=False),
                              nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2**len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.shape
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, [0, mod_pad_w, 0, mod_pad_h])
        return x


@GENERATORS.register()
class NAFNetLocal(Local_Base, NAFNet):

    def __init__(self,
                 *args,
                 train_size=(1, 3, 256, 256),
                 fast_imp=False,
                 **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with paddle.no_grad():
            self.convert(base_size=base_size,
                         train_size=train_size,
                         fast_imp=fast_imp)
