# MIT License
# Copyright (c) 2019 Hyeonwoo Kang
# code was based on https://github.com/znxlwm/UGATIT-pytorch

import functools
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ...modules.norm import build_norm_layer
from ...modules.utils import spectral_norm
from .builder import GENERATORS


@GENERATORS.register()
class ResnetUGATITGenerator(nn.Layer):
    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf=64,
                 n_blocks=6,
                 img_size=256,
                 light=False,
                 norm_type='instance'):
        assert (n_blocks >= 0)
        super(ResnetUGATITGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        norm_layer = build_norm_layer(norm_type)
        DownBlock = []
        DownBlock += [
            nn.Pad2D(padding=[3, 3, 3, 3], mode="reflect"),
            nn.Conv2D(input_nc,
                      ngf,
                      kernel_size=7,
                      stride=1,
                      padding=0,
                      bias_attr=False),
            norm_layer(ngf),
            nn.ReLU()
        ]

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [
                nn.Pad2D(padding=[1, 1, 1, 1], mode="reflect"),
                nn.Conv2D(ngf * mult,
                          ngf * mult * 2,
                          kernel_size=3,
                          stride=2,
                          padding=0,
                          bias_attr=False),
                norm_layer(ngf * mult * 2),
                nn.ReLU()
            ]

        # Down-Sampling Bottleneck
        mult = 2**n_downsampling
        for i in range(n_blocks):
            DownBlock += [
                ResnetBlock(ngf * mult, use_bias=False, norm_layer=norm_layer)
            ]

        # Class Activation Map
        self.gap_fc = nn.Linear(ngf * mult, 1, bias_attr=False)
        self.gmp_fc = nn.Linear(ngf * mult, 1, bias_attr=False)
        self.conv1x1 = nn.Conv2D(ngf * mult * 2,
                                 ngf * mult,
                                 kernel_size=1,
                                 stride=1,
                                 bias_attr=True)
        self.relu = nn.ReLU()

        # Gamma, Beta block
        if self.light:
            FC = [
                nn.Linear(ngf * mult, ngf * mult, bias_attr=False),
                nn.ReLU(),
                nn.Linear(ngf * mult, ngf * mult, bias_attr=False),
                nn.ReLU()
            ]
        else:
            FC = [
                nn.Linear(img_size // mult * img_size // mult * ngf * mult,
                          ngf * mult,
                          bias_attr=False),
                nn.ReLU(),
                nn.Linear(ngf * mult, ngf * mult, bias_attr=False),
                nn.ReLU()
            ]
        self.gamma = nn.Linear(ngf * mult, ngf * mult, bias_attr=False)
        self.beta = nn.Linear(ngf * mult, ngf * mult, bias_attr=False)

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i + 1),
                    ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock2 += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Pad2D(padding=[1, 1, 1, 1], mode="reflect"),
                nn.Conv2D(ngf * mult,
                          int(ngf * mult / 2),
                          kernel_size=3,
                          stride=1,
                          padding=0,
                          bias_attr=False),
                ILN(int(ngf * mult / 2)),
                nn.ReLU()
            ]

        UpBlock2 += [
            nn.Pad2D(padding=[3, 3, 3, 3], mode="reflect"),
            nn.Conv2D(ngf,
                      output_nc,
                      kernel_size=7,
                      stride=1,
                      padding=0,
                      bias_attr=False),
            nn.Tanh()
        ]

        self.DownBlock = nn.Sequential(*DownBlock)
        self.FC = nn.Sequential(*FC)
        self.UpBlock2 = nn.Sequential(*UpBlock2)

    def forward(self, input):
        x = self.DownBlock(input)

        gap = F.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.reshape([x.shape[0], -1]))
        gap_weight = list(self.gap_fc.parameters())[0].transpose([1, 0])
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = F.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.reshape([x.shape[0], -1]))
        gmp_weight = list(self.gmp_fc.parameters())[0].transpose([1, 0])
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = paddle.concat([gap_logit, gmp_logit], 1)
        x = paddle.concat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))

        heatmap = paddle.sum(x, axis=1, keepdim=True)

        if self.light:
            x_ = F.adaptive_avg_pool2d(x, 1)
            x_ = self.FC(x_.reshape([x_.shape[0], -1]))
        else:
            x_ = self.FC(x.reshape([x.shape[0], -1]))
        gamma, beta = self.gamma(x_), self.beta(x_)

        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i + 1))(x, gamma, beta)
        out = self.UpBlock2(x)

        return out, cam_logit, heatmap


class ResnetBlock(nn.Layer):
    def __init__(self, dim, use_bias, norm_layer):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [
            nn.Pad2D(padding=[1, 1, 1, 1], mode="reflect"),
            nn.Conv2D(dim,
                      dim,
                      kernel_size=3,
                      stride=1,
                      padding=0,
                      bias_attr=use_bias),
            norm_layer(dim),
            nn.ReLU()
        ]

        conv_block += [
            nn.Pad2D(padding=[1, 1, 1, 1], mode="reflect"),
            nn.Conv2D(dim,
                      dim,
                      kernel_size=3,
                      stride=1,
                      padding=0,
                      bias_attr=use_bias),
            norm_layer(dim)
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetAdaILNBlock(nn.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = nn.Pad2D(padding=[1, 1, 1, 1], mode="reflect")
        self.conv1 = nn.Conv2D(dim,
                               dim,
                               kernel_size=3,
                               stride=1,
                               padding=0,
                               bias_attr=use_bias)
        self.norm1 = AdaILN(dim)
        self.relu1 = nn.ReLU()

        self.pad2 = nn.Pad2D(padding=[1, 1, 1, 1], mode="reflect")
        self.conv2 = nn.Conv2D(dim,
                               dim,
                               kernel_size=3,
                               stride=1,
                               padding=0,
                               bias_attr=use_bias)
        self.norm2 = AdaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x


class AdaILN(nn.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(AdaILN, self).__init__()
        self.eps = eps
        shape = (1, num_features, 1, 1)

        self.rho = self.create_parameter(shape)
        self.rho.set_value(paddle.full(shape, 0.9))

    def forward(self, input, gamma, beta):
        in_mean, in_var = paddle.mean(input, [2, 3],
                                      keepdim=True), paddle.var(input, [2, 3],
                                                                keepdim=True)
        out_in = (input - in_mean) / paddle.sqrt(in_var + self.eps)
        ln_mean, ln_var = paddle.mean(input, [1, 2, 3],
                                      keepdim=True), paddle.var(input,
                                                                [1, 2, 3],
                                                                keepdim=True)
        out_ln = (input - ln_mean) / paddle.sqrt(ln_var + self.eps)

        out = self.rho.expand([input.shape[0], -1, -1, -1]) * out_in + (
            1 - self.rho.expand([input.shape[0], -1, -1, -1])) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(
            2).unsqueeze(3)

        return out


class ILN(nn.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        shape = (1, num_features, 1, 1)
        self.rho = self.create_parameter(shape)
        self.gamma = self.create_parameter(shape)
        self.beta = self.create_parameter(shape)
        self.rho.set_value(paddle.full(shape, 0.0))
        self.gamma.set_value(paddle.full(shape, 1.0))
        self.beta.set_value(paddle.full(shape, 0.0))

    def forward(self, input):
        in_mean, in_var = paddle.mean(input, [2, 3],
                                      keepdim=True), paddle.var(input, [2, 3],
                                                                keepdim=True)
        out_in = (input - in_mean) / paddle.sqrt(in_var + self.eps)
        ln_mean, ln_var = paddle.mean(input, [1, 2, 3],
                                      keepdim=True), paddle.var(input,
                                                                [1, 2, 3],
                                                                keepdim=True)
        out_ln = (input - ln_mean) / paddle.sqrt(ln_var + self.eps)
        out = self.rho.expand([input.shape[0], -1, -1, -1]) * out_in + (
            1 - self.rho.expand([input.shape[0], -1, -1, -1])) * out_ln
        out = out * self.gamma.expand([input.shape[0], -1, -1, -1
                                       ]) + self.beta.expand(
                                           [input.shape[0], -1, -1, -1])

        return out
