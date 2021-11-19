#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# code was heavily based on https://github.com/znxlwm/UGATIT-pytorch
# MIT License
# Copyright (c) 2019 Hyeonwoo Kang

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .builder import GENERATORS


@GENERATORS.register()
class ResnetUGATITP2CGenerator(nn.Layer):
    def __init__(self,
                 input_nc=3,
                 output_nc=3,
                 ngf=32,
                 img_size=256,
                 n_blocks=4,
                 light=True):
        super(ResnetUGATITP2CGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        DownBlock = []
        DownBlock += [
            nn.Pad2D([3, 3, 3, 3], 'reflect'),
            nn.Conv2D(input_nc, ngf, kernel_size=7, stride=1, bias_attr=False),
            nn.InstanceNorm2D(ngf, weight_attr=False, bias_attr=False),
            nn.ReLU()
        ]

        DownBlock += [HourGlass(ngf, ngf), HourGlass(ngf, ngf)]

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [
                nn.Pad2D([1, 1, 1, 1], 'reflect'),
                nn.Conv2D(ngf * mult,
                          ngf * mult * 2,
                          kernel_size=3,
                          stride=2,
                          bias_attr=False),
                nn.InstanceNorm2D(ngf * mult * 2,
                                  weight_attr=False,
                                  bias_attr=False),
                nn.ReLU()
            ]

        # Encoder Bottleneck
        mult = 2**n_downsampling
        for i in range(n_blocks):
            setattr(self, 'EncodeBlock' + str(i + 1), ResnetBlock(ngf * mult))

        # Class Activation Map
        self.gap_fc = nn.Linear(ngf * mult, 1, bias_attr=False)
        self.gmp_fc = nn.Linear(ngf * mult, 1, bias_attr=False)
        self.conv1x1 = nn.Conv2D(ngf * mult * 2,
                                 ngf * mult,
                                 kernel_size=1,
                                 stride=1)
        self.relu = nn.ReLU()

        # Gamma, Beta block
        FC = []
        if self.light:
            FC += [
                nn.Linear(ngf * mult, ngf * mult, bias_attr=False),
                nn.ReLU(),
                nn.Linear(ngf * mult, ngf * mult, bias_attr=False),
                nn.ReLU()
            ]

        else:
            FC += [
                nn.Linear(img_size // mult * img_size // mult * ngf * mult,
                          ngf * mult,
                          bias_attr=False),
                nn.ReLU(),
                nn.Linear(ngf * mult, ngf * mult, bias_attr=False),
                nn.ReLU()
            ]

        # Decoder Bottleneck
        mult = 2**n_downsampling
        for i in range(n_blocks):
            setattr(self, 'DecodeBlock' + str(i + 1),
                    ResnetSoftAdaLINBlock(ngf * mult))

        # Up-Sampling
        UpBlock = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock += [
                nn.Upsample(scale_factor=2),
                nn.Pad2D([1, 1, 1, 1], 'reflect'),
                nn.Conv2D(ngf * mult,
                          ngf * mult // 2,
                          kernel_size=3,
                          stride=1,
                          bias_attr=False),
                LIN(ngf * mult // 2),
                nn.ReLU()
            ]

        UpBlock += [HourGlass(ngf, ngf), HourGlass(ngf, ngf, False)]

        UpBlock += [
            nn.Pad2D([3, 3, 3, 3], 'reflect'),
            nn.Conv2D(3, output_nc, kernel_size=7, stride=1, bias_attr=False),
            nn.Tanh()
        ]

        self.DownBlock = nn.Sequential(*DownBlock)
        self.FC = nn.Sequential(*FC)
        self.UpBlock = nn.Sequential(*UpBlock)

    def forward(self, x):
        bs = x.shape[0]

        x = self.DownBlock(x)

        content_features = []
        for i in range(self.n_blocks):
            x = getattr(self, 'EncodeBlock' + str(i + 1))(x)
            content_features.append(
                F.adaptive_avg_pool2d(x, 1).reshape([bs, -1]))

        gap = F.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.reshape([bs, -1]))
        gap_weight = list(self.gap_fc.parameters())[0].transpose([1, 0])
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = F.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.reshape([bs, -1]))
        gmp_weight = list(self.gmp_fc.parameters())[0].transpose([1, 0])
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = paddle.concat([gap_logit, gmp_logit], 1)
        x = paddle.concat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))

        heatmap = paddle.sum(x, axis=1, keepdim=True)

        if self.light:
            x_ = F.adaptive_avg_pool2d(x, 1)
            style_features = self.FC(x_.reshape([bs, -1]))
        else:
            style_features = self.FC(x.reshape([bs, -1]))

        for i in range(self.n_blocks):
            x = getattr(self,
                        'DecodeBlock' + str(i + 1))(x,
                                                    content_features[4 - i - 1],
                                                    style_features)

        out = self.UpBlock(x)

        return out, cam_logit, heatmap


class ConvBlock(nn.Layer):
    def __init__(self, dim_in, dim_out):
        super(ConvBlock, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.conv_block1 = self.__convblock(dim_in, dim_out // 2)
        self.conv_block2 = self.__convblock(dim_out // 2, dim_out // 4)
        self.conv_block3 = self.__convblock(dim_out // 4, dim_out // 4)

        if self.dim_in != self.dim_out:
            self.conv_skip = nn.Sequential(
                nn.InstanceNorm2D(dim_in, weight_attr=False, bias_attr=False),
                nn.ReLU(),
                nn.Conv2D(dim_in,
                          dim_out,
                          kernel_size=1,
                          stride=1,
                          bias_attr=False))

    @staticmethod
    def __convblock(dim_in, dim_out):
        return nn.Sequential(
            nn.InstanceNorm2D(dim_in, weight_attr=False, bias_attr=False),
            nn.ReLU(), nn.Pad2D([1, 1, 1, 1], 'reflect'),
            nn.Conv2D(dim_in, dim_out, kernel_size=3, stride=1,
                      bias_attr=False))

    def forward(self, x):
        residual = x

        x1 = self.conv_block1(x)
        x2 = self.conv_block2(x1)
        x3 = self.conv_block3(x2)
        out = paddle.concat([x1, x2, x3], 1)

        if self.dim_in != self.dim_out:
            residual = self.conv_skip(residual)

        return residual + out


class HourGlassBlock(nn.Layer):
    def __init__(self, dim_in):
        super(HourGlassBlock, self).__init__()

        self.n_skip = 4
        self.n_block = 9

        for i in range(self.n_skip):
            setattr(self, 'ConvBlockskip' + str(i + 1),
                    ConvBlock(dim_in, dim_in))

        for i in range(self.n_block):
            setattr(self, 'ConvBlock' + str(i + 1), ConvBlock(dim_in, dim_in))

    def forward(self, x):
        skips = []
        for i in range(self.n_skip):
            skips.append(getattr(self, 'ConvBlockskip' + str(i + 1))(x))
            x = F.avg_pool2d(x, 2)
            x = getattr(self, 'ConvBlock' + str(i + 1))(x)

        x = self.ConvBlock5(x)

        for i in range(self.n_skip):
            x = getattr(self, 'ConvBlock' + str(i + 6))(x)
            x = F.upsample(x, scale_factor=2)
            x = skips[self.n_skip - i - 1] + x

        return x


class HourGlass(nn.Layer):
    def __init__(self, dim_in, dim_out, use_res=True):
        super(HourGlass, self).__init__()
        self.use_res = use_res

        self.HG = nn.Sequential(
            HourGlassBlock(dim_in), ConvBlock(dim_out, dim_out),
            nn.Conv2D(dim_out,
                      dim_out,
                      kernel_size=1,
                      stride=1,
                      bias_attr=False),
            nn.InstanceNorm2D(dim_out, weight_attr=False, bias_attr=False),
            nn.ReLU())

        self.Conv1 = nn.Conv2D(dim_out, 3, kernel_size=1, stride=1)

        if self.use_res:
            self.Conv2 = nn.Conv2D(dim_out, dim_out, kernel_size=1, stride=1)
            self.Conv3 = nn.Conv2D(3, dim_out, kernel_size=1, stride=1)

    def forward(self, x):
        ll = self.HG(x)
        tmp_out = self.Conv1(ll)

        if self.use_res:
            ll = self.Conv2(ll)
            tmp_out_ = self.Conv3(tmp_out)
            return x + ll + tmp_out_

        else:
            return tmp_out


class ResnetBlock(nn.Layer):
    def __init__(self, dim, use_bias=False):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [
            nn.Pad2D([1, 1, 1, 1], 'reflect'),
            nn.Conv2D(dim, dim, kernel_size=3, stride=1, bias_attr=use_bias),
            nn.InstanceNorm2D(dim, weight_attr=False, bias_attr=False),
            nn.ReLU()
        ]

        conv_block += [
            nn.Pad2D([1, 1, 1, 1], 'reflect'),
            nn.Conv2D(dim, dim, kernel_size=3, stride=1, bias_attr=use_bias),
            nn.InstanceNorm2D(dim, weight_attr=False, bias_attr=False)
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetSoftAdaLINBlock(nn.Layer):
    def __init__(self, dim, use_bias=False):
        super(ResnetSoftAdaLINBlock, self).__init__()
        self.pad1 = nn.Pad2D([1, 1, 1, 1], 'reflect')
        self.conv1 = nn.Conv2D(dim,
                               dim,
                               kernel_size=3,
                               stride=1,
                               bias_attr=use_bias)
        self.norm1 = SoftAdaLIN(dim)
        self.relu1 = nn.ReLU()

        self.pad2 = nn.Pad2D([1, 1, 1, 1], 'reflect')
        self.conv2 = nn.Conv2D(dim,
                               dim,
                               kernel_size=3,
                               stride=1,
                               bias_attr=use_bias)
        self.norm2 = SoftAdaLIN(dim)

    def forward(self, x, content_features, style_features):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, content_features, style_features)
        out = self.relu1(out)

        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, content_features, style_features)
        return out + x


class SoftAdaLIN(nn.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(SoftAdaLIN, self).__init__()
        self.norm = AdaLIN(num_features, eps)

        self.w_gamma = self.create_parameter(
            [1, num_features], default_initializer=nn.initializer.Constant(0.))
        self.w_beta = self.create_parameter(
            [1, num_features], default_initializer=nn.initializer.Constant(0.))

        self.c_gamma = nn.Sequential(
            nn.Linear(num_features, num_features, bias_attr=False), nn.ReLU(),
            nn.Linear(num_features, num_features, bias_attr=False))
        self.c_beta = nn.Sequential(
            nn.Linear(num_features, num_features, bias_attr=False), nn.ReLU(),
            nn.Linear(num_features, num_features, bias_attr=False))
        self.s_gamma = nn.Linear(num_features, num_features, bias_attr=False)
        self.s_beta = nn.Linear(num_features, num_features, bias_attr=False)

    def forward(self, x, content_features, style_features):
        content_gamma, content_beta = self.c_gamma(
            content_features), self.c_beta(content_features)
        style_gamma, style_beta = self.s_gamma(style_features), self.s_beta(
            style_features)

        w_gamma_, w_beta_ = self.w_gamma.expand(
            [x.shape[0], -1]), self.w_beta.expand([x.shape[0], -1])
        soft_gamma = (1. - w_gamma_) * style_gamma + w_gamma_ * content_gamma
        soft_beta = (1. - w_beta_) * style_beta + w_beta_ * content_beta

        out = self.norm(x, soft_gamma, soft_beta)
        return out


class AdaLIN(nn.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(AdaLIN, self).__init__()
        self.eps = eps
        self.rho = self.create_parameter(
            [1, num_features, 1, 1],
            default_initializer=nn.initializer.Constant(0.9))

    def forward(self, x, gamma, beta):
        in_mean, in_var = paddle.mean(x, axis=[2, 3],
                                      keepdim=True), paddle.var(x,
                                                                axis=[2, 3],
                                                                keepdim=True)
        out_in = (x - in_mean) / paddle.sqrt(in_var + self.eps)
        ln_mean, ln_var = paddle.mean(x, axis=[1, 2, 3],
                                      keepdim=True), paddle.var(x,
                                                                axis=[1, 2, 3],
                                                                keepdim=True)
        out_ln = (x - ln_mean) / paddle.sqrt(ln_var + self.eps)
        out = self.rho.expand([x.shape[0], -1, -1, -1]) * out_in + \
              (1-self.rho.expand([x.shape[0], -1, -1, -1])) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(
            2).unsqueeze(3)

        return out


class LIN(nn.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(LIN, self).__init__()
        self.eps = eps
        self.rho = self.create_parameter(
            [1, num_features, 1, 1],
            default_initializer=nn.initializer.Constant(0.))
        self.gamma = self.create_parameter(
            [1, num_features, 1, 1],
            default_initializer=nn.initializer.Constant(1.))
        self.beta = self.create_parameter(
            [1, num_features, 1, 1],
            default_initializer=nn.initializer.Constant(0.))

    def forward(self, x):
        in_mean, in_var = paddle.mean(x, axis=[2, 3],
                                      keepdim=True), paddle.var(x,
                                                                axis=[2, 3],
                                                                keepdim=True)
        out_in = (x - in_mean) / paddle.sqrt(in_var + self.eps)
        ln_mean, ln_var = paddle.mean(x, axis=[1, 2, 3],
                                      keepdim=True), paddle.var(x,
                                                                axis=[1, 2, 3],
                                                                keepdim=True)
        out_ln = (x - ln_mean) / paddle.sqrt(ln_var + self.eps)
        out = self.rho.expand([x.shape[0], -1, -1, -1]) * out_in + \
              (1-self.rho.expand([x.shape[0], -1, -1, -1])) * out_ln
        out = out * self.gamma.expand([x.shape[0], -1, -1, -1
                                       ]) + self.beta.expand(
                                           [x.shape[0], -1, -1, -1])

        return out
