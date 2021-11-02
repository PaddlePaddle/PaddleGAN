#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .builder import GENERATORS


def make_multi_blocks(func, num_layers):
    """Make layers by stacking the same blocks.

    Args:
        func (nn.Layer): nn.Layer class for basic block.
        num_layers (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    Blocks = nn.Sequential()
    for i in range(num_layers):
        Blocks.add_sublayer('block%d' % i, func())
    return Blocks


class PA(nn.Layer):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2D(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = x * y

        return out


class PAConv(nn.Layer):
    def __init__(self, nf, k_size=3):

        super(PAConv, self).__init__()
        self.k2 = nn.Conv2D(nf, nf, 1)  # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2D(nf,
                            nf,
                            kernel_size=k_size,
                            padding=(k_size - 1) // 2,
                            bias_attr=False)  # 3x3 convolution
        self.k4 = nn.Conv2D(nf,
                            nf,
                            kernel_size=k_size,
                            padding=(k_size - 1) // 2,
                            bias_attr=False)  # 3x3 convolution

    def forward(self, x):

        y = self.k2(x)
        y = self.sigmoid(y)

        out = self.k3(x) * y
        out = self.k4(out)

        return out


class SCPA(nn.Layer):
    """
    SCPA is modified from SCNet (Jiang-Jiang Liu et al. Improving Convolutional Networks with Self-Calibrated Convolutions. In CVPR, 2020)
    """
    def __init__(self, nf, reduction=2, stride=1, dilation=1):
        super(SCPA, self).__init__()
        group_width = nf // reduction

        self.conv1_a = nn.Conv2D(nf,
                                 group_width,
                                 kernel_size=1,
                                 bias_attr=False)
        self.conv1_b = nn.Conv2D(nf,
                                 group_width,
                                 kernel_size=1,
                                 bias_attr=False)

        self.k1 = nn.Sequential(
            nn.Conv2D(group_width,
                      group_width,
                      kernel_size=3,
                      stride=stride,
                      padding=dilation,
                      dilation=dilation,
                      bias_attr=False))

        self.PAConv = PAConv(group_width)

        self.conv3 = nn.Conv2D(group_width * reduction,
                               nf,
                               kernel_size=1,
                               bias_attr=False)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        residual = x

        out_a = self.conv1_a(x)
        out_b = self.conv1_b(x)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out_a = self.k1(out_a)
        out_b = self.PAConv(out_b)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out = self.conv3(paddle.concat([out_a, out_b], axis=1))
        out += residual

        return out


@GENERATORS.register()
class PAN(nn.Layer):
    def __init__(self, in_nc, out_nc, nf, unf, nb, scale=4):
        super(PAN, self).__init__()
        # SCPA
        SCPA_block_f = functools.partial(SCPA, nf=nf, reduction=2)
        self.scale = scale

        ### first convolution
        self.conv_first = nn.Conv2D(in_nc, nf, 3, 1, 1)

        ### main blocks
        self.SCPA_trunk = make_multi_blocks(SCPA_block_f, nb)
        self.trunk_conv = nn.Conv2D(nf, nf, 3, 1, 1)

        #### upsampling
        self.upconv1 = nn.Conv2D(nf, unf, 3, 1, 1)
        self.att1 = PA(unf)
        self.HRconv1 = nn.Conv2D(unf, unf, 3, 1, 1)

        if self.scale == 4:
            self.upconv2 = nn.Conv2D(unf, unf, 3, 1, 1)
            self.att2 = PA(unf)
            self.HRconv2 = nn.Conv2D(unf, unf, 3, 1, 1)

        self.conv_last = nn.Conv2D(unf, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):

        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.SCPA_trunk(fea))
        fea = fea + trunk

        if self.scale == 2 or self.scale == 3:
            fea = self.upconv1(
                F.interpolate(fea, scale_factor=self.scale, mode='nearest'))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
        elif self.scale == 4:
            fea = self.upconv1(
                F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
            fea = self.upconv2(
                F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.att2(fea))
            fea = self.lrelu(self.HRconv2(fea))

        out = self.conv_last(fea)

        ILR = F.interpolate(x,
                            scale_factor=self.scale,
                            mode='bilinear',
                            align_corners=False)
        out = out + ILR
        return out
