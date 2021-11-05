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

# code was heavily based on https://github.com/eladrich/pixel2style2pixel
# MIT License
# Copyright (c) 2020 Elad Richardson, Yuval Alaluf

import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from collections import namedtuple

from .builder import GENERATORS
from .generator_styleganv2 import StyleGANv2Generator
from ...modules.equalized import EqualLinear


class Flatten(nn.Layer):
    def forward(self, input):
        return input.reshape((input.shape[0], -1))


def l2_norm(input, axis=1):
    norm = paddle.norm(input, 2, axis, True)
    output = paddle.div(input, norm)
    return output


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    """ A named tuple describing a ResNet block. """


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)
            ] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    else:
        raise ValueError(
            "Invalid number of layers: {}. Must be one of [50, 100, 152]".
            format(num_layers))
    return blocks


class SEModule(nn.Layer):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc1 = nn.Conv2D(channels,
                             channels // reduction,
                             kernel_size=1,
                             padding=0,
                             bias_attr=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2D(channels // reduction,
                             channels,
                             kernel_size=1,
                             padding=0,
                             bias_attr=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class BottleneckIR(nn.Layer):
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2D(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2D(in_channel, depth, (1, 1), stride, bias_attr=False),
                nn.BatchNorm2D(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2D(in_channel),
            nn.Conv2D(in_channel, depth, (3, 3), (1, 1), 1, bias_attr=False),
            nn.PReLU(depth),
            nn.Conv2D(depth, depth, (3, 3), stride, 1, bias_attr=False),
            nn.BatchNorm2D(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class BottleneckIRSE(nn.Layer):
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIRSE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2D(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2D(in_channel, depth, (1, 1), stride, bias_attr=False),
                nn.BatchNorm2D(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2D(in_channel),
            nn.Conv2D(in_channel, depth, (3, 3), (1, 1), 1, bias_attr=False),
            nn.PReLU(depth),
            nn.Conv2D(depth, depth, (3, 3), stride, 1, bias_attr=False),
            nn.BatchNorm2D(depth), SEModule(depth, 16))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class GradualStyleBlock(nn.Layer):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [
            nn.Conv2D(in_c, out_c, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        ]
        for i in range(num_pools - 1):
            modules += [
                nn.Conv2D(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.reshape((-1, self.out_c))
        x = self.linear(x)
        return x


class GradualStyleEncoder(nn.Layer):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [50, 100,
                              152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = BottleneckIR
        elif mode == 'ir_se':
            unit_module = BottleneckIRSE
        self.input_layer = nn.Sequential(
            nn.Conv2D(opts.input_nc, 64, (3, 3), 1, 1, bias_attr=False),
            nn.BatchNorm2D(64), nn.PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel, bottleneck.depth,
                                bottleneck.stride))
        self.body = nn.Sequential(*modules)

        self.styles = nn.LayerList()
        self.style_count = 18
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2D(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2D(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Tensor) top feature map to be upsampled.
          y: (Tensor) lateral feature map.
        Returns:
          (Tensor) added feature map.
        Note in Pypaddle, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.shape
        return F.interpolate(
            x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._sub_layers.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = paddle.stack(latents, 1)
        return out


class BackboneEncoderUsingLastLayerIntoW(nn.Layer):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoW, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoW')
        assert num_layers in [50, 100,
                              152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = BottleneckIR
        elif mode == 'ir_se':
            unit_module = BottleneckIRSE
        self.input_layer = nn.Sequential(
            nn.Conv2D(opts.input_nc, 64, (3, 3), 1, 1, bias_attr=False),
            nn.BatchNorm2D(64), nn.PReLU(64))
        self.output_pool = nn.AdaptiveAvgPool2D((1, 1))
        self.linear = EqualLinear(512, 512, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel, bottleneck.depth,
                                bottleneck.stride))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_pool(x)
        x = x.reshape((-1, 512))
        x = self.linear(x)
        return x


class BackboneEncoderUsingLastLayerIntoWPlus(nn.Layer):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoWPlus, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoWPlus')
        assert num_layers in [50, 100,
                              152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = BottleneckIR
        elif mode == 'ir_se':
            unit_module = BottleneckIRSE
        self.input_layer = nn.Sequential(
            nn.Conv2D(opts.input_nc, 64, (3, 3), 1, 1, bias_attr=False),
            nn.BatchNorm2D(64), nn.PReLU(64))
        self.output_layer_2 = nn.Sequential(nn.BatchNorm2D(512),
                                            nn.AdaptiveAvgPool2D((7, 7)),
                                            Flatten(),
                                            nn.Linear(512 * 7 * 7, 512))
        self.linear = EqualLinear(512, 512 * 18, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel, bottleneck.depth,
                                bottleneck.stride))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer_2(x)
        x = self.linear(x)
        x = x.reshape((-1, 18, 512))
        return x


@GENERATORS.register()
class Pixel2Style2Pixel(nn.Layer):
    def __init__(self, opts):
        super(Pixel2Style2Pixel, self).__init__()
        self.set_opts(opts)
        # Define architecture
        self.encoder = self.set_encoder()
        self.decoder = StyleGANv2Generator(opts.size, opts.style_dim,
                                           opts.n_mlp, opts.channel_multiplier)
        self.face_pool = nn.AdaptiveAvgPool2D((256, 256))
        self.style_dim = self.decoder.style_dim
        self.n_latent = self.decoder.n_latent
        if self.opts.start_from_latent_avg:
            if self.opts.learn_in_w:
                self.register_buffer('latent_avg',
                                     paddle.zeros([1, self.style_dim]))
            else:
                self.register_buffer(
                    'latent_avg',
                    paddle.zeros([1, self.n_latent, self.style_dim]))

    def set_encoder(self):
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = GradualStyleEncoder(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
            encoder = BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
            encoder = BackboneEncoderUsingLastLayerIntoWPlus(
                50, 'ir_se', self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(
                self.opts.encoder_type))
        return encoder

    def forward(self,
                x,
                resize=True,
                latent_mask=None,
                input_code=False,
                randomize_noise=True,
                inject_latent=None,
                return_latents=False,
                alpha=None):
        if input_code:
            codes = x
        else:
            codes = self.encoder(x)
            # normalize with respect to the center of an average face
            if self.opts.start_from_latent_avg:
                if self.opts.learn_in_w:
                    codes = codes + self.latent_avg.tile([codes.shape[0], 1])
                else:
                    codes = codes + self.latent_avg.tile([codes.shape[0], 1, 1])

        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (
                            1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        input_is_latent = not input_code
        images, result_latent = self.decoder([codes],
                                             input_is_latent=input_is_latent,
                                             randomize_noise=randomize_noise,
                                             return_latents=return_latents)

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, result_latent
        else:
            return images

    def set_opts(self, opts):
        self.opts = opts
