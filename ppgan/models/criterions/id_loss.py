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

import os
from collections import namedtuple

import paddle
import paddle.nn as nn
from paddle.vision.transforms import Resize

from .builder import CRITERIONS
from ppgan.utils.download import get_path_from_url

model_cfgs = {
    'model_urls':
    'https://paddlegan.bj.bcebos.com/models/model_ir_se50.pdparams',
}

class Flatten(nn.Layer):

    def forward(self, input):
        return paddle.reshape(input, [input.shape[0], -1])


def l2_norm(input, axis=1):
    norm = paddle.norm(input, 2, axis, True)
    output = paddle.divide(input, norm)
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


class bottleneck_IR(nn.Layer):

    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
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


class bottleneck_IR_SE(nn.Layer):

    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
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

"""
Modified Backbone implementation from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch)
"""

class Backbone(nn.Layer):

    def __init__(self,
                 input_size,
                 num_layers,
                 mode='ir',
                 drop_ratio=0.4,
                 affine=True):
        super(Backbone, self).__init__()
        assert input_size in [112, 224], "input_size should be 112 or 224"
        assert num_layers in [50, 100,
                              152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = paddle.nn.Sequential(
            nn.Conv2D(3, 64, (3, 3), 1, 1, bias_attr=False), nn.BatchNorm2D(64),
            nn.PReLU(64))
        if input_size == 112:
            self.output_layer = nn.Sequential(nn.BatchNorm2D(512),
                                              nn.Dropout(drop_ratio), Flatten(),
                                              nn.Linear(512 * 7 * 7, 512),
                                              nn.BatchNorm1D(512))
        else:
            self.output_layer = nn.Sequential(nn.BatchNorm2D(512),
                                              nn.Dropout(drop_ratio), Flatten(),
                                              nn.Linear(512 * 14 * 14, 512),
                                              nn.BatchNorm1D(512))

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
        x = self.output_layer(x)
        return l2_norm(x)


@CRITERIONS.register()
class IDLoss(paddle.nn.Layer):

    def __init__(self, base_dir='./'):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112,
                                num_layers=50,
                                drop_ratio=0.6,
                                mode='ir_se')

        facenet_weights_path = os.path.join(base_dir, 'data/gpen/weights',
                                            'model_ir_se50.pdparams')

        if not os.path.isfile(facenet_weights_path):
            facenet_weights_path = get_path_from_url(model_cfgs['model_urls'])

        self.facenet.load_dict(paddle.load(facenet_weights_path))

        self.face_pool = paddle.nn.AdaptiveAvgPool2D((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        _, _, h, w = x.shape
        assert h == w
        ss = h // 256
        x = x[:, :, 35 * ss:-33 * ss, 32 * ss:-36 * ss]
        transform = Resize(size=(112, 112))

        for num in range(x.shape[0]):
            mid_feats = transform(x[num]).unsqueeze(0)
            if num == 0:
                x_feats = mid_feats
            else:
                x_feats = paddle.concat([x_feats, mid_feats], axis=0)

        x_feats = self.facenet(x_feats)
        return x_feats

    def forward(self, y_hat, y, x):
        n_samples = x.shape[0]
        y_feats = self.extract_feats(y)
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target
            count += 1

        return loss / count
