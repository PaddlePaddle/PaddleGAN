#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn
import paddle.nn.functional as F

from paddle.utils.download import get_weights_path_from_url
import numpy as np
import math

#resnet18_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
model_urls = {
    'resnet18': ('https://paddle-hapi.bj.bcebos.com/models/resnet18.pdparams',
                 '0ba53eea9bc970962d0ef96f7b94057e'),
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias_attr=False)


class BasicBlock(paddle.nn.Layer):
    def __init__(self, in_chan, out_chan, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm(out_chan)
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = nn.BatchNorm(out_chan)
        self.relu = nn.ReLU()
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan,
                          out_chan,
                          kernel_size=1,
                          stride=stride,
                          bias_attr=False),
                nn.BatchNorm(out_chan),
            )

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(self.bn1(residual))
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + residual
        out = self.relu(out)
        return out


def create_layer_basic(in_chan, out_chan, bnum, stride=1):
    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    for i in range(bnum - 1):
        layers.append(BasicBlock(out_chan, out_chan, stride=1))
    return nn.Sequential(*layers)


class Resnet18(paddle.nn.Layer):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.conv1 = nn.Conv2d(3,
                               64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias_attr=False)
        self.bn1 = nn.BatchNorm(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = create_layer_basic(128, 256, bnum=2, stride=2)
        self.layer4 = create_layer_basic(256, 512, bnum=2, stride=2)
        # self.init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.maxpool(x)

        x = self.layer1(x)
        feat8 = self.layer2(x)  # 1/8
        feat16 = self.layer3(feat8)  # 1/16
        feat32 = self.layer4(feat16)  # 1/32
        return feat8, feat16, feat32


def resnet18(pretrained=False, **kwargs):
    model = Resnet18()
    arch = 'resnet18'
    if pretrained:
        #weight_path = get_weights_path_from_url(model_urls[arch][0],
        #                                        model_urls[arch][1])
        #assert weight_path.endswith(
        #    '.pdparams'), "suffix of weight must be .pdparams"
        weight_path = './resnet.pdparams'
        param, _ = paddle.load(weight_path)
        model.set_dict(param)

    return model


if __name__ == "__main__":
    paddle.disable_static()
    net = resnet18(pretrained=True)
    x = paddle.to_tensor(
        np.random.uniform(0, 1, (16, 3, 224, 224)).astype(np.float32))
    out = net(x)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
