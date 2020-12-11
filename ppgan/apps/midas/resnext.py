# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
from paddle.vision.models.resnet import ResNet
from paddle.vision.models.resnet import BottleneckBlock

from paddle.utils.download import get_weights_path_from_url

__all__ = ['resnext101_32x8d_wsl']


class ResNetEx(ResNet):
    """ResNet extention model, support ResNeXt.
    """
    def __init__(self,
                 block,
                 depth,
                 num_classes=1000,
                 with_pool=True,
                 groups=1,
                 width_per_group=64):
        self.groups = groups
        self.base_width = width_per_group

        super(ResNetEx, self).__init__(block, depth, num_classes, with_pool)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes,
                          planes * block.expansion,
                          1,
                          stride=stride,
                          bias_attr=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      norm_layer=norm_layer))

        return nn.Sequential(*layers)


def _resnext(arch, Block, depth, **kwargs):
    model = ResNetEx(Block, depth, **kwargs)
    return model


def resnext101_32x8d_wsl(**kwargs):
    """ResNet101 32x8d wsl model
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnext('resnet101_32x8d', BottleneckBlock, 101, **kwargs)
