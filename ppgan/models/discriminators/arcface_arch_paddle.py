#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .builder import DISCRIMINATORS


def conv3x3(inplanes, outplanes, stride=1):
    """A simple wrapper for 3x3 convolution with padding.

    Args:
        inplanes (int): Channel number of inputs.
        outplanes (int): Channel number of outputs.
        stride (int): Stride in convolution. Default: 1.
    """
    return nn.Conv2D(inplanes,
                     outplanes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias_attr=False)


class BasicBlock(nn.Layer):
    """Basic residual block used in the ResNetArcFace architecture.

    Args:
        inplanes (int): Channel number of inputs.
        planes (int): Channel number of outputs.
        stride (int): Stride in convolution. Default: 1.
        downsample (nn.Module): The downsample module. Default: None.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2D(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2D(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class IRBlock(nn.Layer):
    """Improved residual block (IR Block) used in the ResNetArcFace architecture.

    Args:
        inplanes (int): Channel number of inputs.
        planes (int): Channel number of outputs.
        stride (int): Stride in convolution. Default: 1.
        downsample (nn.Module): The downsample module. Default: None.
        use_se (bool): Whether use the SEBlock (squeeze and excitation block). Default: True.
    """
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2D(inplanes)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2D(inplanes)
        self.prelu = PReLU_layer()
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2D(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.prelu(out)
        return out


class Bottleneck(nn.Layer):
    """Bottleneck block used in the ResNetArcFace architecture.

    Args:
        inplanes (int): Channel number of inputs.
        planes (int): Channel number of outputs.
        stride (int): Stride in convolution. Default: 1.
        downsample (nn.Module): The downsample module. Default: None.
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv3 = nn.Conv2D(planes,
                               planes * self.expansion,
                               kernel_size=1,
                               bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class PReLU_layer(nn.Layer):
    def __init__(self, init_value=0.25, num=1):
        super(PReLU_layer, self).__init__()
        x = self.create_parameter(
            attr=None,
            shape=[num],
            dtype=paddle.get_default_dtype(),
            is_bias=False,
            default_initializer=nn.initializer.Constant(init_value))
        self.add_parameter('weight', x)

    def forward(self, x):
        return F.prelu(x, self.weight)


class SEBlock(nn.Layer):
    """The squeeze-and-excitation block (SEBlock) used in the IRBlock.

    Args:
        channel (int): Channel number of inputs.
        reduction (int): Channel reduction ration. Default: 16.
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction),
                                nn.PReLU(),
                                nn.Linear(channel // reduction, channel),
                                nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def constant_init(param, **kwargs):
    initializer = nn.initializer.Constant(**kwargs)
    initializer(param, param.block)


@DISCRIMINATORS.register()
class ResNetArcFace(nn.Layer):
    """ArcFace with ResNet architectures.

    Ref: ArcFace: Additive Angular Margin Loss for Deep Face Recognition.

    Args:
        block (str): Block used in the ArcFace architecture.
        layers (tuple(int)): Block numbers in each layer.
        use_se (bool): Whether use the SEBlock (squeeze and excitation block). Default: True.
    """
    def __init__(self, block, layers, use_se=True, reprod_logger=None):
        if block == 'IRBlock':
            block = IRBlock
        self.inplanes = 64
        self.use_se = use_se
        super(ResNetArcFace, self).__init__()
        self.conv1 = nn.Conv2D(1, 64, kernel_size=3, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(64)
        self.maxpool = nn.MaxPool2D(kernel_size=2, stride=2)
        self.prelu = PReLU_layer()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn4 = nn.BatchNorm2D(512)
        self.dropout = nn.Dropout()
        self.fc5 = nn.Linear(512 * 8 * 8, 512)
        self.bn5 = nn.BatchNorm1D(512)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, paddle.nn.Conv2D):
            nn.initializer.XavierNormal(m.weight)
        elif isinstance(m, paddle.nn.BatchNorm2D) or isinstance(
                m, paddle.nn.BatchNorm1D):
            constant_init(m.weight, value=1.)
            constant_init(m.bias, value=0.)
        elif isinstance(m, paddle.nn.Linear):
            nn.initializer.XavierNormal(m.weight)
            constant_init(m.bias, value=0.)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias_attr=False),
                nn.BatchNorm2D(planes * block.expansion))
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample,
                  use_se=self.use_se))
        self.inplanes = planes
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn4(x)
        x = self.dropout(x)
        x = x.reshape([x.shape[0], -1])
        x = self.fc5(x)
        x = self.bn5(x)
        return x
