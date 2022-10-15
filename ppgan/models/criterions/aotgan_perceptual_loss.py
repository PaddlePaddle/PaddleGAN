#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.utils import spectral_norm

from ppgan.utils.download import get_path_from_url
from .builder import CRITERIONS

# VGG19（ImageNet pretrained）
class VGG19F(nn.Layer):
    def __init__(self):
        super(VGG19F, self).__init__()

        self.feature_0 = nn.Conv2D(3, 64, 3, 1, 1)
        self.relu_1 = nn.ReLU()
        self.feature_2 = nn.Conv2D(64, 64, 3, 1, 1)
        self.relu_3 = nn.ReLU()

        self.mp_4 = nn.MaxPool2D(2, 2, 0)
        self.feature_5 = nn.Conv2D(64, 128, 3, 1, 1)
        self.relu_6 = nn.ReLU()
        self.feature_7 = nn.Conv2D(128, 128, 3, 1, 1)
        self.relu_8 = nn.ReLU()

        self.mp_9 = nn.MaxPool2D(2, 2, 0)
        self.feature_10 = nn.Conv2D(128, 256, 3, 1, 1)
        self.relu_11 = nn.ReLU()
        self.feature_12 = nn.Conv2D(256, 256, 3, 1, 1)
        self.relu_13 = nn.ReLU()
        self.feature_14 = nn.Conv2D(256, 256, 3, 1, 1)
        self.relu_15 = nn.ReLU()
        self.feature_16 = nn.Conv2D(256, 256, 3, 1, 1)
        self.relu_17 = nn.ReLU()

        self.mp_18 = nn.MaxPool2D(2, 2, 0)
        self.feature_19 = nn.Conv2D(256, 512, 3, 1, 1)
        self.relu_20 = nn.ReLU()
        self.feature_21 = nn.Conv2D(512, 512, 3, 1, 1)
        self.relu_22 = nn.ReLU()
        self.feature_23 = nn.Conv2D(512, 512, 3, 1, 1)
        self.relu_24 = nn.ReLU()
        self.feature_25 = nn.Conv2D(512, 512, 3, 1, 1)
        self.relu_26 = nn.ReLU()

        self.mp_27 = nn.MaxPool2D(2, 2, 0)
        self.feature_28 = nn.Conv2D(512, 512, 3, 1, 1)
        self.relu_29 = nn.ReLU()
        self.feature_30 = nn.Conv2D(512, 512, 3, 1, 1)
        self.relu_31 = nn.ReLU()
        self.feature_32 = nn.Conv2D(512, 512, 3, 1, 1)
        self.relu_33 = nn.ReLU()
        self.feature_34 = nn.Conv2D(512, 512, 3, 1, 1)
        self.relu_35 = nn.ReLU()

    def forward(self, x):
        x = self.stand(x)
        feats = []
        group = []
        x = self.feature_0(x)
        x = self.relu_1(x)
        group.append(x)
        x = self.feature_2(x)
        x = self.relu_3(x)
        group.append(x)
        feats.append(group)

        group = []
        x = self.mp_4(x)
        x = self.feature_5(x)
        x = self.relu_6(x)
        group.append(x)
        x = self.feature_7(x)
        x = self.relu_8(x)
        group.append(x)
        feats.append(group)

        group = []
        x = self.mp_9(x)
        x = self.feature_10(x)
        x = self.relu_11(x)
        group.append(x)
        x = self.feature_12(x)
        x = self.relu_13(x)
        group.append(x)
        x = self.feature_14(x)
        x = self.relu_15(x)
        group.append(x)
        x = self.feature_16(x)
        x = self.relu_17(x)
        group.append(x)
        feats.append(group)

        group = []
        x = self.mp_18(x)
        x = self.feature_19(x)
        x = self.relu_20(x)
        group.append(x)
        x = self.feature_21(x)
        x = self.relu_22(x)
        group.append(x)
        x = self.feature_23(x)
        x = self.relu_24(x)
        group.append(x)
        x = self.feature_25(x)
        x = self.relu_26(x)
        group.append(x)
        feats.append(group)

        group = []
        x = self.mp_27(x)
        x = self.feature_28(x)
        x = self.relu_29(x)
        group.append(x)
        x = self.feature_30(x)
        x = self.relu_31(x)
        group.append(x)
        x = self.feature_32(x)
        x = self.relu_33(x)
        group.append(x)
        x = self.feature_34(x)
        x = self.relu_35(x)
        group.append(x)
        feats.append(group)

        return feats

    def stand(self, x):
        mean = paddle.to_tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1])
        std = paddle.to_tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1])
        y = (x + 1.) / 2.
        y = (y - mean) / std
        return y

# l1 loss
class L1():
    def __init__(self,):
        self.calc = nn.L1Loss()

    def __call__(self, x, y):
        return self.calc(x, y)

# perceptual loss
class Perceptual():
    def __init__(self, vgg, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(Perceptual, self).__init__()
        self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y, img_size):
        x = F.interpolate(x, (img_size, img_size), mode='bilinear', align_corners=True)
        y = F.interpolate(y, (img_size, img_size), mode='bilinear', align_corners=True)
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        content_loss = 0.0
        for i in range(len(self.weights)):
            content_loss += self.weights[i] * self.criterion(x_features[i][0], y_features[i][0]) # 此vgg19预训练模型无bn层，所以尝试不用rate
        return content_loss

# style loss
class Style():
    def __init__(self, vgg):
        super(Style, self).__init__()
        self.vgg = vgg
        self.criterion = nn.L1Loss()

    def compute_gram(self, x):
        b, c, h, w = x.shape
        f = x.reshape([b, c, w * h])
        f_T = f.transpose([0, 2, 1])
        G = paddle.matmul(f, f_T) / (h * w * c)
        return G

    def __call__(self, x, y, img_size):
        x = F.interpolate(x, (img_size, img_size), mode='bilinear', align_corners=True)
        y = F.interpolate(y, (img_size, img_size), mode='bilinear', align_corners=True)
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        style_loss = 0.0
        blocks = [2, 3, 4, 5]
        layers = [2, 4, 4, 2]
        for b, l in list(zip(blocks, layers)):
            b = b - 1
            l = l - 1
            style_loss += self.criterion(self.compute_gram(x_features[b][l]), self.compute_gram(y_features[b][l]))
        return style_loss

# sum of weighted losses
@CRITERIONS.register()
class AOTGANCriterionLoss(nn.Layer):
    def __init__(self,
                 pretrained,
                ):
        super(AOTGANCriterionLoss, self).__init__()
        self.model = VGG19F()
        weight_path = get_path_from_url(pretrained)
        vgg_weight = paddle.load(weight_path)
        self.model.set_state_dict(vgg_weight)
        print('PerceptualVGG loaded pretrained weight.')
        self.l1_loss = L1()
        self.perceptual_loss = Perceptual(self.model)
        self.style_loss = Style(self.model)

    def forward(self, img_r, img_f, img_size):
        l1_loss = self.l1_loss(img_r, img_f)
        perceptual_loss = self.perceptual_loss(img_r, img_f, img_size)
        style_loss = self.style_loss(img_r, img_f, img_size)

        return l1_loss, perceptual_loss, style_loss
