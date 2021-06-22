#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
from ...utils.download import get_path_from_url

from .builder import GENERATORS


def calc_mean_std(feat, eps=1e-5):
    """calculate mean and standard deviation.

    Args:
        feat (Tensor): Tensor with shape (N, C, H, W).
        eps (float): Default: 1e-5.

    Return:
        mean and std of feat
        shape: [N, C, 1, 1]
    """
    size = feat.shape
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.reshape([N, C, -1])
    feat_var = paddle.var(feat_var, axis=2) + eps
    feat_std = paddle.sqrt(feat_var)
    feat_std = feat_std.reshape([N, C, 1, 1])
    feat_mean = feat.reshape([N, C, -1])
    feat_mean = paddle.mean(feat_mean, axis=2)
    feat_mean = feat_mean.reshape([N, C, 1, 1])
    return feat_mean, feat_std


def mean_variance_norm(feat):
    """mean_variance_norm.

    Args:
        feat (Tensor): Tensor with shape (N, C, H, W).

    Return:
        Normalized feat with shape (N, C, H, W)
    """
    size = feat.shape
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


def adaptive_instance_normalization(content_feat, style_feat):
    """adaptive_instance_normalization.

    Args:
        content_feat (Tensor): Tensor with shape (N, C, H, W).
        style_feat (Tensor): Tensor with shape (N, C, H, W).

    Return:
        Normalized content_feat with shape (N, C, H, W)
    """
    assert (content_feat.shape[:2] == style_feat.shape[:2])
    size = content_feat.shape
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat -
                       content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class ResnetBlock(nn.Layer):
    """Residual block.

    It has a style of:
        ---Pad-Conv-ReLU-Pad-Conv-+-
         |________________________|

    Args:
        dim (int): Channel number of intermediate features.
    """
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(nn.Pad2D([1, 1, 1, 1], mode='reflect'),
                                        nn.Conv2D(dim, dim, (3, 3)), nn.ReLU(),
                                        nn.Pad2D([1, 1, 1, 1], mode='reflect'),
                                        nn.Conv2D(dim, dim, (3, 3)))

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ConvBlock(nn.Layer):
    """convolution block.

    It has a style of:
        ---Pad-Conv-ReLU---

    Args:
        dim1 (int): Channel number of input features.
        dim2 (int): Channel number of output features.
    """
    def __init__(self, dim1, dim2):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(nn.Pad2D([1, 1, 1, 1], mode='reflect'),
                                        nn.Conv2D(dim1, dim2, (3, 3)),
                                        nn.ReLU())

    def forward(self, x):
        out = self.conv_block(x)
        return out


@GENERATORS.register()
class DecoderNet(nn.Layer):
    """Decoder of Drafting module.
    Paper:
        Drafting and Revision: Laplacian Pyramid Network for Fast High-Quality
        Artistic Style Transfer.
    """
    def __init__(self):
        super(DecoderNet, self).__init__()

        self.resblock_41 = ResnetBlock(512)
        self.convblock_41 = ConvBlock(512, 256)
        self.resblock_31 = ResnetBlock(256)
        self.convblock_31 = ConvBlock(256, 128)

        self.convblock_21 = ConvBlock(128, 128)
        self.convblock_22 = ConvBlock(128, 64)

        self.convblock_11 = ConvBlock(64, 64)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.final_conv = nn.Sequential(nn.Pad2D([1, 1, 1, 1], mode='reflect'),
                                        nn.Conv2D(64, 3, (3, 3)))

    def forward(self, cF, sF):

        out = adaptive_instance_normalization(cF['r41'], sF['r41'])
        out = self.resblock_41(out)
        out = self.convblock_41(out)

        out = self.upsample(out)
        out += adaptive_instance_normalization(cF['r31'], sF['r31'])
        out = self.resblock_31(out)
        out = self.convblock_31(out)

        out = self.upsample(out)
        out += adaptive_instance_normalization(cF['r21'], sF['r21'])
        out = self.convblock_21(out)
        out = self.convblock_22(out)

        out = self.upsample(out)
        out = self.convblock_11(out)
        out = self.final_conv(out)
        return out




@GENERATORS.register()
class Encoder(nn.Layer):
    """Encoder of Drafting module.
    Paper:
        Drafting and Revision: Laplacian Pyramid Network for Fast High-Quality
        Artistic Style Transfer.
    """
    def __init__(self):
        super(Encoder, self).__init__()
        vgg_net = nn.Sequential(
            nn.Conv2D(3, 3, (1, 1)),
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(3, 64, (3, 3)),
            nn.ReLU(),  # relu1-1
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(64, 64, (3, 3)),
            nn.ReLU(),  # relu1-2
            nn.MaxPool2D((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(64, 128, (3, 3)),
            nn.ReLU(),  # relu2-1
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(128, 128, (3, 3)),
            nn.ReLU(),  # relu2-2
            nn.MaxPool2D((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(128, 256, (3, 3)),
            nn.ReLU(),  # relu3-1
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-2
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-3
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-4
            nn.MaxPool2D((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(256, 512, (3, 3)),
            nn.ReLU(),  # relu4-1, this is the last layer used
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-2
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-3
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-4
            nn.MaxPool2D((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-1
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-2
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-3
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(512, 512, (3, 3)),
            nn.ReLU()  # relu5-4
        )
        weight_path = get_path_from_url(
            'https://paddlegan.bj.bcebos.com/models/vgg_normalised.pdparams')
        vgg_net.set_dict(paddle.load(weight_path))
        self.enc_1 = nn.Sequential(*list(
            vgg_net.children())[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*list(
            vgg_net.children())[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*list(
            vgg_net.children())[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*list(
            vgg_net.children())[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*list(
            vgg_net.children())[31:44])  # relu4_1 -> relu5_1

    def forward(self, x):
        out = {}
        x = self.enc_1(x)
        out['r11'] = x
        x = self.enc_2(x)
        out['r21'] = x
        x = self.enc_3(x)
        out['r31'] = x
        x = self.enc_4(x)
        out['r41'] = x
        x = self.enc_5(x)
        out['r51'] = x
        return out


@GENERATORS.register()
class RevisionNet(nn.Layer):
    """RevisionNet of Revision module.
    Paper:
        Drafting and Revision: Laplacian Pyramid Network for Fast High-Quality
        Artistic Style Transfer.
    """
    def __init__(self, input_nc=6):
        super(RevisionNet, self).__init__()
        DownBlock = []
        DownBlock += [
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(input_nc, 64, (3, 3)),
            nn.ReLU()
        ]
        DownBlock += [
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(64, 64, (3, 3), stride=2),
            nn.ReLU()
        ]

        self.resblock = ResnetBlock(64)

        UpBlock = []
        UpBlock += [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(64, 64, (3, 3)),
            nn.ReLU()
        ]
        UpBlock += [
            nn.Pad2D([1, 1, 1, 1], mode='reflect'),
            nn.Conv2D(64, 3, (3, 3))
        ]

        self.DownBlock = nn.Sequential(*DownBlock)
        self.UpBlock = nn.Sequential(*UpBlock)

    def forward(self, input):
        """
        Args:
            input (Tensor): (b, 6, 256, 256) is concat of last input and this lap.

        Returns:
            Tensor: (b, 3, 256, 256).
        """
        out = self.DownBlock(input)
        out = self.resblock(out)
        out = self.UpBlock(out)
        return out
