# code was heavily based on https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py
# BSD 3-Clause License
# Copyright (c) Soumith Chintala 2016


import math
import paddle
import paddle.nn as nn
from paddle.nn import Conv2D, AvgPool2D, MaxPool2D, BatchNorm, Linear, AdaptiveAvgPool2D

__all__ = ['InceptionV3']


class InceptionV3(nn.Layer):
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {
        64: 0,  # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 class_dim=1000,
                 aux_logits=False,
                 resize_input=True,
                 normalize_input=True):
        super(InceptionV3, self).__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)
        self.class_dim = class_dim
        self.aux_logits = aux_logits

        assert self.last_needed_block <= 3, 'Last possible output block index is 3'
        self.blocks = []

        self.Conv2d_1a_3x3 = ConvBNLayer(3,
                                         32,
                                         3,
                                         stride=2,
                                         name='Conv2d_1a_3x3')
        self.Conv2d_2a_3x3 = ConvBNLayer(32, 32, 3, name='Conv2d_2a_3x3')
        self.Conv2d_2b_3x3 = ConvBNLayer(32,
                                         64,
                                         3,
                                         padding=1,
                                         name='Conv2d_2b_3x3')
        self.maxpool1 = MaxPool2D(kernel_size=3, stride=2)

        block0 = [
            self.Conv2d_1a_3x3, self.Conv2d_2a_3x3, self.Conv2d_2b_3x3,
            self.maxpool1
        ]
        self.blocks.append(nn.Sequential(*block0))
        ### block1

        if self.last_needed_block >= 1:
            self.Conv2d_3b_1x1 = ConvBNLayer(64, 80, 1, name='Conv2d_3b_1x1')
            self.Conv2d_4a_3x3 = ConvBNLayer(80, 192, 3, name='Conv2d_4a_3x3')
            self.maxpool2 = MaxPool2D(kernel_size=3, stride=2)
            block1 = [self.Conv2d_3b_1x1, self.Conv2d_4a_3x3, self.maxpool2]
            self.blocks.append(nn.Sequential(*block1))

        ### block2
        ### Mixed_5b 5c 5d
        if self.last_needed_block >= 2:
            self.Mixed_5b = Fid_inceptionA(192,
                                           pool_features=32,
                                           name='Mixed_5b')
            self.Mixed_5c = Fid_inceptionA(256,
                                           pool_features=64,
                                           name='Mixed_5c')
            self.Mixed_5d = Fid_inceptionA(288,
                                           pool_features=64,
                                           name='Mixed_5d')

            ### Mixed_6
            self.Mixed_6a = InceptionB(288, name='Mixed_6a')
            self.Mixed_6b = Fid_inceptionC(768, c7=128, name='Mixed_6b')
            self.Mixed_6c = Fid_inceptionC(768, c7=160, name='Mixed_6c')
            self.Mixed_6d = Fid_inceptionC(768, c7=160, name='Mixed_6d')
            self.Mixed_6e = Fid_inceptionC(768, c7=192, name='Mixed_6e')

            block2 = [
                self.Mixed_5b, self.Mixed_5c, self.Mixed_5d, self.Mixed_6a,
                self.Mixed_6b, self.Mixed_6c, self.Mixed_6d, self.Mixed_6e
            ]
            self.blocks.append(nn.Sequential(*block2))

        if self.aux_logits:
            self.AuxLogits = InceptionAux(768, self.class_dim, name='AuxLogits')
        ### block3
        ### Mixed_7
        if self.last_needed_block >= 3:
            self.Mixed_7a = InceptionD(768, name='Mixed_7a')
            self.Mixed_7b = Fid_inceptionE_1(1280, name='Mixed_7b')
            self.Mixed_7c = Fid_inceptionE_2(2048, name='Mixed_7c')
            self.avgpool = AdaptiveAvgPool2D(output_size=1)

            block3 = [self.Mixed_7a, self.Mixed_7b, self.Mixed_7c, self.avgpool]
            self.blocks.append(nn.Sequential(*block3))

    def forward(self, x):
        out = []
        aux = None
        if self.resize_input:
            x = nn.functional.interpolate(x, 
                                          size=[299, 299], 
                                          mode='bilinear', 
                                          align_corners=False, 
                                          align_mode=0)

        if self.normalize_input:
            x = x * 2 - 1

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if self.aux_logits and (idx == 2):
                aux = self.AuxLogits(x)
            if idx in self.output_blocks:
                out.append(x)
            if idx == self.last_needed_block:
                break

        return out, aux


class InceptionA(nn.Layer):
    def __init__(self, in_channels, pool_features, name=None):
        super(InceptionA, self).__init__()
        self.branch1x1 = ConvBNLayer(in_channels,
                                     64,
                                     1,
                                     name=name + '.branch1x1')

        self.branch5x5_1 = ConvBNLayer(in_channels,
                                       48,
                                       1,
                                       name=name + '.branch5x5_1')
        self.branch5x5_2 = ConvBNLayer(48,
                                       64,
                                       5,
                                       padding=2,
                                       name=name + '.branch5x5_2')

        self.branch3x3dbl_1 = ConvBNLayer(in_channels,
                                          64,
                                          1,
                                          name=name + '.branch3x3dbl_1')
        self.branch3x3dbl_2 = ConvBNLayer(64,
                                          96,
                                          3,
                                          padding=1,
                                          name=name + '.branch3x3dbl_2')
        self.branch3x3dbl_3 = ConvBNLayer(96,
                                          96,
                                          3,
                                          padding=1,
                                          name=name + '.branch3x3dbl_3')

        self.branch_pool0 = AvgPool2D(kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   exclusive=True)
        self.branch_pool = ConvBNLayer(in_channels,
                                       pool_features,
                                       1,
                                       name=name + '.branch_pool')

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.branch_pool0(x)
        branch_pool = self.branch_pool(branch_pool)
        return paddle.concat(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=1)


class InceptionB(nn.Layer):
    def __init__(self, in_channels, name=None):
        super(InceptionB, self).__init__()
        self.branch3x3 = ConvBNLayer(in_channels,
                                     384,
                                     3,
                                     stride=2,
                                     name=name + '.branch3x3')

        self.branch3x3dbl_1 = ConvBNLayer(in_channels,
                                          64,
                                          1,
                                          name=name + '.branch3x3dbl_1')
        self.branch3x3dbl_2 = ConvBNLayer(64,
                                          96,
                                          3,
                                          padding=1,
                                          name=name + '.branch3x3dbl_2')
        self.branch3x3dbl_3 = ConvBNLayer(96,
                                          96,
                                          3,
                                          stride=2,
                                          name=name + '.branch3x3dbl_3')

        self.branch_pool = MaxPool2D(kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.branch_pool(x)
        return paddle.concat([branch3x3, branch3x3dbl, branch_pool],
                                   axis=1)


class InceptionC(nn.Layer):
    def __init__(self, in_channels, c7, name=None):
        super(InceptionC, self).__init__()
        self.branch1x1 = ConvBNLayer(in_channels,
                                     192,
                                     1,
                                     name=name + '.branch1x1')

        self.branch7x7_1 = ConvBNLayer(in_channels,
                                       c7,
                                       1,
                                       name=name + '.branch7x7_1')
        self.branch7x7_2 = ConvBNLayer(c7,
                                       c7, (1, 7),
                                       padding=(0, 3),
                                       name=name + '.branch7x7_2')
        self.branch7x7_3 = ConvBNLayer(c7,
                                       192, (7, 1),
                                       padding=(3, 0),
                                       name=name + '.branch7x7_3')

        self.branch7x7dbl_1 = ConvBNLayer(in_channels,
                                          c7,
                                          1,
                                          name=name + '.branch7x7dbl_1')
        self.branch7x7dbl_2 = ConvBNLayer(c7,
                                          c7, (7, 1),
                                          padding=(3, 0),
                                          name=name + '.branch7x7dbl_2')
        self.branch7x7dbl_3 = ConvBNLayer(c7,
                                          c7, (1, 7),
                                          padding=(0, 3),
                                          name=name + '.branch7x7dbl_3')
        self.branch7x7dbl_4 = ConvBNLayer(c7,
                                          c7, (7, 1),
                                          padding=(3, 0),
                                          name=name + '.branch7x7dbl_4')
        self.branch7x7dbl_5 = ConvBNLayer(c7,
                                          192, (1, 7),
                                          padding=(0, 3),
                                          name=name + '.branch7x7dbl_5')

        self.branch_pool0 = AvgPool2D(kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   exclusive=True)
        self.branch_pool = ConvBNLayer(in_channels,
                                       192,
                                       1,
                                       name=name + '.branch_pool')

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = self.branch_pool0(x)
        branch_pool = self.branch_pool(branch_pool)

        return paddle.concat(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=1)


class InceptionD(nn.Layer):
    def __init__(self, in_channels, name=None):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = ConvBNLayer(in_channels,
                                       192,
                                       1,
                                       name=name + '.branch3x3_1')
        self.branch3x3_2 = ConvBNLayer(192,
                                       320,
                                       3,
                                       stride=2,
                                       name=name + '.branch3x3_2')

        self.branch7x7x3_1 = ConvBNLayer(in_channels,
                                         192,
                                         1,
                                         name=name + '.branch7x7x3_1')
        self.branch7x7x3_2 = ConvBNLayer(192,
                                         192, (1, 7),
                                         padding=(0, 3),
                                         name=name + '.branch7x7x3_2')
        self.branch7x7x3_3 = ConvBNLayer(192,
                                         192, (7, 1),
                                         padding=(3, 0),
                                         name=name + '.branch7x7x3_3')
        self.branch7x7x3_4 = ConvBNLayer(192,
                                         192,
                                         3,
                                         stride=2,
                                         name=name + '.branch7x7x3_4')

        self.branch_pool = MaxPool2D(kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = self.branch_pool(x)

        return paddle.concat([branch3x3, branch7x7x3, branch_pool],
                                   axis=1)


class InceptionE(nn.Layer):
    def __init__(self, in_channels, name=None):
        super(InceptionE, self).__init__()
        self.branch1x1 = ConvBNLayer(in_channels,
                                     320,
                                     1,
                                     name=name + '.branch1x1')

        self.branch3x3_1 = ConvBNLayer(in_channels,
                                       384,
                                       1,
                                       name=name + '.branch3x3_1')
        self.branch3x3_2a = ConvBNLayer(384,
                                        384, (1, 3),
                                        padding=(0, 1),
                                        name=name + '.branch3x3_2a')
        self.branch3x3_2b = ConvBNLayer(384,
                                        384, (3, 1),
                                        padding=(1, 0),
                                        name=name + '.branch3x3_2b')

        self.branch3x3dbl_1 = ConvBNLayer(in_channels,
                                          448,
                                          1,
                                          name=name + '.branch3x3dbl_1')
        self.branch3x3dbl_2 = ConvBNLayer(448,
                                          384,
                                          3,
                                          padding=1,
                                          name=name + '.branch3x3dbl_2')
        self.branch3x3dbl_3a = ConvBNLayer(384,
                                           384, (1, 3),
                                           padding=(0, 1),
                                           name=name + '.branch3x3dbl_3a')
        self.branch3x3dbl_3b = ConvBNLayer(384,
                                           384, (3, 1),
                                           padding=(1, 0),
                                           name=name + '.branch3x3dbl_3b')

        self.branch_pool0 = AvgPool2D(kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   exclusive=True)
        self.branch_pool = ConvBNLayer(in_channels,
                                       192,
                                       1,
                                       name=name + '.branch_pool')

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3_1 = self.branch3x3_1(x)
        branch3x3_2a = self.branch3x3_2a(branch3x3_1)
        branch3x3_2b = self.branch3x3_2b(branch3x3_1)
        branch3x3 = paddle.concat([branch3x3_2a, branch3x3_2b], axis=1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl_3a = self.branch3x3dbl_3a(branch3x3dbl)
        branch3x3dbl_3b = self.branch3x3dbl_3b(branch3x3dbl)
        branch3x3dbl = paddle.concat([branch3x3dbl_3a, branch3x3dbl_3b],
                                           axis=1)

        branch_pool = self.branch_pool0(x)
        branch_pool = self.branch_pool(branch_pool)

        return paddle.concat(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=1)


class InceptionAux(nn.Layer):
    def __init__(self, in_channels, num_classes, name=None):
        super(InceptionAux, self).__init__()
        self.num_classes = num_classes
        self.pool0 = AvgPool2D(kernel_size=5, stride=3)
        self.conv0 = ConvBNLayer(in_channels, 128, 1, name=name + '.conv0')
        self.conv1 = ConvBNLayer(128, 768, 5, name=name + '.conv1')
        self.pool1 = AvgPool2D(global_pooling=True)

    def forward(self, x):
        x = self.pool0(x)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = paddle.flatten(x, axis=1)
        x = paddle.static.nn.fc(x, size=self.num_classes)
        return x


class Fid_inceptionA(nn.Layer):
    """ FID block in inception v3
    """
    def __init__(self, in_channels, pool_features, name=None):
        super(Fid_inceptionA, self).__init__()
        self.branch1x1 = ConvBNLayer(in_channels,
                                     64,
                                     1,
                                     name=name + '.branch1x1')

        self.branch5x5_1 = ConvBNLayer(in_channels,
                                       48,
                                       1,
                                       name=name + '.branch5x5_1')
        self.branch5x5_2 = ConvBNLayer(48,
                                       64,
                                       5,
                                       padding=2,
                                       name=name + '.branch5x5_2')

        self.branch3x3dbl_1 = ConvBNLayer(in_channels,
                                          64,
                                          1,
                                          name=name + '.branch3x3dbl_1')
        self.branch3x3dbl_2 = ConvBNLayer(64,
                                          96,
                                          3,
                                          padding=1,
                                          name=name + '.branch3x3dbl_2')
        self.branch3x3dbl_3 = ConvBNLayer(96,
                                          96,
                                          3,
                                          padding=1,
                                          name=name + '.branch3x3dbl_3')

        self.branch_pool0 = AvgPool2D(kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   exclusive=True)
        self.branch_pool = ConvBNLayer(in_channels,
                                       pool_features,
                                       1,
                                       name=name + '.branch_pool')

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.branch_pool0(x)
        branch_pool = self.branch_pool(branch_pool)
        return paddle.concat(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=1)


class Fid_inceptionC(nn.Layer):
    """ FID block in inception v3
    """
    def __init__(self, in_channels, c7, name=None):
        super(Fid_inceptionC, self).__init__()
        self.branch1x1 = ConvBNLayer(in_channels,
                                     192,
                                     1,
                                     name=name + '.branch1x1')

        self.branch7x7_1 = ConvBNLayer(in_channels,
                                       c7,
                                       1,
                                       name=name + '.branch7x7_1')
        self.branch7x7_2 = ConvBNLayer(c7,
                                       c7, (1, 7),
                                       padding=(0, 3),
                                       name=name + '.branch7x7_2')
        self.branch7x7_3 = ConvBNLayer(c7,
                                       192, (7, 1),
                                       padding=(3, 0),
                                       name=name + '.branch7x7_3')

        self.branch7x7dbl_1 = ConvBNLayer(in_channels,
                                          c7,
                                          1,
                                          name=name + '.branch7x7dbl_1')
        self.branch7x7dbl_2 = ConvBNLayer(c7,
                                          c7, (7, 1),
                                          padding=(3, 0),
                                          name=name + '.branch7x7dbl_2')
        self.branch7x7dbl_3 = ConvBNLayer(c7,
                                          c7, (1, 7),
                                          padding=(0, 3),
                                          name=name + '.branch7x7dbl_3')
        self.branch7x7dbl_4 = ConvBNLayer(c7,
                                          c7, (7, 1),
                                          padding=(3, 0),
                                          name=name + '.branch7x7dbl_4')
        self.branch7x7dbl_5 = ConvBNLayer(c7,
                                          192, (1, 7),
                                          padding=(0, 3),
                                          name=name + '.branch7x7dbl_5')

        self.branch_pool0 = AvgPool2D(kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   exclusive=True)
        self.branch_pool = ConvBNLayer(in_channels,
                                       192,
                                       1,
                                       name=name + '.branch_pool')

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = self.branch_pool0(x)
        branch_pool = self.branch_pool(branch_pool)

        return paddle.concat(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=1)


class Fid_inceptionE_1(nn.Layer):
    """ FID block in inception v3
        """
    def __init__(self, in_channels, name=None):
        super(Fid_inceptionE_1, self).__init__()
        self.branch1x1 = ConvBNLayer(in_channels,
                                     320,
                                     1,
                                     name=name + '.branch1x1')

        self.branch3x3_1 = ConvBNLayer(in_channels,
                                       384,
                                       1,
                                       name=name + '.branch3x3_1')
        self.branch3x3_2a = ConvBNLayer(384,
                                        384, (1, 3),
                                        padding=(0, 1),
                                        name=name + '.branch3x3_2a')
        self.branch3x3_2b = ConvBNLayer(384,
                                        384, (3, 1),
                                        padding=(1, 0),
                                        name=name + '.branch3x3_2b')

        self.branch3x3dbl_1 = ConvBNLayer(in_channels,
                                          448,
                                          1,
                                          name=name + '.branch3x3dbl_1')
        self.branch3x3dbl_2 = ConvBNLayer(448,
                                          384,
                                          3,
                                          padding=1,
                                          name=name + '.branch3x3dbl_2')
        self.branch3x3dbl_3a = ConvBNLayer(384,
                                           384, (1, 3),
                                           padding=(0, 1),
                                           name=name + '.branch3x3dbl_3a')
        self.branch3x3dbl_3b = ConvBNLayer(384,
                                           384, (3, 1),
                                           padding=(1, 0),
                                           name=name + '.branch3x3dbl_3b')

        self.branch_pool0 = AvgPool2D(kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   exclusive=True)
        self.branch_pool = ConvBNLayer(in_channels,
                                       192,
                                       1,
                                       name=name + '.branch_pool')

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3_1 = self.branch3x3_1(x)
        branch3x3_2a = self.branch3x3_2a(branch3x3_1)
        branch3x3_2b = self.branch3x3_2b(branch3x3_1)
        branch3x3 = paddle.concat([branch3x3_2a, branch3x3_2b], axis=1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl_3a = self.branch3x3dbl_3a(branch3x3dbl)
        branch3x3dbl_3b = self.branch3x3dbl_3b(branch3x3dbl)
        branch3x3dbl = paddle.concat([branch3x3dbl_3a, branch3x3dbl_3b],
                                           axis=1)

        branch_pool = self.branch_pool0(x)
        branch_pool = self.branch_pool(branch_pool)

        return paddle.concat(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=1)


class Fid_inceptionE_2(nn.Layer):
    """ FID block in inception v3
    """
    def __init__(self, in_channels, name=None):
        super(Fid_inceptionE_2, self).__init__()
        self.branch1x1 = ConvBNLayer(in_channels,
                                     320,
                                     1,
                                     name=name + '.branch1x1')

        self.branch3x3_1 = ConvBNLayer(in_channels,
                                       384,
                                       1,
                                       name=name + '.branch3x3_1')
        self.branch3x3_2a = ConvBNLayer(384,
                                        384, (1, 3),
                                        padding=(0, 1),
                                        name=name + '.branch3x3_2a')
        self.branch3x3_2b = ConvBNLayer(384,
                                        384, (3, 1),
                                        padding=(1, 0),
                                        name=name + '.branch3x3_2b')

        self.branch3x3dbl_1 = ConvBNLayer(in_channels,
                                          448,
                                          1,
                                          name=name + '.branch3x3dbl_1')
        self.branch3x3dbl_2 = ConvBNLayer(448,
                                          384,
                                          3,
                                          padding=1,
                                          name=name + '.branch3x3dbl_2')
        self.branch3x3dbl_3a = ConvBNLayer(384,
                                           384, (1, 3),
                                           padding=(0, 1),
                                           name=name + '.branch3x3dbl_3a')
        self.branch3x3dbl_3b = ConvBNLayer(384,
                                           384, (3, 1),
                                           padding=(1, 0),
                                           name=name + '.branch3x3dbl_3b')
        ### same with paper
        self.branch_pool0 = MaxPool2D(kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.branch_pool = ConvBNLayer(in_channels,
                                       192,
                                       1,
                                       name=name + '.branch_pool')

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3_1 = self.branch3x3_1(x)
        branch3x3_2a = self.branch3x3_2a(branch3x3_1)
        branch3x3_2b = self.branch3x3_2b(branch3x3_1)
        branch3x3 = paddle.concat([branch3x3_2a, branch3x3_2b], axis=1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl_3a = self.branch3x3dbl_3a(branch3x3dbl)
        branch3x3dbl_3b = self.branch3x3dbl_3b(branch3x3dbl)
        branch3x3dbl = paddle.concat([branch3x3dbl_3a, branch3x3dbl_3b],
                                           axis=1)

        branch_pool = self.branch_pool0(x)
        branch_pool = self.branch_pool(branch_pool)

        return paddle.concat(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=1)


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 padding=0,
                 groups=1,
                 act='relu',
                 name=None):
        super(ConvBNLayer, self).__init__()
        self.conv = Conv2D(in_channels=in_channels,
                           out_channels=num_filters,
                           kernel_size=filter_size,
                           stride=stride,
                           padding=padding,
                           groups=groups,
                           weight_attr=paddle.ParamAttr(name=name + ".conv.weight"),
                           bias_attr=False)
        self.bn = BatchNorm(num_filters,
                            act=act,
                            epsilon=0.001,
                            param_attr=paddle.ParamAttr(name=name + ".bn.weight"),
                            bias_attr=paddle.ParamAttr(name=name + ".bn.bias"),
                            moving_mean_name=name + '.bn.running_mean',
                            moving_variance_name=name + '.bn.running_var')

    def forward(self, inputs):
        y = self.conv(inputs)
        y = self.bn(y)
        return y
