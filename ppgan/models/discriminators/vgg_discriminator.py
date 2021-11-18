# code was based on https://github.com/xinntao/ESRGAN

import paddle.nn as nn

from .builder import DISCRIMINATORS


@DISCRIMINATORS.register()
class VGGDiscriminator128(nn.Layer):
    """VGG style discriminator with input size 128 x 128.

    It is used to train SRGAN and ESRGAN.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features.
            Default: 64.
    """
    def __init__(self, in_channels, num_feat, norm_layer='batch'):
        super(VGGDiscriminator128, self).__init__()

        self.conv0_0 = nn.Conv2D(in_channels, num_feat, 3, 1, 1, bias_attr=True)
        self.conv0_1 = nn.Conv2D(num_feat, num_feat, 4, 2, 1, bias_attr=False)
        self.bn0_1 = nn.BatchNorm2D(num_feat)

        self.conv1_0 = nn.Conv2D(num_feat,
                                 num_feat * 2,
                                 3,
                                 1,
                                 1,
                                 bias_attr=False)
        self.bn1_0 = nn.BatchNorm2D(num_feat * 2)
        self.conv1_1 = nn.Conv2D(num_feat * 2,
                                 num_feat * 2,
                                 4,
                                 2,
                                 1,
                                 bias_attr=False)
        self.bn1_1 = nn.BatchNorm2D(num_feat * 2)

        self.conv2_0 = nn.Conv2D(num_feat * 2,
                                 num_feat * 4,
                                 3,
                                 1,
                                 1,
                                 bias_attr=False)
        self.bn2_0 = nn.BatchNorm2D(num_feat * 4)
        self.conv2_1 = nn.Conv2D(num_feat * 4,
                                 num_feat * 4,
                                 4,
                                 2,
                                 1,
                                 bias_attr=False)
        self.bn2_1 = nn.BatchNorm2D(num_feat * 4)

        self.conv3_0 = nn.Conv2D(num_feat * 4,
                                 num_feat * 8,
                                 3,
                                 1,
                                 1,
                                 bias_attr=False)
        self.bn3_0 = nn.BatchNorm2D(num_feat * 8)
        self.conv3_1 = nn.Conv2D(num_feat * 8,
                                 num_feat * 8,
                                 4,
                                 2,
                                 1,
                                 bias_attr=False)
        self.bn3_1 = nn.BatchNorm2D(num_feat * 8)

        self.conv4_0 = nn.Conv2D(num_feat * 8,
                                 num_feat * 8,
                                 3,
                                 1,
                                 1,
                                 bias_attr=False)
        self.bn4_0 = nn.BatchNorm2D(num_feat * 8)
        self.conv4_1 = nn.Conv2D(num_feat * 8,
                                 num_feat * 8,
                                 4,
                                 2,
                                 1,
                                 bias_attr=False)
        self.bn4_1 = nn.BatchNorm2D(num_feat * 8)

        self.linear1 = nn.Linear(num_feat * 8 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        assert x.shape[2] == 128 and x.shape[3] == 128, (
            f'Input spatial size must be 128x128, '
            f'but received {x.shape}.')

        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(
            self.conv0_1(feat)))  # output spatial size: (64, 64)

        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(
            self.conv1_1(feat)))  # output spatial size: (32, 32)

        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(
            self.conv2_1(feat)))  # output spatial size: (16, 16)

        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(
            self.conv3_1(feat)))  # output spatial size: (8, 8)

        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(
            self.conv4_1(feat)))  # output spatial size: (4, 4)

        feat = feat.reshape([feat.shape[0], -1])
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out
