# Refer https://github.com/intel-isl/MiDaS

import paddle
import paddle.nn as nn


def _make_encoder(backbone,
                  features,
                  use_pretrained,
                  groups=1,
                  expand=False,
                  exportable=True):
    if backbone == "resnext101_wsl":
        # resnext101_wsl
        pretrained = _make_pretrained_resnext101_wsl(use_pretrained)
        scratch = _make_scratch([256, 512, 1024, 2048],
                                features,
                                groups=groups,
                                expand=expand)
    else:
        print(f"Backbone '{backbone}' not implemented")
        assert False
    return pretrained, scratch


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Layer()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand == True:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2D(in_shape[0],
                                  out_shape1,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  bias_attr=False,
                                  groups=groups)
    scratch.layer2_rn = nn.Conv2D(in_shape[1],
                                  out_shape2,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  bias_attr=False,
                                  groups=groups)
    scratch.layer3_rn = nn.Conv2D(in_shape[2],
                                  out_shape3,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  bias_attr=False,
                                  groups=groups)
    scratch.layer4_rn = nn.Conv2D(in_shape[3],
                                  out_shape4,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  bias_attr=False,
                                  groups=groups)

    return scratch


def _make_resnet_backbone(resnet):
    pretrained = nn.Layer()
    pretrained.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                      resnet.maxpool, resnet.layer1)

    pretrained.layer2 = resnet.layer2
    pretrained.layer3 = resnet.layer3
    pretrained.layer4 = resnet.layer4

    return pretrained


def _make_pretrained_resnext101_wsl(use_pretrained):
    from .resnext import resnext101_32x8d_wsl
    resnet = resnext101_32x8d_wsl()
    return _make_resnet_backbone(resnet)


class ResidualConvUnit(nn.Layer):
    """Residual convolution module.
    """
    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2D(features,
                               features,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias_attr=True)

        self.conv2 = nn.Conv2D(features,
                               features,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias_attr=True)

        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        x = self.relu(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class FeatureFusionBlock(nn.Layer):
    """Feature fusion block.
    """
    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)
        output = nn.functional.interpolate(output,
                                           scale_factor=2,
                                           mode="bilinear",
                                           align_corners=True)

        return output
