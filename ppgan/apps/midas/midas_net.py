# Refer https://github.com/intel-isl/MiDaS
"""MidashNet: Network for monocular depth estimation trained by mixing several datasets.
"""
import numpy as np
import paddle
import paddle.nn as nn

from .blocks import FeatureFusionBlock, _make_encoder


class BaseModel(paddle.nn.Layer):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = paddle.load(path)
        self.set_dict(parameters)


class MidasNet(BaseModel):
    """Network for monocular depth estimation.
    """
    def __init__(self, path=None, features=256, non_negative=True):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        print("Loading weights: ", path)

        super(MidasNet, self).__init__()

        use_pretrained = False if path is None else True

        self.pretrained, self.scratch = _make_encoder(
            backbone="resnext101_wsl",
            features=features,
            use_pretrained=use_pretrained)

        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        output_conv = [
            nn.Conv2D(features, 128, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2D(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU() if non_negative else nn.Identity(),
        ]
        if non_negative:
            output_conv.append(nn.ReLU())

        self.scratch.output_conv = nn.Sequential(*output_conv)

        if path:
            self.load(path)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return paddle.squeeze(out, axis=1)
