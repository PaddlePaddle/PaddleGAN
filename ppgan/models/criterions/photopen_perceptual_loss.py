import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.vision.models.vgg as vgg
from paddle import ParamAttr
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D

from ppgan.utils.download import get_path_from_url
from .builder import CRITERIONS

class ConvBlock(nn.Layer):
    def __init__(self, input_channels, output_channels, groups, name=None):
        super(ConvBlock, self).__init__()

        self.groups = groups
        self._conv_1 = Conv2D(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False)
        if groups == 2 or groups == 3 or groups == 4:
            self._conv_2 = Conv2D(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_attr=False)
        if groups == 3 or groups == 4:
            self._conv_3 = Conv2D(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_attr=False)
        if groups == 4:
            self._conv_4 = Conv2D(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_attr=False)

        self._pool = MaxPool2D(kernel_size=2, stride=2, padding=0)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        if self.groups == 2 or self.groups == 3 or self.groups == 4:
            x = self._conv_2(x)
            x = F.relu(x)
        if self.groups == 3 or self.groups == 4:
            x = self._conv_3(x)
            x = F.relu(x)
        if self.groups == 4:
            x = self._conv_4(x)
            x = F.relu(x)
        x = self._pool(x)
        return x

class VGG19(nn.Layer):
    def __init__(self, layers=19, class_dim=1000):
        super(VGG19, self).__init__()

        self.layers = layers
        self.vgg_configure = {
            11: [1, 1, 2, 2, 2],
            13: [2, 2, 2, 2, 2],
            16: [2, 2, 3, 3, 3],
            19: [2, 2, 4, 4, 4]
        }
        assert self.layers in self.vgg_configure.keys(), \
            "supported layers are {} but input layer is {}".format(
                vgg_configure.keys(), layers)
        self.groups = self.vgg_configure[self.layers]

        self._conv_block_1 = ConvBlock(3, 64, self.groups[0], name="conv1_")
        self._conv_block_2 = ConvBlock(64, 128, self.groups[1], name="conv2_")
        self._conv_block_3 = ConvBlock(128, 256, self.groups[2], name="conv3_")
        self._conv_block_4 = ConvBlock(256, 512, self.groups[3], name="conv4_")
        self._conv_block_5 = ConvBlock(512, 512, self.groups[4], name="conv5_")

        self._drop = Dropout(p=0.5, mode="downscale_in_infer")
        self._fc1 = Linear(
            7 * 7 * 512,
            4096,)
        self._fc2 = Linear(
            4096,
            4096,)
        self._out = Linear(
            4096,
            class_dim,)

    def forward(self, inputs):
        features = []
        features.append(inputs)
        x = self._conv_block_1(inputs)
        features.append(x)
        x = self._conv_block_2(x)
        features.append(x)
        x = self._conv_block_3(x)
        features.append(x)
        x = self._conv_block_4(x)
        features.append(x)
        x = self._conv_block_5(x)

        x = paddle.reshape(x, [0, -1])
        x = self._fc1(x)
        x = F.relu(x)
        x = self._drop(x)
        x = self._fc2(x)
        x = F.relu(x)
        x = self._drop(x)
        x = self._out(x)
        return x, features

@CRITERIONS.register()
class PhotoPenPerceptualLoss(nn.Layer):
    def __init__(self, 
                 crop_size, 
                 lambda_vgg, 
#                  pretrained='test/vgg19pretrain.pdparams',
                 pretrained='https://paddlegan.bj.bcebos.com/models/vgg19pretrain.pdparams',
                ):
        super(PhotoPenPerceptualLoss, self).__init__()
        self.model = VGG19()
        weight_path = get_path_from_url(pretrained)
        vgg_weight = paddle.load(weight_path)
        self.model.set_state_dict(vgg_weight)
        print('PerceptualVGG loaded pretrained weight.')
        self.rates = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.crop_size = crop_size
        self.lambda_vgg = lambda_vgg
        
    def forward(self, img_r, img_f):
        img_r = F.interpolate(img_r, (self.crop_size, self.crop_size))
        img_f = F.interpolate(img_f, (self.crop_size, self.crop_size))
        _, feat_r = self.model(img_r)
        _, feat_f = self.model(img_f)
        g_vggloss = paddle.to_tensor(0.)
        for i in range(len(feat_r)):
            g_vggloss += self.rates[i] * nn.L1Loss()(feat_r[i], feat_f[i])
        g_vggloss *= self.lambda_vgg
        
        return g_vggloss
