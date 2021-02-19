"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

Lines (19 to 80) were adapted from https://github.com/1adrianb/face-alignment
Lines (83 to 235) were adapted from https://github.com/protossw512/AdaptiveWingLoss
"""

from collections import namedtuple
from copy import deepcopy
from functools import partial

from munch import Munch
import numpy as np
import cv2
from skimage.filters import gaussian
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


from paddle.fluid.dygraph import layers
from paddle.framework import get_default_dtype, set_default_dtype
from paddle.fluid.initializer import Constant
from paddle.fluid.param_attr import ParamAttr
from paddle.nn.functional import batch_norm, instance_norm

from ppgan.models.generators.builder import GENERATORS

class _BatchNormBase(layers.Layer):
    """
    BatchNorm base .
    """

    def __init__(self,
                 num_features,
                 momentum=0.9,
                 epsilon=1e-05,
                 weight_attr=None,
                 bias_attr=None,
                 data_format='NCHW',
                 use_global_stats=None,
                 name=None):
        super(_BatchNormBase, self).__init__()
        self._num_features = num_features
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self._use_global_stats = use_global_stats

        if get_default_dtype() == 'float16':
            set_default_dtype('float32')

        param_shape = [num_features]

        # create parameter
        if weight_attr == False:
            self.weight = self.create_parameter(
                attr=None, shape=param_shape, default_initializer=Constant(1.0))
            self.weight.stop_gradient = True
        else:
            self.weight = self.create_parameter(
                attr=self._weight_attr,
                shape=param_shape,
                default_initializer=Constant(1.0))
            self.weight.stop_gradient = self._weight_attr != None and self._weight_attr.learning_rate == 0.

        if bias_attr == False:
            self.bias = self.create_parameter(
                attr=None,
                shape=param_shape,
                default_initializer=Constant(0.0),
                is_bias=True)
            self.bias.stop_gradient = True
        else:
            self.bias = self.create_parameter(
                attr=self._bias_attr, shape=param_shape, is_bias=True)
            self.bias.stop_gradient = self._bias_attr != None and self._bias_attr.learning_rate == 0.

        moving_mean_name = None
        moving_variance_name = None

        if name is not None:
            moving_mean_name = name + "_mean"
            moving_variance_name = name + "_variance"

        self.running_mean = self.create_parameter(
            attr=ParamAttr(
                name=moving_mean_name,
                initializer=Constant(0.0),
                trainable=False,
                do_model_average=True),
            shape=param_shape)
        self.running_mean.stop_gradient = True

        self.running_var = self.create_parameter(
            attr=ParamAttr(
                name=moving_variance_name,
                initializer=Constant(1.0),
                trainable=False,
                do_model_average=True),
            shape=param_shape)
        self.running_var.stop_gradient = True

        self._data_format = data_format
        self._in_place = False
        self._momentum = momentum
        self._epsilon = epsilon
        self._fuse_with_relu = False
        self._name = name

    def _check_input_dim(self, input):
        raise NotImplementedError("BatchNorm Base error")

    def _check_data_format(self, input):
        raise NotImplementedError("BatchNorm Base data format error")

    def forward(self, input):

        self._check_data_format(self._data_format)

        self._check_input_dim(input)

        # if self.training:
        #     print(
        #         "When training, we now always track global mean and variance.")

        return batch_norm(
            input,
            self.running_mean,
            self.running_var,
            weight=self.weight,
            bias=self.bias,
            training=self.training,
            momentum=self._momentum,
            epsilon=self._epsilon,
            data_format=self._data_format,
            use_global_stats=self._use_global_stats)


class BatchNorm2D(_BatchNormBase):

    def _check_data_format(self, input):
        if input == 'NCHW':
            self._data_format = input
        elif input == "NHWC":
            self._data_format = input
        else:
            raise ValueError('expected NCHW or NHWC for data_format input')

    def _check_input_dim(self, input):
        if len(input.shape) != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(
                len(input.shape)))


class _InstanceNormBase(layers.Layer):
    """
    This class is based class for InstanceNorm1D, 2d, 3d. 

    See InstaceNorm1D, InstanceNorm2D or InstanceNorm3D for more details.
    """

    def __init__(self,
                 num_features,
                 epsilon=1e-5,
                 momentum=0.9,
                 weight_attr=None,
                 bias_attr=None,
                 data_format="NCHW",
                 name=None):
        super(_InstanceNormBase, self).__init__()

        if weight_attr == False or bias_attr == False:
            assert weight_attr == bias_attr, "weight_attr and bias_attr must be set to Fasle at the same time in InstanceNorm"
        self._epsilon = epsilon
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr

        if weight_attr != False and bias_attr != False:
            self.weight = self.create_parameter(
                attr=self._weight_attr,
                shape=[num_features],
                default_initializer=Constant(1.0),
                is_bias=False)
            self.bias = self.create_parameter(
                attr=self._bias_attr,
                shape=[num_features],
                default_initializer=Constant(0.0),
                is_bias=True)
        else:
            self.weight = None
            self.bias = None

    def _check_input_dim(self, input):
        raise NotImplementedError("InstanceNorm Base error")

    def forward(self, input):
        self._check_input_dim(input)

        return instance_norm(
            input, weight=self.weight, bias=self.bias, eps=self._epsilon)


class InstanceNorm2D(_InstanceNormBase):

    def _check_input_dim(self, input):
        if len(input.shape) != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(
                len(input.shape)))



class HourGlass(nn.Layer):
    def __init__(self, num_modules, depth, num_features, first_one=False):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.coordconv = CoordConvTh(64, 64, True, True, 256, first_one,
                                     out_channels=256,
                                     kernel_size=1, stride=1, padding=0)
        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_sublayer('b1_' + str(level), ConvBlock(256, 256))
        self.add_sublayer('b2_' + str(level), ConvBlock(256, 256))
        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_sublayer('b2_plus_' + str(level), ConvBlock(256, 256))
        self.add_sublayer('b3_' + str(level), ConvBlock(256, 256))

    def _forward(self, level, inp):
        up1 = inp
        up1 = self._sub_layers['b1_' + str(level)](up1)
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._sub_layers['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._sub_layers['b2_plus_' + str(level)](low2)
        low3 = low2
        low3 = self._sub_layers['b3_' + str(level)](low3)
        up2 = F.interpolate(low3, scale_factor=2, mode='nearest')

        return up1 + up2

    def forward(self, x, heatmap):
        x, last_channel = self.coordconv(x, heatmap)
        return self._forward(self.depth, x), last_channel


class AddCoordsTh(nn.Layer):
    def __init__(self, height=64, width=64, with_r=False, with_boundary=False):
        super(AddCoordsTh, self).__init__()
        self.with_r = with_r
        self.with_boundary = with_boundary

        with paddle.no_grad():
            x_coords = paddle.arange(height).unsqueeze(1).expand((height, width)).astype('float32')
            y_coords = paddle.arange(width).unsqueeze(0).expand((height, width)).astype('float32')
            x_coords = (x_coords / (height - 1)) * 2 - 1
            y_coords = (y_coords / (width - 1)) * 2 - 1
            coords = paddle.stack([x_coords, y_coords], axis=0)  # (2, height, width)

            if self.with_r:
                rr = paddle.sqrt(paddle.pow(x_coords, 2) + paddle.pow(y_coords, 2))  # (height, width)
                rr = (rr / paddle.max(rr)).unsqueeze(0)
                coords = paddle.concat([coords, rr], axis=0)

            self.coords = coords.unsqueeze(0)  # (1, 2 or 3, height, width)
            self.x_coords = x_coords
            self.y_coords = y_coords

    def forward(self, x, heatmap=None):
        """
        x: (batch, c, x_dim, y_dim)
        """
        coords = self.coords.tile((x.shape[0], 1, 1, 1))

        if self.with_boundary and heatmap is not None:
            boundary_channel = paddle.clip(heatmap[:, -1:, :, :], 0.0, 1.0)
            zero_tensor = paddle.zeros_like(self.x_coords)
            xx_boundary_channel = paddle.where(boundary_channel > 0.05, self.x_coords, zero_tensor)
            yy_boundary_channel = paddle.where(boundary_channel > 0.05, self.y_coords, zero_tensor)
            coords = paddle.concat([coords, xx_boundary_channel, yy_boundary_channel], axis=1)

        x_and_coords = paddle.concat([x, coords], axis=1)
        return x_and_coords


class CoordConvTh(nn.Layer):
    """CoordConv layer as in the paper."""
    def __init__(self, height, width, with_r, with_boundary,
                 in_channels, first_one=False, *args, **kwargs):
        super(CoordConvTh, self).__init__()
        self.addcoords = AddCoordsTh(height, width, with_r, with_boundary)
        in_channels += 2
        if with_r:
            in_channels += 1
        if with_boundary and not first_one:
            in_channels += 2
        self.conv = nn.Conv2D(in_channels=in_channels, *args, **kwargs)

    def forward(self, input_tensor, heatmap=None):
        ret = self.addcoords(input_tensor, heatmap)
        last_channel = ret[:, -2:, :, :]
        ret = self.conv(ret)
        return ret, last_channel


class ConvBlock(nn.Layer):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = BatchNorm2D(in_planes)
        conv3x3 = partial(nn.Conv2D, kernel_size=3, stride=1, padding=1, bias_attr=False, dilation=1)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = BatchNorm2D(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = BatchNorm2D(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        self.downsample = None
        if in_planes != out_planes:
            self.downsample = nn.Sequential(BatchNorm2D(in_planes),
                                            nn.ReLU(True),
                                            nn.Conv2D(in_planes, out_planes, 1, 1, bias_attr=False))

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = paddle.concat((out1, out2, out3), 1)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out3 += residual
        return out3


@GENERATORS.register()
class FAN(nn.Layer):
    def __init__(self, num_modules=1, end_relu=False, num_landmarks=98, fname_pretrained=None):
        super(FAN, self).__init__()
        self.num_modules = num_modules
        self.end_relu = end_relu

        # Base part
        self.conv1 = CoordConvTh(256, 256, True, False,
                                 in_channels=3, out_channels=64,
                                 kernel_size=7, stride=2, padding=3)
        self.bn1 = BatchNorm2D(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        # Stacking part
        self.add_sublayer('m0', HourGlass(1, 4, 256, first_one=True))
        self.add_sublayer('top_m_0', ConvBlock(256, 256))
        self.add_sublayer('conv_last0', nn.Conv2D(256, 256, 1, 1, 0))
        self.add_sublayer('bn_end0', BatchNorm2D(256))
        self.add_sublayer('l0', nn.Conv2D(256, num_landmarks+1, 1, 1, 0))

        if fname_pretrained is not None:
            self.load_pretrained_weights(fname_pretrained)

    def load_pretrained_weights(self, fname):
        import pickle
        import six

        with open(fname, 'rb') as f:
            checkpoint = pickle.load(f) if six.PY2 else pickle.load(
                f, encoding='latin1')
        
        model_weights = self.state_dict()
        model_weights.update({k: v for k, v in checkpoint['state_dict'].items()
                              if k in model_weights})
        self.set_state_dict(model_weights)

    def forward(self, x):
        x, _ = self.conv1(x)
        x = F.relu(self.bn1(x), True)
        x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        outputs = []
        boundary_channels = []
        tmp_out = None
        ll, boundary_channel = self._sub_layers['m0'](x, tmp_out)
        ll = self._sub_layers['top_m_0'](ll)
        ll = F.relu(self._sub_layers['bn_end0']
                    (self._sub_layers['conv_last0'](ll)), True)

        # Predict heatmaps
        tmp_out = self._sub_layers['l0'](ll)
        if self.end_relu:
            tmp_out = F.relu(tmp_out)  # HACK: Added relu
        outputs.append(tmp_out)
        boundary_channels.append(boundary_channel)
        return outputs, boundary_channels

    @paddle.no_grad()
    def get_heatmap(self, x, b_preprocess=True):
        ''' outputs 0-1 normalized heatmap '''
        x = F.interpolate(x, size=[256, 256], mode='bilinear')
        x_01 = x*0.5 + 0.5
        outputs, _ = self(x_01)
        heatmaps = outputs[-1][:, :-1, :, :]
        scale_factor = x.shape[2] // heatmaps.shape[2]
        if b_preprocess:
            heatmaps = F.interpolate(heatmaps, scale_factor=scale_factor,
                                     mode='bilinear', align_corners=True)
            heatmaps = preprocess(heatmaps)
        return heatmaps


# ========================== #
#   Mask related functions   #
# ========================== #


def normalize(x, eps=1e-6):
    """Apply min-max normalization."""
    # x = x.contiguous()
    N, C, H, W = x.shape
    x_ = paddle.reshape(x, (N*C, -1))
    max_val = paddle.max(x_, axis=1, keepdim=True)[0]
    min_val = paddle.min(x_, axis=1, keepdim=True)[0]
    x_ = (x_ - min_val) / (max_val - min_val + eps)
    out = paddle.reshape(x_, (N, C, H, W))
    return out


def truncate(x, thres=0.1):
    """Remove small values in heatmaps."""
    return paddle.where(x < thres, paddle.zeros_like(x), x)


def resize(x, p=2):
    """Resize heatmaps."""
    return x**p


def shift(x, N):
    """Shift N pixels up or down."""
    x = x.numpy()
    up = N >= 0
    N = abs(N)
    _, _, H, W = x.shape
    head = np.arange(N)
    tail = np.arange(H-N)

    if up:
        head = np.arange(H-N)+N
        tail = np.arange(N)
    else:
        head = np.arange(N) + (H-N)
        tail = np.arange(H-N)

    # permutation indices
    perm = np.concatenate([head, tail])
    out = x[:, :, perm, :]
    out = paddle.to_tensor(out)
    return out


IDXPAIR = namedtuple('IDXPAIR', 'start end')
index_map = Munch(chin=IDXPAIR(0 + 8, 33 - 8),
                  eyebrows=IDXPAIR(33, 51),
                  eyebrowsedges=IDXPAIR(33, 46),
                  nose=IDXPAIR(51, 55),
                  nostrils=IDXPAIR(55, 60),
                  eyes=IDXPAIR(60, 76),
                  lipedges=IDXPAIR(76, 82),
                  lipupper=IDXPAIR(77, 82),
                  liplower=IDXPAIR(83, 88),
                  lipinner=IDXPAIR(88, 96))
OPPAIR = namedtuple('OPPAIR', 'shift resize')


def preprocess(x):
    """Preprocess 98-dimensional heatmaps."""
    N, C, H, W = x.shape
    x = truncate(x)
    x = normalize(x)

    sw = H // 256
    operations = Munch(chin=OPPAIR(0, 3),
                       eyebrows=OPPAIR(-7*sw, 2),
                       nostrils=OPPAIR(8*sw, 4),
                       lipupper=OPPAIR(-8*sw, 4),
                       liplower=OPPAIR(8*sw, 4),
                       lipinner=OPPAIR(-2*sw, 3))

    for part, ops in operations.items():
        start, end = index_map[part]
        x[:, start:end] = resize(shift(x[:, start:end], ops.shift), ops.resize)

    zero_out = paddle.concat([paddle.arange(0, index_map.chin.start),
                          paddle.arange(index_map.chin.end, 33),
                          paddle.to_tensor([index_map.eyebrowsedges.start,
                                            index_map.eyebrowsedges.end,
                                            index_map.lipedges.start,
                                            index_map.lipedges.end])])
    x = x.numpy()
    zero_out = zero_out.numpy()
    x[:, zero_out] = 0
    x = paddle.to_tensor(x)

    start, end = index_map.nose
    x[:, start+1:end] = shift(x[:, start+1:end], 4*sw)
    x[:, start:end] = resize(x[:, start:end], 1)

    start, end = index_map.eyes
    x[:, start:end] = resize(x[:, start:end], 1)
    x[:, start:end] = resize(shift(x[:, start:end], -8), 3) + \
        shift(x[:, start:end], -24)

    # Second-level mask
    x2 = deepcopy(x)
    x2[:, index_map.chin.start:index_map.chin.end] = 0  # start:end was 0:33
    x2[:, index_map.lipedges.start:index_map.lipinner.end] = 0  # start:end was 76:96
    x2[:, index_map.eyebrows.start:index_map.eyebrows.end] = 0  # start:end was 33:51

    x = paddle.sum(x, axis=1, keepdim=True)  # (N, 1, H, W)
    x2 = paddle.sum(x2, axis=1, keepdim=True)  # mask without faceline and mouth

    x = x.numpy()
    x2 = x2.numpy()
    x[x != x] = 0  # set nan to zero
    x2[x != x] = 0  # set nan to zero
    x = paddle.to_tensor(x)
    x2 = paddle.to_tensor(x2)
    return x.clip(0, 1), x2.clip(0, 1)