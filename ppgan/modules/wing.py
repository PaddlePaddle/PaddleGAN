# code was heavily based on https://github.com/clovaai/stargan-v2
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/clovaai/stargan-v2#license

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

from ppgan.models.generators.builder import GENERATORS


class HourGlass(nn.Layer):
    def __init__(self, num_modules, depth, num_features, first_one=False):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.coordconv = CoordConvTh(64,
                                     64,
                                     True,
                                     True,
                                     256,
                                     first_one,
                                     out_channels=256,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
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
            x_coords = paddle.arange(height).unsqueeze(1).expand(
                (height, width)).astype('float32')
            y_coords = paddle.arange(width).unsqueeze(0).expand(
                (height, width)).astype('float32')
            x_coords = (x_coords / (height - 1)) * 2 - 1
            y_coords = (y_coords / (width - 1)) * 2 - 1
            coords = paddle.stack([x_coords, y_coords],
                                  axis=0)  # (2, height, width)

            if self.with_r:
                rr = paddle.sqrt(
                    paddle.pow(x_coords, 2) +
                    paddle.pow(y_coords, 2))  # (height, width)
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
            xx_boundary_channel = paddle.where(boundary_channel > 0.05,
                                               self.x_coords, zero_tensor)
            yy_boundary_channel = paddle.where(boundary_channel > 0.05,
                                               self.y_coords, zero_tensor)
            coords = paddle.concat(
                [coords, xx_boundary_channel, yy_boundary_channel], axis=1)

        x_and_coords = paddle.concat([x, coords], axis=1)
        return x_and_coords


class CoordConvTh(nn.Layer):
    """CoordConv layer as in the paper."""
    def __init__(self,
                 height,
                 width,
                 with_r,
                 with_boundary,
                 in_channels,
                 first_one=False,
                 *args,
                 **kwargs):
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
        self.bn1 = nn.BatchNorm2D(in_planes)
        conv3x3 = partial(nn.Conv2D,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias_attr=False,
                          dilation=1)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2D(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.BatchNorm2D(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        self.downsample = None
        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2D(in_planes), nn.ReLU(True),
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


# ========================== #
#   Mask related functions   #
# ========================== #


def normalize(x, eps=1e-6):
    """Apply min-max normalization."""
    # x = x.contiguous()
    N, C, H, W = x.shape
    x_ = paddle.reshape(x, (N * C, -1))
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
    tail = np.arange(H - N)

    if up:
        head = np.arange(H - N) + N
        tail = np.arange(N)
    else:
        head = np.arange(N) + (H - N)
        tail = np.arange(H - N)

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
                       eyebrows=OPPAIR(-7 * sw, 2),
                       nostrils=OPPAIR(8 * sw, 4),
                       lipupper=OPPAIR(-8 * sw, 4),
                       liplower=OPPAIR(8 * sw, 4),
                       lipinner=OPPAIR(-2 * sw, 3))

    for part, ops in operations.items():
        start, end = index_map[part]
        x[:, start:end] = resize(shift(x[:, start:end], ops.shift), ops.resize)

    zero_out = paddle.concat([
        paddle.arange(0, index_map.chin.start),
        paddle.arange(index_map.chin.end, 33),
        paddle.to_tensor([
            index_map.eyebrowsedges.start, index_map.eyebrowsedges.end,
            index_map.lipedges.start, index_map.lipedges.end
        ])
    ])
    x = x.numpy()
    zero_out = zero_out.numpy()
    x[:, zero_out] = 0
    x = paddle.to_tensor(x)

    start, end = index_map.nose
    x[:, start + 1:end] = shift(x[:, start + 1:end], 4 * sw)
    x[:, start:end] = resize(x[:, start:end], 1)

    start, end = index_map.eyes
    x[:, start:end] = resize(x[:, start:end], 1)
    x[:, start:end] = resize(shift(x[:, start:end], -8), 3) + \
        shift(x[:, start:end], -24)

    # Second-level mask
    x2 = deepcopy(x)
    x2[:, index_map.chin.start:index_map.chin.end] = 0  # start:end was 0:33
    x2[:, index_map.lipedges.start:index_map.lipinner.
       end] = 0  # start:end was 76:96
    x2[:, index_map.eyebrows.start:index_map.eyebrows.
       end] = 0  # start:end was 33:51

    x = paddle.sum(x, axis=1, keepdim=True)  # (N, 1, H, W)
    x2 = paddle.sum(x2, axis=1, keepdim=True)  # mask without faceline and mouth

    x = x.numpy()
    x2 = x2.numpy()
    x[x != x] = 0  # set nan to zero
    x2[x != x] = 0  # set nan to zero
    x = paddle.to_tensor(x)
    x2 = paddle.to_tensor(x2)
    return x.clip(0, 1), x2.clip(0, 1)
