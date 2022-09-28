# code was heavily based on https://github.com/cszn/KAIR
# MIT License
# Copyright (c) 2019 Kai Zhang
"""
Droppath, reimplement from https://github.com/yueatsprograms/Stochastic_Depth
"""
from itertools import repeat
import collections.abc
import math
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .builder import GENERATORS


def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


class DropPath(nn.Layer):
    """DropPath class"""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def drop_path(self, inputs):
        """drop path op
        Args:
            input: tensor with arbitrary shape
            drop_prob: float number of drop path probability, default: 0.0
            training: bool, if current mode is training, default: False
        Returns:
            output: output tensor after drop path
        """
        # if prob is 0 or eval mode, return original input
        if self.drop_prob == 0. or not self.training:
            return inputs
        keep_prob = 1 - self.drop_prob
        keep_prob = paddle.to_tensor(keep_prob, dtype='float32')
        shape = (
            inputs.shape[0], ) + (1, ) * (inputs.ndim - 1)  # shape=(N, 1, 1, 1)
        random_tensor = keep_prob + paddle.rand(shape, dtype=inputs.dtype)
        random_tensor = random_tensor.floor()  # mask
        output = inputs.divide(
            keep_prob
        ) * random_tensor  # divide is to keep same output expectation
        return output

    def forward(self, inputs):
        return self.drop_path(inputs)


to_2tuple = _ntuple(2)


@paddle.jit.not_to_static
def swapdim(x, dim1, dim2):
    a = list(range(len(x.shape)))
    a[dim1], a[dim2] = a[dim2], a[dim1]
    return x.transpose(a)


class Identity(nn.Layer):
    """ Identity layer
    The output of this layer is the input without any change.
    Use this layer to avoid if condition in some forward methods
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Mlp(nn.Layer):

    def __init__(self, in_features, hidden_features, dropout):
        super(Mlp, self).__init__()
        w_attr_1, b_attr_1 = self._init_weights()
        self.fc1 = nn.Linear(in_features,
                             hidden_features,
                             weight_attr=w_attr_1,
                             bias_attr=b_attr_1)

        w_attr_2, b_attr_2 = self._init_weights()
        self.fc2 = nn.Linear(hidden_features,
                             in_features,
                             weight_attr=w_attr_2,
                             bias_attr=b_attr_2)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.XavierUniform())
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Normal(
            std=1e-6))
        return weight_attr, bias_attr

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class WindowAttention(nn.Layer):
    """Window based multihead attention, with relative position bias.
    Both shifted window and non-shifted window are supported.
    Args:
        dim (int): input dimension (channels)
        window_size (int): height and width of the window
        num_heads (int): number of attention heads
        qkv_bias (bool): if True, enable learnable bias to q,k,v, default: True
        qk_scale (float): override default qk scale head_dim**-0.5 if set, default: None
        attention_dropout (float): dropout of attention
        dropout (float): dropout for output
    """

    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attention_dropout=0.,
                 dropout=0.):
        super(WindowAttention, self).__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.dim = dim
        self.dim_head = dim // num_heads
        self.scale = qk_scale or self.dim_head**-0.5

        self.relative_position_bias_table = paddle.create_parameter(
            shape=[(2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                   num_heads],
            dtype='float32',
            default_initializer=paddle.nn.initializer.TruncatedNormal(std=.02))

        weight_attr, bias_attr = self._init_weights()

        # relative position index for each token inside window
        coords_h = paddle.arange(self.window_size[0])
        coords_w = paddle.arange(self.window_size[1])
        coords = paddle.stack(paddle.meshgrid([coords_h, coords_w
                                               ]))  # [2, window_h, window_w]
        coords_flatten = paddle.flatten(coords, 1)  # [2, window_h * window_w]
        # 2, window_h * window_w, window_h * window_h
        relative_coords = coords_flatten.unsqueeze(
            2) - coords_flatten.unsqueeze(1)
        # winwod_h*window_w, window_h*window_w, 2
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # [window_size * window_size, window_size*window_size]
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim,
                             dim * 3,
                             weight_attr=weight_attr,
                             bias_attr=bias_attr if qkv_bias else False)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim,
                              dim,
                              weight_attr=weight_attr,
                              bias_attr=bias_attr)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(axis=-1)

    def transpose_multihead(self, x):
        tensor_shape = list(x.shape[:-1])
        new_shape = tensor_shape + [self.num_heads, self.dim_head]
        x = x.reshape(new_shape)
        x = x.transpose([0, 2, 1, 3])
        return x

    def get_relative_pos_bias_from_pos_index(self):
        # relative_position_bias_table is a ParamBase object
        # https://github.com/PaddlePaddle/Paddle/blob/067f558c59b34dd6d8626aad73e9943cf7f5960f/python/paddle/fluid/framework.py#L5727
        table = self.relative_position_bias_table  # N x num_heads
        # index is a tensor
        index = self.relative_position_index.reshape(
            [-1])  # window_h*window_w * window_h*window_w
        # NOTE: paddle does NOT support indexing Tensor by a Tensor
        relative_position_bias = paddle.index_select(x=table, index=index)
        return relative_position_bias

    def forward(self, x, mask=None):
        qkv = self.qkv(x).chunk(3, axis=-1)
        q, k, v = map(self.transpose_multihead, qkv)
        q = q * self.scale
        attn = paddle.matmul(q, k, transpose_y=True)
        relative_position_bias = self.get_relative_pos_bias_from_pos_index()
        relative_position_bias = relative_position_bias.reshape([
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1
        ])
        # nH, window_h*window_w, window_h*window_w
        relative_position_bias = relative_position_bias.transpose([2, 0, 1])
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(
                [x.shape[0] // nW, nW, self.num_heads, x.shape[1], x.shape[1]])
            attn += mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape([-1, self.num_heads, x.shape[1], x.shape[1]])
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_dropout(attn)

        z = paddle.matmul(attn, v)
        z = z.transpose([0, 2, 1, 3])
        tensor_shape = list(z.shape[:-2])
        new_shape = tensor_shape + [self.dim]
        z = z.reshape(new_shape)
        z = self.proj(z)
        z = self.proj_dropout(z)

        return z

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0))
        return weight_attr, bias_attr

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops


def windows_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.reshape(
        [B, H // window_size, window_size, W // window_size, window_size, C])
    windows = x.transpose([0, 1, 3, 2, 4,
                           5]).reshape([-1, window_size, window_size, C])
    return windows


def windows_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(
        [B, H // window_size, W // window_size, window_size, window_size, -1])
    x = x.transpose([0, 1, 3, 2, 4, 5]).reshape([B, H, W, -1])
    return x


class SwinTransformerBlock(nn.Layer):
    """Swin transformer block
    Contains window multi head self attention, droppath, mlp, norm and residual.
    Attributes:
        dim: int, input dimension (channels)
        input_resolution: int, input resoultion
        num_heads: int, number of attention heads
        windos_size: int, window size, default: 7
        shift_size: int, shift size for SW-MSA, default: 0
        mlp_ratio: float, ratio of mlp hidden dim and input embedding dim, default: 4.
        qkv_bias: bool, if True, enable learnable bias to q,k,v, default: True
        qk_scale: float, override default qk scale head_dim**-0.5 if set, default: None
        dropout: float, dropout for output, default: 0.
        attention_dropout: float, dropout of attention, default: 0.
        droppath: float, drop path rate, default: 0.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim,
                                    window_size=to_2tuple(self.window_size),
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    attention_dropout=attention_dropout,
                                    dropout=dropout)
        self.drop_path = DropPath(droppath) if droppath > 0. else Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       dropout=dropout)

        attn_mask = self.calculate_mask(self.input_resolution)

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = x_size
            img_mask = paddle.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0

            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = windows_partition(
                img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.reshape(
                [-1, self.window_size * self.window_size])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)

            huns = -100.0 * paddle.ones_like(attn_mask)
            attn_mask = huns * (attn_mask != 0).astype("float32")

            return attn_mask
        else:
            return None

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.reshape([B, H, W, C])

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = paddle.roll(x,
                                    shifts=(-self.shift_size, -self.shift_size),
                                    axis=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = windows_partition(shifted_x, self.window_size)
        x_windows = x_windows.reshape(
            [-1, self.window_size * self.window_size, C])

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size

        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(x_windows,
                                     mask=self.calculate_mask(x_size))

        # merge windows
        attn_windows = attn_windows.reshape(
            [-1, self.window_size, self.window_size, C])
        shifted_x = windows_reverse(attn_windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = paddle.roll(shifted_x,
                            shifts=(self.shift_size, self.shift_size),
                            axis=(1, 2))
        else:
            x = shifted_x

        x = x.reshape([B, H * W, C])

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Layer):
    """ Patch Merging class
    Merge multiple patch into one path and keep the out dim.
    Spefically, merge adjacent 2x2 patches(dim=C) into 1 patch.
    The concat dim 4*C is rescaled to 2*C
    Args:
        input_resolution (tuple | ints): the size of input
        dim: dimension of single patch
        reduction: nn.Linear which maps 4C to 2C dim
        norm: nn.LayerNorm, applied after linear layer.
    """

    def __init__(self, input_resolution, dim):
        super(PatchMerging, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias_attr=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        h, w = self.input_resolution
        b, _, c = x.shape
        x = x.reshape([b, h, w, c])

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = paddle.concat([x0, x1, x2, x3], -1)  #[B, H/2, W/2, 4*C]
        x = x.reshape([b, -1, 4 * c])  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Layer):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        dropout (float, optional): Dropout rate. Default: 0.0
        attention_dropout (float, optional): Attention dropout rate. Default: 0.0
        droppath (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        downsample (nn.Layer | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.,
                 downsample=None):
        super(BasicLayer, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.LayerList()
        for i in range(depth):
            self.blocks.append(
                SwinTransformerBlock(dim=dim,
                                     input_resolution=input_resolution,
                                     num_heads=num_heads,
                                     window_size=window_size,
                                     shift_size=0 if
                                     (i % 2 == 0) else window_size // 2,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     dropout=dropout,
                                     attention_dropout=attention_dropout,
                                     droppath=droppath[i] if isinstance(
                                         droppath, list) else droppath))

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for block in self.blocks:
            x = block(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RSTB(nn.Layer):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        downsample (nn.Layer | None, optional): Downsample layer at the end of the layer. Default: None
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 downsample=None,
                 img_size=224,
                 patch_size=4,
                 resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias,
                                         qk_scale=qk_scale,
                                         dropout=drop,
                                         attention_dropout=attn_drop,
                                         droppath=drop_path,
                                         downsample=downsample)

        if resi_connection == '1conv':
            self.conv = nn.Conv2D(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2D(dim, dim // 4, 3, 1, 1),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Conv2D(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Conv2D(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      in_chans=0,
                                      embed_dim=dim,
                                      norm_layer=None)

        self.patch_unembed = PatchUnEmbed(img_size=img_size,
                                          patch_size=patch_size,
                                          in_chans=0,
                                          embed_dim=dim,
                                          norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(
            self.conv(self.patch_unembed(self.residual_group(x, x_size),
                                         x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Layer):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Layer, optional): Normalization layer. Default: None
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose([0, 2, 1])  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Layer):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Layer, optional): Normalization layer. Default: None
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose([0, 2,
                         1]).reshape([B, self.embed_dim, x_size[0],
                                      x_size[1]])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2D(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2D(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2D(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops


@GENERATORS.register()
class SwinIR(nn.Layer):
    r""" SwinIR
        A Pypaddle impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Layer): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=[6, 6, 6, 6],
                 num_heads=[6, 6, 6, 6],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv'):
        super(SwinIR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = np.array([0.4488, 0.4371, 0.4040], dtype=np.float32)
            self.mean = paddle.Tensor(rgb_mean).reshape([1, 3, 1, 1])
        else:
            self.mean = paddle.zeros([1., 1., 1., 1.], dtype=paddle.float32)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size

        # 1. shallow feature extraction
        self.conv_first = nn.Conv2D(num_in_ch, embed_dim, 3, 1, 1)

        # 2. deep feature extraction
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = paddle.nn.ParameterList([
                paddle.create_parameter(
                    shape=[1, num_patches, embed_dim],
                    dtype='float32',
                    default_initializer=paddle.nn.initializer.TruncatedNormal(
                        std=.02))
            ])

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in paddle.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.LayerList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]
                                  ):sum(depths[:i_layer +
                                               1])],  # no impact on SR results
                downsample=None,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2D(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2D(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2D(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2D(embed_dim // 4, embed_dim, 3, 1, 1))

        # 3, high quality image reconstruction ################################
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2D(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU())
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2D(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(
                upscale, embed_dim, num_out_ch,
                (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2D(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU())
            self.conv_up1 = nn.Conv2D(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2D(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2D(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2D(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2D(embed_dim, num_out_ch, 3, 1, 1)

    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.shape
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(
                self.conv_up1(
                    paddle.nn.functional.interpolate(x,
                                                     scale_factor=2,
                                                     mode='nearest')))
            x = self.lrelu(
                self.conv_up2(
                    paddle.nn.functional.interpolate(x,
                                                     scale_factor=2,
                                                     mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x[:, :, :H * self.upscale, :W * self.upscale]

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops
