# MIT License
# Copyright (c) 2018 Jason Antic

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.models import resnet34, resnet101

from .hook import hook_outputs, model_sizes, dummy_eval
from ...modules.nn import Spectralnorm


class SequentialEx(nn.Layer):
    "Like `nn.Sequential`, but with ModuleList semantics, and can access module input"

    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.LayerList(layers)

    def forward(self, x):
        res = x
        for l in self.layers:
            if isinstance(l, MergeLayer):
                l.orig = x
            nres = l(res)
            # We have to remove res.orig to avoid hanging refs and therefore memory leaks
            # l.orig = None
            res = nres
        return res

    def __getitem__(self, i):
        return self.layers[i]

    def append(self, l):
        return self.layers.append(l)

    def extend(self, l):
        return self.layers.extend(l)

    def insert(self, i, l):
        return self.layers.insert(i, l)


class Deoldify(SequentialEx):
    def __init__(self,
                 encoder,
                 n_classes,
                 model_type='stable',
                 blur=False,
                 blur_final=True,
                 self_attention=False,
                 y_range=None,
                 last_cross=True,
                 bottle=False,
                 norm_type='Batch',
                 nf_factor=1,
                 **kwargs):

        imsize = (256, 256)
        sfs_szs = model_sizes(encoder, size=imsize)
        sfs_idxs = list(reversed(_get_sfs_idxs(sfs_szs)))
        self.sfs = hook_outputs([encoder[i] for i in sfs_idxs], detach=False)
        x = dummy_eval(encoder, imsize).detach()

        nf = 512 * nf_factor
        extra_bn = norm_type == 'Spectral'
        ni = sfs_szs[-1][1]
        middle_conv = nn.Sequential(
            custom_conv_layer(ni,
                              ni * 2,
                              norm_type=norm_type,
                              extra_bn=extra_bn),
            custom_conv_layer(ni * 2,
                              ni,
                              norm_type=norm_type,
                              extra_bn=extra_bn),
        )

        layers = [encoder, nn.BatchNorm(ni), nn.ReLU(), middle_conv]

        for i, idx in enumerate(sfs_idxs):
            not_final = i != len(sfs_idxs) - 1
            up_in_c, x_in_c = int(x.shape[1]), int(sfs_szs[idx][1])
            do_blur = blur and (not_final or blur_final)
            sa = self_attention and (i == len(sfs_idxs) - 3)

            if model_type == 'stable':
                n_out = nf if not_final else nf // 2
                unet_block = UnetBlockWide(up_in_c,
                                           x_in_c,
                                           n_out,
                                           self.sfs[i],
                                           final_div=not_final,
                                           blur=blur,
                                           self_attention=sa,
                                           norm_type=norm_type,
                                           extra_bn=extra_bn,
                                           **kwargs)
            elif model_type == 'artistic':
                unet_block = UnetBlockDeep(up_in_c,
                                           x_in_c,
                                           self.sfs[i],
                                           final_div=not_final,
                                           blur=blur,
                                           self_attention=sa,
                                           norm_type=norm_type,
                                           extra_bn=extra_bn,
                                           nf_factor=nf_factor,
                                           **kwargs)
            else:
                raise ValueError(
                    'Expected model_type in [stable, artistic], but got {}'.
                    format(model_type))

            unet_block.eval()
            layers.append(unet_block)
            x = unet_block(x)

        ni = x.shape[1]
        if imsize != sfs_szs[0][-2:]:
            layers.append(PixelShuffle_ICNR(ni, **kwargs))
        if last_cross:
            layers.append(MergeLayer(dense=True))
            ni += 3
            layers.append(
                res_block(ni, bottle=bottle, norm_type=norm_type, **kwargs))
        layers += [
            custom_conv_layer(ni,
                              n_classes,
                              ks=1,
                              use_activ=False,
                              norm_type=norm_type)
        ]
        if y_range is not None:
            layers.append(SigmoidRange(*y_range))
        super().__init__(*layers)


def custom_conv_layer(ni: int,
                      nf: int,
                      ks: int = 3,
                      stride: int = 1,
                      padding: int = None,
                      bias: bool = None,
                      is_1d: bool = False,
                      norm_type='Batch',
                      use_activ: bool = True,
                      leaky: float = None,
                      transpose: bool = False,
                      self_attention: bool = False,
                      extra_bn: bool = False,
                      **kwargs):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and batchnorm (if `bn`) layers."
    if padding is None:
        padding = (ks - 1) // 2 if not transpose else 0
    bn = norm_type in ('Batch', 'Batchzero') or extra_bn == True
    if bias is None:
        bias = not bn
    conv_func = nn.Conv2DTranspose if transpose else nn.Conv1D if is_1d else nn.Conv2D

    conv = conv_func(ni,
                     nf,
                     kernel_size=ks,
                     bias_attr=bias,
                     stride=stride,
                     padding=padding)
    if norm_type == 'Weight':
        conv = nn.utils.weight_norm(conv)
    elif norm_type == 'Spectral':
        conv = Spectralnorm(conv)
    layers = [conv]
    if use_activ:
        layers.append(relu(True, leaky=leaky))
    if bn:
        layers.append((nn.BatchNorm if is_1d else nn.BatchNorm)(nf))
    if self_attention:
        layers.append(SelfAttention(nf))

    return nn.Sequential(*layers)


def relu(inplace: bool = False, leaky: float = None):
    "Return a relu activation, maybe `leaky` and `inplace`."
    return nn.LeakyReLU(leaky) if leaky is not None else nn.ReLU()


class UnetBlockWide(nn.Layer):
    "A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."

    def __init__(self,
                 up_in_c: int,
                 x_in_c: int,
                 n_out: int,
                 hook,
                 final_div: bool = True,
                 blur: bool = False,
                 leaky: float = None,
                 self_attention: bool = False,
                 **kwargs):
        super().__init__()
        self.hook = hook
        up_out = x_out = n_out // 2
        self.shuf = CustomPixelShuffle_ICNR(up_in_c,
                                            up_out,
                                            blur=blur,
                                            leaky=leaky,
                                            **kwargs)
        self.bn = nn.BatchNorm(x_in_c)
        ni = up_out + x_in_c
        self.conv = custom_conv_layer(ni,
                                      x_out,
                                      leaky=leaky,
                                      self_attention=self_attention,
                                      **kwargs)
        self.relu = relu(leaky=leaky)

    def forward(self, up_in):
        s = self.hook.stored
        up_out = self.shuf(up_in)
        ssh = s.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, s.shape[-2:], mode='nearest')
        cat_x = self.relu(paddle.concat([up_out, self.bn(s)], axis=1))
        return self.conv(cat_x)


class UnetBlockDeep(nn.Layer):
    "A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."

    def __init__(self,
                 up_in_c: int,
                 x_in_c: int,
                 hook,
                 final_div: bool = True,
                 blur: bool = False,
                 leaky: float = None,
                 self_attention: bool = False,
                 nf_factor: float = 1.0,
                 **kwargs):
        super().__init__()
        self.hook = hook
        self.shuf = CustomPixelShuffle_ICNR(up_in_c,
                                            up_in_c // 2,
                                            blur=blur,
                                            leaky=leaky,
                                            **kwargs)
        self.bn = nn.BatchNorm(x_in_c)
        ni = up_in_c // 2 + x_in_c
        nf = int((ni if final_div else ni // 2) * nf_factor)
        self.conv1 = custom_conv_layer(ni, nf, leaky=leaky, **kwargs)
        self.conv2 = custom_conv_layer(nf,
                                       nf,
                                       leaky=leaky,
                                       self_attention=self_attention,
                                       **kwargs)
        self.relu = relu(leaky=leaky)

    def forward(self, up_in):
        s = self.hook.stored
        up_out = self.shuf(up_in)
        ssh = s.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, s.shape[-2:], mode='nearest')
        cat_x = self.relu(paddle.concat([up_out, self.bn(s)], axis=1))
        return self.conv2(self.conv1(cat_x))


def ifnone(a, b):
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a


class PixelShuffle_ICNR(nn.Layer):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`, `icnr` init, and `weight_norm`."

    def __init__(self,
                 ni: int,
                 nf: int = None,
                 scale: int = 2,
                 blur: bool = False,
                 norm_type='Weight',
                 leaky: float = None):
        super().__init__()
        nf = ifnone(nf, ni)
        self.conv = conv_layer(ni,
                               nf * (scale**2),
                               ks=1,
                               norm_type=norm_type,
                               use_activ=False)

        self.shuf = PixelShuffle(scale)

        self.pad = ReplicationPad2d([1, 0, 1, 0])
        self.blur = nn.AvgPool2D(2, stride=1)
        self.relu = relu(True, leaky=leaky)

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x)) if self.blur else x


def conv_layer(ni: int,
               nf: int,
               ks: int = 3,
               stride: int = 1,
               padding: int = None,
               bias: bool = None,
               is_1d: bool = False,
               norm_type='Batch',
               use_activ: bool = True,
               leaky: float = None,
               transpose: bool = False,
               init=None,
               self_attention: bool = False):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and batchnorm (if `bn`) layers."
    if padding is None: padding = (ks - 1) // 2 if not transpose else 0
    bn = norm_type in ('Batch', 'BatchZero')
    if bias is None: bias = not bn
    conv_func = nn.Conv2DTranspose if transpose else nn.Conv1D if is_1d else nn.Conv2D

    conv = conv_func(ni,
                     nf,
                     kernel_size=ks,
                     bias_attr=bias,
                     stride=stride,
                     padding=padding)
    if norm_type == 'Weight':
        conv = nn.utils.weight_norm(conv)
    elif norm_type == 'Spectral':
        conv = Spectralnorm(conv)

    layers = [conv]
    if use_activ: layers.append(relu(True, leaky=leaky))
    if bn: layers.append((nn.BatchNorm if is_1d else nn.BatchNorm)(nf))
    if self_attention: layers.append(SelfAttention(nf))
    return nn.Sequential(*layers)


class CustomPixelShuffle_ICNR(nn.Layer):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`, `icnr` init, and `weight_norm`."

    def __init__(self,
                 ni: int,
                 nf: int = None,
                 scale: int = 2,
                 blur: bool = False,
                 leaky: float = None,
                 **kwargs):
        super().__init__()
        nf = ifnone(nf, ni)
        self.conv = custom_conv_layer(ni,
                                      nf * (scale**2),
                                      ks=1,
                                      use_activ=False,
                                      **kwargs)

        self.shuf = PixelShuffle(scale)

        self.pad = ReplicationPad2d([1, 0, 1, 0])
        self.blur = nn.AvgPool2D(2, stride=1)
        self.relu = nn.LeakyReLU(
            leaky) if leaky is not None else nn.ReLU()  #relu(True, leaky=leaky)

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x)) if self.blur else x


class MergeLayer(nn.Layer):
    "Merge a shortcut with the result of the module by adding them or concatenating thme if `dense=True`."

    def __init__(self, dense: bool = False):
        super().__init__()
        self.dense = dense
        self.orig = None

    def forward(self, x):
        out = paddle.concat([x, self.orig],
                            axis=1) if self.dense else (x + self.orig)
        self.orig = None
        return out


def res_block(nf,
              dense: bool = False,
              norm_type='Batch',
              bottle: bool = False,
              **conv_kwargs):
    "Resnet block of `nf` features. `conv_kwargs` are passed to `conv_layer`."
    norm2 = norm_type
    if not dense and (norm_type == 'Batch'): norm2 = 'BatchZero'
    nf_inner = nf // 2 if bottle else nf
    return SequentialEx(
        conv_layer(nf, nf_inner, norm_type=norm_type, **conv_kwargs),
        conv_layer(nf_inner, nf, norm_type=norm2, **conv_kwargs),
        MergeLayer(dense))


class SigmoidRange(nn.Layer):
    "Sigmoid module with range `(low,x_max)`"

    def __init__(self, low, high):
        super().__init__()
        self.low, self.high = low, high

    def forward(self, x):
        return sigmoid_range(x, self.low, self.high)


def sigmoid_range(x, low, high):
    "Sigmoid function with range `(low, high)`"
    return F.sigmoid(x) * (high - low) + low


class PixelShuffle(nn.Layer):
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        return F.pixel_shuffle(x, self.upscale_factor)


class ReplicationPad2d(nn.Layer):
    def __init__(self, size):
        super(ReplicationPad2d, self).__init__()
        self.size = size

    def forward(self, x):
        return F.pad(x, self.size, mode="replicate")


def conv1d(ni: int,
           no: int,
           ks: int = 1,
           stride: int = 1,
           padding: int = 0,
           bias: bool = False):
    "Create and initialize a `nn.Conv1d` layer with spectral normalization."
    conv = nn.Conv1D(ni, no, ks, stride=stride, padding=padding, bias_attr=bias)
    return Spectralnorm(conv)


class SelfAttention(nn.Layer):
    "Self attention layer for nd."

    def __init__(self, n_channels):
        super().__init__()
        self.query = conv1d(n_channels, n_channels // 8)
        self.key = conv1d(n_channels, n_channels // 8)
        self.value = conv1d(n_channels, n_channels)
        self.gamma = self.create_parameter(
            shape=[1], default_initializer=paddle.nn.initializer.Constant(
                0.0))  #nn.Parameter(tensor([0.]))

    def forward(self, x):
        #Notation from https://arxiv.org/pdf/1805.08318.pdf
        size = x.shape
        x = paddle.reshape(x, list(size[:2]) + [-1])
        f, g, h = self.query(x), self.key(x), self.value(x)

        beta = paddle.nn.functional.softmax(paddle.bmm(
            paddle.transpose(f, [0, 2, 1]), g),
                                            axis=1)
        o = self.gamma * paddle.bmm(h, beta) + x
        return paddle.reshape(o, size)


def _get_sfs_idxs(sizes):
    "Get the indexes of the layers where the size of the activation changes."
    feature_szs = [size[-1] for size in sizes]
    sfs_idxs = list(
        np.where(np.array(feature_szs[:-1]) != np.array(feature_szs[1:]))[0])
    if feature_szs[0] != feature_szs[1]:
        sfs_idxs = [0] + sfs_idxs
    return sfs_idxs


def build_model(model_type='stable'):
    if model_type == 'stable':
        backbone = resnet101()
        nf_factor = 2
    elif model_type == 'artistic':
        backbone = resnet34()
        nf_factor = 1.5
    else:
        raise ValueError(
            'Expected model_type in [stable, artistic], but got {}'.format(
                model_type))

    cut = -2
    encoder = nn.Sequential(*list(backbone.children())[:cut])

    model = Deoldify(encoder,
                     3,
                     model_type=model_type,
                     blur=True,
                     y_range=(-3, 3),
                     norm_type='Spectral',
                     self_attention=True,
                     nf_factor=nf_factor)
    return model
