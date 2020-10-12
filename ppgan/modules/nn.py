# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import math


class _SpectralNorm(nn.SpectralNorm):
    def __init__(self,
                 weight_shape,
                 dim=0,
                 power_iters=1,
                 eps=1e-12,
                 dtype='float32'):
        super(_SpectralNorm, self).__init__(weight_shape, dim, power_iters, eps,
                                            dtype)

    def forward(self, weight):
        inputs = {'Weight': weight, 'U': self.weight_u, 'V': self.weight_v}
        out = self._helper.create_variable_for_type_inference(self._dtype)
        _power_iters = self._power_iters if self.training else 0
        self._helper.append_op(type="spectral_norm",
                               inputs=inputs,
                               outputs={
                                   "Out": out,
                               },
                               attrs={
                                   "dim": self._dim,
                                   "power_iters": _power_iters,
                                   "eps": self._eps,
                               })

        return out


class Spectralnorm(paddle.nn.Layer):
    def __init__(self, layer, dim=0, power_iters=1, eps=1e-12, dtype='float32'):
        super(Spectralnorm, self).__init__()
        self.spectral_norm = _SpectralNorm(layer.weight.shape, dim, power_iters,
                                           eps, dtype)
        self.dim = dim
        self.power_iters = power_iters
        self.eps = eps
        self.layer = layer
        weight = layer._parameters['weight']
        del layer._parameters['weight']
        self.weight_orig = self.create_parameter(weight.shape,
                                                 dtype=weight.dtype)
        self.weight_orig.set_value(weight)

    def forward(self, x):
        weight = self.spectral_norm(self.weight_orig)
        self.layer.weight = weight
        out = self.layer(x)
        return out


def initial_type(input,
                 op_type,
                 fan_out,
                 init="normal",
                 use_bias=False,
                 kernel_size=0,
                 stddev=0.02,
                 name=None):
    if init == "kaiming":
        if op_type == 'conv':
            fan_in = input.shape[1] * kernel_size * kernel_size
        elif op_type == 'deconv':
            fan_in = fan_out * kernel_size * kernel_size
        else:
            if len(input.shape) > 2:
                fan_in = input.shape[1] * input.shape[2] * input.shape[3]
            else:
                fan_in = input.shape[1]
        bound = 1 / math.sqrt(fan_in)
        param_attr = paddle.ParamAttr(
            # name=name + "_w",
            initializer=paddle.nn.initializer.Uniform(low=-bound, high=bound))
        if use_bias == True:
            bias_attr = paddle.ParamAttr(
                # name=name + '_b',
                initializer=paddle.nn.initializer.Uniform(low=-bound,
                                                          high=bound))
        else:
            bias_attr = False
    elif init == 'xavier':
        param_attr = paddle.ParamAttr(
            # name=name + "_w",
            initializer=paddle.nn.initializer.Xavier(uniform=False))
        if use_bias == True:
            bias_attr = paddle.ParamAttr(
                # name=name + "_b",
                initializer=paddle.nn.initializer.Constant(0.0))
        else:
            bias_attr = False
    else:
        param_attr = paddle.ParamAttr(
            # name=name + "_w",
            initializer=paddle.nn.initializer.NormalInitializer(loc=0.0,
                                                                scale=stddev))
        if use_bias == True:
            bias_attr = paddle.ParamAttr(
                # name=name + "_b",
                initializer=paddle.nn.initializer.Constant(0.0))
        else:
            bias_attr = False
    return param_attr, bias_attr


class Conv2d(paddle.nn.Conv2d):
    def __init__(self,
                 num_channels,
                 num_filters,
                 kernel_size,
                 padding=0,
                 stride=1,
                 dilation=1,
                 groups=1,
                 weight_attr=None,
                 bias_attr=None,
                 data_format="NCHW",
                 init_type='xavier'):
        param_attr, bias_attr = initial_type(
            input=input,
            op_type='conv',
            fan_out=num_filters,
            init=init_type,
            use_bias=True if bias_attr != False else False,
            kernel_size=kernel_size)

        super(Conv2d, self).__init__(in_channels=num_channels,
                                     out_channels=num_filters,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     dilation=dilation,
                                     groups=groups,
                                     weight_attr=param_attr,
                                     bias_attr=bias_attr,
                                     data_format=data_format)


class ConvTranspose2d(paddle.nn.ConvTranspose2d):
    def __init__(self,
                 num_channels,
                 num_filters,
                 kernel_size,
                 padding=0,
                 stride=1,
                 dilation=1,
                 groups=1,
                 weight_attr=None,
                 bias_attr=None,
                 data_format="NCHW",
                 init_type='normal'):

        param_attr, bias_attr = initial_type(
            input=input,
            op_type='deconv',
            fan_out=num_filters,
            init=init_type,
            use_bias=True if bias_attr != False else False,
            kernel_size=kernel_size)

        super(ConvTranspose2d, self).__init__(in_channels=num_channels,
                                              out_channels=num_filters,
                                              kernel_size=kernel_size,
                                              padding=padding,
                                              stride=stride,
                                              dilation=dilation,
                                              groups=groups,
                                              weight_attr=weight_attr,
                                              bias_attr=bias_attr,
                                              data_format=data_format)
