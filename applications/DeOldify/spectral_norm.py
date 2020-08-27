import numpy as np
from paddle import fluid
from paddle.fluid import dygraph
from paddle.fluid import layers as F
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.data_feeder import check_variable_and_dtype

import paddle
import paddle.nn as nn

class _SpectralNorm(nn.SpectralNorm):
    def __init__(self,
                 weight_shape,
                 dim=0,
                 power_iters=1,
                 eps=1e-12,
                 dtype='float32'):
        super(_SpectralNorm, self).__init__(weight_shape, dim, power_iters, eps, dtype)

    def forward(self, weight):
        check_variable_and_dtype(weight, "weight", ['float32', 'float64'],
                                 'SpectralNorm')
        inputs = {'Weight': weight, 'U': self.weight_u, 'V': self.weight_v}
        out = self._helper.create_variable_for_type_inference(self._dtype)
        _power_iters = self._power_iters if self.training else 0
        self._helper.append_op(
            type="spectral_norm",
            inputs=inputs,
            outputs={"Out": out, },
            attrs={
                "dim": self._dim,
                "power_iters": _power_iters, #self._power_iters,
                "eps": self._eps,
            })

        return out


class Spectralnorm(nn.Layer):

    def __init__(self,
                 layer,
                 dim=0,
                 power_iters=1,
                 eps=1e-12,
                 dtype='float32'):
        super(Spectralnorm, self).__init__()
        self.spectral_norm = _SpectralNorm(layer.weight.shape, dim, power_iters, eps, dtype)
        self.dim = dim
        self.power_iters = power_iters
        self.eps = eps
        self.layer = layer
        weight = layer._parameters['weight']
        del layer._parameters['weight']
        self.weight_orig = self.create_parameter(weight.shape, dtype=weight.dtype)
        self.weight_orig.set_value(weight)
        

    def forward(self, x):
        weight = self.spectral_norm(self.weight_orig)
        self.layer.weight = weight
        out = self.layer(x)
        return out
