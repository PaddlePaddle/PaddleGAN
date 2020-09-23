import paddle
import functools
import paddle.nn as nn


class Identity(nn.Layer):
    def forward(self, x):
        return x


def build_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Args:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(
            nn.BatchNorm,
            param_attr=paddle.ParamAttr(
                initializer=nn.initializer.Normal(1.0, 0.02)),
            bias_attr=paddle.ParamAttr(
                initializer=nn.initializer.Constant(0.0)),
            trainable_statistics=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(
            nn.InstanceNorm,
            param_attr=paddle.ParamAttr(
                initializer=nn.initializer.Constant(1.0),
                learning_rate=0.0,
                trainable=False),
            bias_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(0.0),
                                       learning_rate=0.0,
                                       trainable=False))
    elif norm_type == 'none':

        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' %
                                  norm_type)
    return norm_layer
