import paddle.nn.functional as F
import paddle
import paddle.nn as nn
from .builder import GENERATORS
import math
from .initializer import NewMSRAInitializer


class KaimingUniform(NewMSRAInitializer):

    def __init__(self, fan_in=None):
        super(KaimingUniform, self).__init__(uniform=True,
                                             fan_in=fan_in,
                                             seed=0)


def convWithBias(in_channels, out_channels, kernel_size, stride, padding):
    if isinstance(kernel_size, int):
        fan_in = kernel_size * kernel_size * in_channels
    else:
        fan_in = kernel_size[0] * kernel_size[1] * in_channels
    bound = 1 / math.sqrt(fan_in)
    bias_attr = paddle.framework.ParamAttr(
        initializer=nn.initializer.Uniform(-bound, bound))
    weight_attr = paddle.framework.ParamAttr(initializer=KaimingUniform())
    conv = nn.Conv2D(in_channels,
                     out_channels,
                     kernel_size,
                     stride,
                     padding,
                     weight_attr=weight_attr,
                     bias_attr=bias_attr)
    return conv


@GENERATORS.register()
class PReNet(nn.Layer):

    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PReNet, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(convWithBias(6, 32, 3, 1, 1), nn.ReLU())
        self.res_conv1 = nn.Sequential(convWithBias(32, 32, 3, 1, 1), nn.ReLU(),
                                       convWithBias(32, 32, 3, 1, 1), nn.ReLU())
        self.res_conv2 = nn.Sequential(convWithBias(32, 32, 3, 1, 1), nn.ReLU(),
                                       convWithBias(32, 32, 3, 1, 1), nn.ReLU())
        self.res_conv3 = nn.Sequential(convWithBias(32, 32, 3, 1, 1), nn.ReLU(),
                                       convWithBias(32, 32, 3, 1, 1), nn.ReLU())
        self.res_conv4 = nn.Sequential(convWithBias(32, 32, 3, 1, 1), nn.ReLU(),
                                       convWithBias(32, 32, 3, 1, 1), nn.ReLU())
        self.res_conv5 = nn.Sequential(convWithBias(32, 32, 3, 1, 1), nn.ReLU(),
                                       convWithBias(32, 32, 3, 1, 1), nn.ReLU())
        self.conv_i = nn.Sequential(convWithBias(32 + 32, 32, 3, 1, 1),
                                    nn.Sigmoid())
        self.conv_f = nn.Sequential(convWithBias(32 + 32, 32, 3, 1, 1),
                                    nn.Sigmoid())
        self.conv_g = nn.Sequential(convWithBias(32 + 32, 32, 3, 1, 1),
                                    nn.Tanh())
        self.conv_o = nn.Sequential(convWithBias(32 + 32, 32, 3, 1, 1),
                                    nn.Sigmoid())
        self.conv = nn.Sequential(convWithBias(32, 3, 3, 1, 1), )

    def forward(self, input):
        batch_size, row, col = input.shape[0], input.shape[2], input.shape[3]

        x = input
        # h  = paddle.create_parameter(shape=(batch_size, 32, row, col),dtype='float32',is_bias=True)
        # c  = paddle.create_parameter(shape=(batch_size, 32, row, col),dtype='float32',is_bias=True)

        h = paddle.to_tensor(paddle.zeros(shape=(batch_size, 32, row, col),
                                          dtype='float32'),
                             stop_gradient=False)
        c = paddle.to_tensor(paddle.zeros(shape=(batch_size, 32, row, col),
                                          dtype='float32'),
                             stop_gradient=False)

        x_list = []
        for _ in range(self.iteration):
            x = paddle.concat((input, x), 1)
            x = self.conv0(x)

            x = paddle.concat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * paddle.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x = x + input
            x_list.append(x)
        return x
