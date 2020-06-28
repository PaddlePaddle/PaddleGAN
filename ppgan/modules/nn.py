import paddle

from paddle.fluid.dygraph import Layer
from paddle import fluid


class MSELoss():
    def __init__(self):
        pass

    def __call__(self, prediction, label):
        return fluid.layers.mse_loss(prediction, label)

class L1Loss():
    def __init__(self):
        pass
    
    def __call__(self, prediction, label):
        return fluid.layers.reduce_mean(fluid.layers.elementwise_sub(prediction, label, act='abs'))

class ReflectionPad2d(Layer):
    def __init__(self, size):
        super(ReflectionPad2d, self).__init__()
        self.size = size

    def forward(self, x):
        return fluid.layers.pad2d(x, [self.size] * 4, mode="reflect")


class LeakyReLU(Layer):
    def __init__(self, alpha, inplace=False):
        super(LeakyReLU, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return fluid.layers.leaky_relu(x, self.alpha)


class Tanh(Layer):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        return fluid.layers.tanh(x)


class Dropout(Layer):
    def __init__(self, prob, mode='upscale_in_train'):
        super(Dropout, self).__init__()
        self.prob = prob
        self.mode = mode

    def forward(self, x):
        return fluid.layers.dropout(x, self.prob, dropout_implementation=self.mode)


class BCEWithLogitsLoss():
    def __init__(self, weight=None, reduction='mean'):
        self.weight = weight
        self.reduction = 'mean'

    def __call__(self, x, label):
        out = paddle.fluid.layers.sigmoid_cross_entropy_with_logits(x, label)
        if self.reduction == 'sum':
            return fluid.layers.reduce_sum(out)
        elif self.reduction == 'mean':
            return fluid.layers.reduce_mean(out)
        else:
            return out


# class BCEWithLogitsLoss(fluid.dygraph.Layer):
#     def __init__(self, weight=None, reduction='mean'):
#         if reduction not in ['sum', 'mean', 'none']:
#             raise ValueError(
#                 "The value of 'reduction' in bce_loss should be 'sum', 'mean' or 'none', but "
#                 "received %s, which is not allowed." % reduction)

#         super(BCEWithLogitsLoss, self).__init__()
#         # self.weight = weight
#         # self.reduction = reduction
#         self.bce_loss = paddle.nn.BCELoss(weight, reduction)

#     def forward(self, input, label):
#         input = paddle.nn.functional.sigmoid(input, True)
#         return self.bce_loss(input, label)


def initial_type(
                 input,
                 op_type,
                 fan_out,
                 init="normal",
                 use_bias=False,
                 filter_size=0,
                 stddev=0.02,
                 name=None):
    if init == "kaiming":
        if op_type == 'conv':
            fan_in = input.shape[1] * filter_size * filter_size
        elif op_type == 'deconv':
            fan_in = fan_out * filter_size * filter_size
        else:
            if len(input.shape) > 2:
                fan_in = input.shape[1] * input.shape[2] * input.shape[3]
            else:
                fan_in = input.shape[1]
        bound = 1 / math.sqrt(fan_in)
        param_attr = fluid.ParamAttr(
            # name=name + "_w",
            initializer=fluid.initializer.Uniform(
                low=-bound, high=bound))
        if use_bias == True:
            bias_attr = fluid.ParamAttr(
                # name=name + '_b',
                initializer=fluid.initializer.Uniform(
                    low=-bound, high=bound))
        else:
            bias_attr = False
    else:
        param_attr = fluid.ParamAttr(
            # name=name + "_w",
            initializer=fluid.initializer.NormalInitializer(
                loc=0.0, scale=stddev))
        if use_bias == True:
            bias_attr = fluid.ParamAttr(
                # name=name + "_b", 
                initializer=fluid.initializer.Constant(0.0))
        else:
            bias_attr = False
    return param_attr, bias_attr


class Conv2D(paddle.nn.Conv2D):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 padding=0,
                 stride=1,
                 dilation=1,
                 groups=1,
                 param_attr=None,
                 bias_attr=None,
                 use_cudnn=True,
                 act=None,
                 data_format="NCHW",
                 dtype='float32',
                 init_type='normal'):
        param_attr, bias_attr = initial_type(
                    input=input,
                    op_type='conv',
                    fan_out=num_filters,
                    init=init_type,
                    use_bias=True if bias_attr != False else False,
                    filter_size=filter_size)

        super(Conv2D, self).__init__(num_channels,
                                    num_filters,
                                    filter_size,
                                    padding,
                                    stride,
                                    dilation,
                                    groups,
                                    param_attr,
                                    bias_attr,
                                    use_cudnn,
                                    act,
                                    data_format,
                                    dtype)


class Conv2DTranspose(paddle.nn.Conv2DTranspose):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 output_size=None,
                 padding=0,
                 stride=1,
                 dilation=1,
                 groups=1,
                 param_attr=None,
                 bias_attr=None,
                 use_cudnn=True,
                 act=None,
                 data_format="NCHW",
                 dtype='float32',
                 init_type='normal'):

        param_attr, bias_attr = initial_type(
                    input=input,
                    op_type='deconv',
                    fan_out=num_filters,
                    init=init_type,
                    use_bias=True if bias_attr != False else False,
                    filter_size=filter_size)

        super(Conv2DTranspose, self).__init__(
                                            num_channels,
                                            num_filters,
                                            filter_size,
                                            output_size,
                                            padding,
                                            stride,
                                            dilation,
                                            groups,
                                            param_attr,
                                            bias_attr,
                                            use_cudnn,
                                            act,
                                            data_format,
                                            dtype)
        
class Pad2D(fluid.dygraph.Layer):
    def __init__(self, paddings, mode, pad_value=0.0):
        super(Pad2D, self).__init__()
        self.paddings = paddings
        self.mode = mode

    def forward(self, x):
        return fluid.layers.pad2d(x, self.paddings, self.mode)