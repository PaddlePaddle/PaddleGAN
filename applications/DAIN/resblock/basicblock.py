import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D

__all__ = ['MultipleBasicBlock', 'MultipleBasicBlock_4']


def conv3x3(in_planes, out_planes, dilation=1, stride=1, param_attr=None):
    return Conv2D(in_planes,
                  out_planes,
                  filter_size=3,
                  stride=stride,
                  padding=int(dilation * (3 - 1) / 2),
                  dilation=dilation,
                  bias_attr=False,
                  param_attr=param_attr)


class BasicBlock(fluid.dygraph.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, dilation=1, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        param_attr = fluid.ParamAttr(
            initializer=fluid.initializer.NormalInitializer(
                loc=0.0, scale=1.0, seed=0))

        self.conv1 = conv3x3(inplanes, planes, dilation, stride, param_attr)
        self.conv2 = conv3x3(planes, planes, param_attr=param_attr)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = fluid.layers.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = fluid.layers.relu(out)

        return out


class MultipleBasicBlock(fluid.dygraph.Layer):
    def __init__(self,
                 input_feature,
                 block,
                 num_blocks,
                 intermediate_feature=64,
                 dense=True):
        super(MultipleBasicBlock, self).__init__()
        self.dense = dense
        self.num_block = num_blocks
        self.intermediate_feature = intermediate_feature

        param_attr = fluid.ParamAttr(
            initializer=fluid.initializer.NormalInitializer(
                loc=0.0, scale=1.0, seed=0))

        self.block1 = Conv2D(input_feature,
                             intermediate_feature,
                             filter_size=7,
                             stride=1,
                             padding=3,
                             bias_attr=True,
                             param_attr=param_attr)

        dim = intermediate_feature
        self.block2 = block(dim, dim, dilation=1) if num_blocks >= 2 else None
        self.block3 = block(dim, dim, dilation=1) if num_blocks >= 3 else None
        self.block4 = block(dim, dim, dilation=1) if num_blocks >= 4 else None
        self.block5 = Conv2D(dim, 3, 3, 1, 1)

    def forward(self, x):
        x = fluid.layers.relu(self.block1(x))
        x = self.block2(x) if self.num_block >= 2 else x
        x = self.block3(x) if self.num_block >= 3 else x
        x = self.block4(x) if self.num_block >= 4 else x
        x = self.block5(x)
        return x


def MultipleBasicBlock_4(input_feature, intermediate_feature=64):
    model = MultipleBasicBlock(input_feature, BasicBlock, 4,
                               intermediate_feature)
    return model
