#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import numpy as np
import paddle
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype
from paddle.fluid.layers import deformable_conv
from paddle.fluid import core, layers
from paddle.fluid.layers import nn, utils
from paddle.nn import Layer
from paddle.fluid.initializer import Normal
from paddle.common_ops_import import *


class DeformConv2D(Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 deformable_groups=1,
                 groups=1,
                 weight_attr=None,
                 bias_attr=None):
        super(DeformConv2D, self).__init__()
        assert weight_attr is not False, "weight_attr should not be False in Conv."
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self._deformable_groups = deformable_groups
        self._groups = groups
        self._in_channels = in_channels
        self._out_channels = out_channels
        self.padding = padding
        self.stride = stride
        self._channel_dim = 1

        self._stride = utils.convert_to_list(stride, 2, 'stride')
        self._dilation = utils.convert_to_list(dilation, 2, 'dilation')
        self._kernel_size = utils.convert_to_list(kernel_size, 2, 'kernel_size')

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups.")

        self._padding = utils.convert_to_list(padding, 2, 'padding')

        filter_shape = [out_channels, in_channels // groups] + self._kernel_size

        def _get_default_param_initializer():
            filter_elem_num = np.prod(self._kernel_size) * self._in_channels
            std = (2.0 / filter_elem_num)**0.5
            return Normal(0.0, std, 0)

        self.weight = self.create_parameter(
            shape=filter_shape,
            attr=self._weight_attr,
            default_initializer=_get_default_param_initializer())
        self.bias = self.create_parameter(
            attr=self._bias_attr, shape=[self._out_channels], is_bias=True)

    def forward(self, x, offset, mask):
        out = deform_conv2d(
            x=x,
            offset=offset,
            mask=mask,
            weight=self.weight,
            bias=self.bias,
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
            deformable_groups=self._deformable_groups,
            groups=self._groups,
            )
        return out


def deform_conv2d(x,
                  offset,
                  weight,
                  mask,
                  bias=None,
                  stride=1,
                  padding=0,
                  dilation=1,
                  deformable_groups=1,
                  groups=1,
                  name=None):
    
    stride = utils.convert_to_list(stride, 2, 'stride')
    padding = utils.convert_to_list(padding, 2, 'padding')
    dilation = utils.convert_to_list(dilation, 2, 'dilation')

    use_deform_conv2d_v1 = True if mask is None else False

    if in_dygraph_mode():
        attrs = ('strides', stride, 'paddings', padding, 'dilations', dilation, 'deformable_groups',deformable_groups,
                 'groups', groups, 'im2col_step', 1)
        if use_deform_conv2d_v1:
            op_type = 'deformable_conv_v1'
            pre_bias = getattr(core.ops, op_type)(x, offset, weight, *attrs)
        else:
            op_type = 'deformable_conv'
            pre_bias = getattr(core.ops, op_type)(x, offset, mask, weight,
                                                  *attrs)
        if bias is not None:
            out = nn.elementwise_add(pre_bias, bias, axis=1)
        else:
            out = pre_bias
    return out


class DeformableConv_dygraph(Layer):
    def __init__(self,num_filters,filter_size,dilation,
                 stride,padding,deformable_groups=1,groups=1):
        super(DeformableConv_dygraph, self).__init__()
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.dilation = dilation
        self.stride = stride
        self.padding = padding
        self.deformable_groups = deformable_groups
        self.groups = groups
        self.defor_conv = DeformConv2D(in_channels=self.num_filters, out_channels=self.num_filters, 
                                                           kernel_size=self.filter_size, stride=self.stride, padding=self.padding, 
                                                           dilation=self.dilation, deformable_groups=self.deformable_groups, groups=self.groups, weight_attr=None, bias_attr=None)


    def forward(self,*input):
        x = input[0]
        offset = input[1]
        mask = input[2]
        out = self.defor_conv(x, offset, mask)
        return out
