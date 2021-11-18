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

# code was based on https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
import functools
import paddle
import paddle.nn as nn

from ...modules.norm import build_norm_layer
from .builder import GENERATORS


@GENERATORS.register()
class UnetGenerator(nn.Layer):
    """Create a Unet-based generator"""
    def __init__(self,
                 input_nc,
                 output_nc,
                 num_downs,
                 ngf=64,
                 norm_type='batch',
                 use_dropout=False):
        """
        Construct a Unet generator
        the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.

        Args:
            input_nc (int): the number of channels in input images.
            output_nc (int): the number of channels in output images.
            num_downs (int): the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck.
            ngf (int): the number of filters in the last conv layer.
            norm_type (str): normalization type, default: 'batch'.

        """
        super(UnetGenerator, self).__init__()
        norm_layer = build_norm_layer(norm_type)
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(
            ngf * 8,
            ngf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True)  # add the innermost layer
        for i in range(num_downs -
                       5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8,
                                                 ngf * 8,
                                                 input_nc=None,
                                                 submodule=unet_block,
                                                 norm_layer=norm_layer,
                                                 use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4,
                                             ngf * 8,
                                             input_nc=None,
                                             submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2,
                                             ngf * 4,
                                             input_nc=None,
                                             submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf,
                                             ngf * 2,
                                             input_nc=None,
                                             submodule=unet_block,
                                             norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Layer):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """
    def __init__(self,
                 outer_nc,
                 inner_nc,
                 input_nc=None,
                 submodule=None,
                 outermost=False,
                 innermost=False,
                 norm_layer=nn.BatchNorm,
                 use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Args:
            outer_nc (int): the number of filters in the outer conv layer
            inner_nc (int): the number of filters in the inner conv layer
            input_nc (int): the number of channels in input images/features
            submodule (UnetSkipConnectionBlock): previously defined submodules
            outermost (bool): if this module is the outermost module
            innermost (bool): if this module is the innermost module
            norm_layer (paddle.nn.Layer): normalization layer
            use_dropout (bool): whether to  use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2D
        else:
            use_bias = norm_layer == nn.InstanceNorm2D
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2D(input_nc,
                             inner_nc,
                             kernel_size=4,
                             stride=2,
                             padding=1,
                             bias_attr=use_bias)
        downrelu = nn.LeakyReLU(0.2)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU()
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.Conv2DTranspose(inner_nc * 2,
                                        outer_nc,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.Conv2DTranspose(inner_nc,
                                        outer_nc,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1,
                                        bias_attr=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.Conv2DTranspose(inner_nc * 2,
                                        outer_nc,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1,
                                        bias_attr=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        # add skip connections
        else:
            return paddle.concat([x, self.model(x)], 1)
