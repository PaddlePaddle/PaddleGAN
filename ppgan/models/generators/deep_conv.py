# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import paddle.nn.functional as F

from .builder import GENERATORS


@GENERATORS.register()
class DeepConvGenerator(nn.Layer):
    """Create a Deep Convolutional generator
       Refer to https://arxiv.org/abs/1511.06434
    """
    def __init__(self, latent_dim, output_nc, size=64, ngf=64):
        """Construct a Deep Convolutional generator
        Args:
            latent_dim (int): the number of latent dimension
            output_nc (int): the number of channels in output images
            size (int): size of output tensor
            ngf (int): the number of filters in the last conv layer
        """
        super(DeepConvGenerator, self).__init__()

        self.latent_dim = latent_dim
        self.ngf = ngf
        self.init_size = size // 4
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, ngf * 2 * self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2D(ngf * 2),
            nn.Upsample(scale_factor=2),
            nn.Conv2D(ngf * 2, ngf * 2, 3, stride=1, padding=1),
            nn.BatchNorm2D(ngf * 2, 0.2),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2D(ngf * 2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2D(ngf, 0.2),
            nn.LeakyReLU(0.2),
            nn.Conv2D(ngf, output_nc, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def random_inputs(self, batch_size):
        return paddle.randn([batch_size, self.latent_dim])

    def forward(self, z):
        out = self.l1(z)
        out = out.reshape(
            [out.shape[0], self.ngf * 2, self.init_size, self.init_size])
        img = self.conv_blocks(out)
        return img


@GENERATORS.register()
class ConditionalDeepConvGenerator(DeepConvGenerator):
    """Create a Conditional Deep Convolutional generator
    """
    def __init__(self, latent_dim, output_nc, n_class=10, **kwargs):
        """Construct a Conditional Deep Convolutional generator
        Args:
            latent_dim (int): the number of latent dimension
            output_nc (int): the number of channels in output images
            n_class (int): the number of class
        """
        super(ConditionalDeepConvGenerator,
              self).__init__(latent_dim + n_class, output_nc, **kwargs)

        self.n_class = n_class
        self.latent_dim = latent_dim

    def random_inputs(self, batch_size):
        return_list = [
            super(ConditionalDeepConvGenerator, self).random_inputs(batch_size)
        ]
        class_id = paddle.randint(0, self.n_class, [batch_size])
        return return_list + [class_id]

    def forward(self, x, class_id=None):
        if self.n_class > 0:
            class_id = (class_id % self.n_class).detach()
            class_id = F.one_hot(class_id, self.n_class).astype('float32')
            class_id = class_id.reshape([x.shape[0], -1])
            x = paddle.concat([x, class_id], 1)

        return super(ConditionalDeepConvGenerator, self).forward(x)
