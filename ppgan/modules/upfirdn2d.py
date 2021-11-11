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

# code was heavily based on https://github.com/rosinality/stylegan2-pytorch
# MIT License
# Copyright (c) 2019 Kim Seonghyeon

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1,
                     pad_y0, pad_y1):
    _, channel, in_h, in_w = input.shape
    input = input.reshape((-1, in_h, in_w, 1))

    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.reshape((-1, in_h, 1, in_w, 1, minor))
    out = out.transpose((0, 1, 3, 5, 2, 4))
    out = out.reshape((-1, 1, 1, 1))
    out = F.pad(out, [0, up_x - 1, 0, up_y - 1])
    out = out.reshape((-1, in_h, in_w, minor, up_y, up_x))
    out = out.transpose((0, 3, 1, 4, 2, 5))
    out = out.reshape((-1, minor, in_h * up_y, in_w * up_x))

    out = F.pad(
        out, [max(pad_x0, 0),
              max(pad_x1, 0),
              max(pad_y0, 0),
              max(pad_y1, 0)])
    out = out[:, :,
              max(-pad_y0, 0):out.shape[2] - max(-pad_y1, 0),
              max(-pad_x0, 0):out.shape[3] - max(-pad_x1, 0), ]

    out = out.reshape(
        ([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]))
    w = paddle.flip(kernel, [0, 1]).reshape((1, 1, kernel_h, kernel_w))
    out = F.conv2d(out, w)
    out = out.reshape((
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    ))
    out = out.transpose((0, 2, 3, 1))
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.reshape((-1, channel, out_h, out_w))


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    out = upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1],
                           pad[0], pad[1])

    return out


def make_kernel(k):
    k = paddle.to_tensor(k, dtype='float32')

    if k.ndim == 1:
        k = k.unsqueeze(0) * k.unsqueeze(1)

    k /= k.sum()

    return k


class Upfirdn2dUpsample(nn.Layer):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor * factor)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input,
                        self.kernel,
                        up=self.factor,
                        down=1,
                        pad=self.pad)

        return out


class Upfirdn2dDownsample(nn.Layer):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input,
                        self.kernel,
                        up=1,
                        down=self.factor,
                        pad=self.pad)

        return out


class Upfirdn2dBlur(nn.Layer):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor * upsample_factor)

        self.register_buffer("kernel", kernel, persistable=False)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out
