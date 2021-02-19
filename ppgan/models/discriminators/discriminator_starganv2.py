
import paddle.nn as nn
import paddle

from .builder import DISCRIMINATORS
from ..generators.generator_starganv2 import ResBlk

import numpy as np


@DISCRIMINATORS.register()
class StarGANv2Discriminator(nn.Layer):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2D(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2D(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2D(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        out = self.main(x)
        out = paddle.reshape(out, (out.shape[0], -1))  # (batch, num_domains)
        # idx = paddle.to_tensor(np.array(range(y.shape[0]))).astype('int')
        idx = paddle.zeros_like(out)
        for i in range(idx.shape[0]):
            idx[i, y[i]] = 1
        # s = out[idx, y]  # (batch, style_dim)

        # s = []
        # for i in range(idx.shape[0]):
        #     s += [out[idx[i].numpy().astype(np.int).tolist()[0], y[i].numpy().astype(np.int).tolist()[0]]]
        # s = paddle.stack(s)
        s = idx * out
        s = paddle.sum(s, axis=1)
        # s = paddle.reshape(s, (s.shape[0], -1))
        return s