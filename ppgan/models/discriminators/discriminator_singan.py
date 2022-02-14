# code was based on https://github.com/tamarott/SinGAN

import paddle.nn as nn

from ..generators.generator_singan import ConvBlock
from .builder import DISCRIMINATORS


@DISCRIMINATORS.register()
class SinGANDiscriminator(nn.Layer):
    def __init__(self, 
                 nfc=32, 
                 min_nfc=32, 
                 input_nc=3, 
                 num_layers=5, 
                 ker_size=3, 
                 padd_size=0):
        super(SinGANDiscriminator, self).__init__()
        self.head = ConvBlock(input_nc, nfc, ker_size, padd_size, 1)
        self.body = nn.Sequential()
        for i in range(num_layers - 2):
            N = int(nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, min_nfc), max(N, min_nfc), ker_size, padd_size, 1)
            self.body.add_sublayer('block%d' % (i + 1), block)
        self.tail = nn.Conv2D(max(N, min_nfc), 1, ker_size, 1, padd_size)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x
