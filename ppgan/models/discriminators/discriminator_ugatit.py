# code was based on https://github.com/znxlwm/UGATIT-pytorch

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ...modules.utils import spectral_norm
from .builder import DISCRIMINATORS


@DISCRIMINATORS.register()
class UGATITDiscriminator(nn.Layer):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(UGATITDiscriminator, self).__init__()
        model = [
            nn.Pad2D(padding=[1, 1, 1, 1], mode="reflect"),
            spectral_norm(
                nn.Conv2D(input_nc,
                          ndf,
                          kernel_size=4,
                          stride=2,
                          padding=0,
                          bias_attr=True)),
            nn.LeakyReLU(0.2)
        ]

        for i in range(1, n_layers - 2):
            mult = 2**(i - 1)
            model += [
                nn.Pad2D(padding=[1, 1, 1, 1], mode="reflect"),
                spectral_norm(
                    nn.Conv2D(ndf * mult,
                              ndf * mult * 2,
                              kernel_size=4,
                              stride=2,
                              padding=0,
                              bias_attr=True)),
                nn.LeakyReLU(0.2)
            ]

        mult = 2**(n_layers - 2 - 1)
        model += [
            nn.Pad2D(padding=[1, 1, 1, 1], mode="reflect"),
            spectral_norm(
                nn.Conv2D(ndf * mult,
                          ndf * mult * 2,
                          kernel_size=4,
                          stride=1,
                          padding=0,
                          bias_attr=True)),
            nn.LeakyReLU(0.2)
        ]

        # Class Activation Map
        mult = 2**(n_layers - 2)
        self.gap_fc = spectral_norm(nn.Linear(ndf * mult, 1, bias_attr=False))
        self.gmp_fc = spectral_norm(nn.Linear(ndf * mult, 1, bias_attr=False))
        self.conv1x1 = nn.Conv2D(ndf * mult * 2,
                                 ndf * mult,
                                 kernel_size=1,
                                 stride=1,
                                 bias_attr=True)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.pad = nn.Pad2D(padding=[1, 1, 1, 1], mode="reflect")
        self.conv = spectral_norm(
            nn.Conv2D(ndf * mult,
                      1,
                      kernel_size=4,
                      stride=1,
                      padding=0,
                      bias_attr=False))

        self.model = nn.Sequential(*model)

    def forward(self, input):
        x = self.model(input)

        gap = F.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.reshape([x.shape[0], -1]))
        gap_weight = list(self.gap_fc.parameters())[0].transpose([1, 0])
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = F.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.reshape([x.shape[0], -1]))
        gmp_weight = list(self.gmp_fc.parameters())[0].transpose([1, 0])
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = paddle.concat([gap_logit, gmp_logit], 1)
        x = paddle.concat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))

        heatmap = paddle.sum(x, 1, keepdim=True)

        x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit, heatmap
