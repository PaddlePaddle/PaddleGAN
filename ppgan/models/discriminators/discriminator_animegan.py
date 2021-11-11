# code was heavily based on https://github.com/TachibanaYoshino/AnimeGANv2
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/TachibanaYoshino/AnimeGANv2#license

import paddle.nn as nn
import paddle.nn.functional as F

from .builder import DISCRIMINATORS
from ...modules.utils import spectral_norm


@DISCRIMINATORS.register()
class AnimeDiscriminator(nn.Layer):
    def __init__(self, channel: int = 64, nblocks: int = 3) -> None:
        super().__init__()
        channel = channel // 2
        last_channel = channel
        f = [
            spectral_norm(
                nn.Conv2D(3, channel, 3, stride=1, padding=1, bias_attr=False)),
            nn.LeakyReLU(0.2)
        ]
        in_h = 256
        for i in range(1, nblocks):
            f.extend([
                spectral_norm(
                    nn.Conv2D(last_channel,
                              channel * 2,
                              3,
                              stride=2,
                              padding=1,
                              bias_attr=False)),
                nn.LeakyReLU(0.2),
                spectral_norm(
                    nn.Conv2D(channel * 2,
                              channel * 4,
                              3,
                              stride=1,
                              padding=1,
                              bias_attr=False)),
                nn.GroupNorm(1, channel * 4),
                nn.LeakyReLU(0.2)
            ])
            last_channel = channel * 4
            channel = channel * 2
            in_h = in_h // 2

        self.body = nn.Sequential(*f)

        self.head = nn.Sequential(*[
            spectral_norm(
                nn.Conv2D(last_channel,
                          channel * 2,
                          3,
                          stride=1,
                          padding=1,
                          bias_attr=False)),
            nn.GroupNorm(1, channel * 2),
            nn.LeakyReLU(0.2),
            spectral_norm(
                nn.Conv2D(
                    channel * 2, 1, 3, stride=1, padding=1, bias_attr=False))
        ])

    def forward(self, x):
        x = self.body(x)
        x = self.head(x)
        return x
