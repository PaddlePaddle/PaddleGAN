# code was heavily based on https://github.com/AliaksandrSiarohin/first-order-model
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/AliaksandrSiarohin/first-order-model/blob/master/LICENSE.md

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .builder import DISCRIMINATORS
from ...modules.first_order import ImagePyramide, detach_kp, kp2gaussian

from ...modules.utils import spectral_norm


@DISCRIMINATORS.register()
class FirstOrderDiscriminator(nn.Layer):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    Args:
      discriminator_cfg:
        scales (list): extract the features of image pyramids
        block_expansion (int): block_expansion * (2**i) output features for each block i
        max_features (int): input features cannot larger than max_features for encoding images
        num_blocks (int): number of blocks for encoding images
        sn (bool): whether to use spentral norm
      common_params:
        num_kp (int): number of keypoints
        num_channels (int): image channels
        estimate_jacobian (bool): whether to estimate jacobian values of keypoints
      train_params:
        loss_weights:
            discriminator_gan (int): weight of discriminator loss
    """
    def __init__(self, discriminator_cfg, common_params, train_params):
        super(FirstOrderDiscriminator, self).__init__()
        self.discriminator = MultiScaleDiscriminator(**discriminator_cfg,
                                                     **common_params)
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, common_params['num_channels'])
        self.loss_weights = train_params['loss_weights']

    def forward(self, x, generated):
        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'].detach())

        kp_driving = generated['kp_driving']
        discriminator_maps_generated = self.discriminator(
            pyramide_generated, kp=detach_kp(kp_driving))
        discriminator_maps_real = self.discriminator(pyramide_real,
                                                     kp=detach_kp(kp_driving))

        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            value = (1 - discriminator_maps_real[key]
                     )**2 + discriminator_maps_generated[key]**2
            value_total += self.loss_weights['discriminator_gan'] * value.mean()
        loss_values['disc_gan'] = value_total

        return loss_values


class DownBlock2d(nn.Layer):
    """
    Simple block for processing video (encoder).
    """
    def __init__(self,
                 in_features,
                 out_features,
                 norm=False,
                 kernel_size=4,
                 pool=False,
                 sn=False):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2D(in_features,
                              out_features,
                              kernel_size=kernel_size)
        if sn:
            self.conv = spectral_norm(self.conv)
        else:
            self.sn = None
        if norm:
            self.norm = nn.InstanceNorm2D(num_features=out_features,
                                          epsilon=1e-05)
        else:
            self.norm = None

        self.pool = pool

    def forward(self, x):

        out = x
        out = self.conv(out)
        if self.norm is not None:
            out = self.norm(out)
        out = F.leaky_relu(out, 0.2)
        if self.pool:
            out = F.avg_pool2d(out, kernel_size=2, stride=2, ceil_mode=False)
        return out


class Discriminator(nn.Layer):
    def __init__(self,
                 num_channels=3,
                 block_expansion=64,
                 num_blocks=4,
                 max_features=512,
                 sn=False,
                 use_kp=False,
                 num_kp=10,
                 kp_variance=0.01,
                 **kwargs):
        super(Discriminator, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock2d(num_channels + num_kp * use_kp if i == 0 else min(
                    max_features, block_expansion * (2**i)),
                            min(max_features, block_expansion * (2**(i + 1))),
                            norm=(i != 0),
                            kernel_size=4,
                            pool=(i != num_blocks - 1),
                            sn=sn))

        self.down_blocks = nn.LayerList(down_blocks)
        self.conv = nn.Conv2D(self.down_blocks[len(self.down_blocks) -
                                               1].conv.parameters()[0].shape[0],
                              1,
                              kernel_size=1)
        if sn:
            self.conv = spectral_norm(self.conv)
        else:
            self.sn = None
        self.use_kp = use_kp
        self.kp_variance = kp_variance

    def forward(self, x, kp=None):
        feature_maps = []
        out = x

        if self.use_kp:
            heatmap = kp2gaussian(kp, x.shape[2:], self.kp_variance)
            out = paddle.concat([out, heatmap], axis=1)
        for down_block in self.down_blocks:
            out = down_block(out)
            feature_maps.append(out)
            out = feature_maps[-1]
        prediction_map = self.conv(out)
        return feature_maps, prediction_map


class MultiScaleDiscriminator(nn.Layer):
    """
    Multi-scale (scale) discriminator
    """
    def __init__(self, scales=(), **kwargs):
        super(MultiScaleDiscriminator, self).__init__()
        self.scales = scales
        self.discs = nn.LayerList()
        self.nameList = []
        for scale in scales:
            self.discs.add_sublayer(
                str(scale).replace('.', '-'), Discriminator(**kwargs))
            self.nameList.append(str(scale).replace('.', '-'))

    def forward(self, x, kp=None):
        out_dict = {}
        for scale, disc in zip(self.nameList, self.discs):
            scale = str(scale).replace('-', '.')
            key = 'prediction_' + scale
            feature_maps, prediction_map = disc(x[key], kp)
            out_dict['feature_maps_' + scale] = feature_maps
            out_dict['prediction_map_' + scale] = prediction_map
        return out_dict
