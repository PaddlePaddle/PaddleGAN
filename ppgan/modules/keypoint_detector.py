# code was heavily based on https://github.com/AliaksandrSiarohin/first-order-model
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/AliaksandrSiarohin/first-order-model/blob/master/LICENSE.md

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from .first_order import Hourglass, make_coordinate_grid, AntiAliasInterpolation2d


class KPDetector(nn.Layer):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """
    def __init__(self,
                 block_expansion,
                 num_kp,
                 num_channels,
                 max_features,
                 num_blocks,
                 temperature,
                 estimate_jacobian=False,
                 scale_factor=1,
                 single_jacobian_map=False,
                 pad=0,
                 mobile_net=False):
        super(KPDetector, self).__init__()

        self.predictor = Hourglass(block_expansion,
                                   in_features=num_channels,
                                   max_features=max_features,
                                   num_blocks=num_blocks,
                                   mobile_net=mobile_net)
        if mobile_net:
            self.kp = nn.Sequential(
                nn.Conv2D(in_channels=self.predictor.out_filters,
                                out_channels=self.predictor.out_filters,
                                kernel_size=3,
                                weight_attr=nn.initializer.KaimingUniform(),
                                padding=pad), 
                nn.Conv2D(in_channels=self.predictor.out_filters,
                                out_channels=self.predictor.out_filters,
                                kernel_size=3,
                                weight_attr=nn.initializer.KaimingUniform(),
                                padding=pad),
                nn.Conv2D(in_channels=self.predictor.out_filters,
                                out_channels=num_kp,
                                kernel_size=3,
                                weight_attr=nn.initializer.KaimingUniform(),
                                padding=pad))
        else:
            self.kp = nn.Conv2D(in_channels=self.predictor.out_filters,
                            out_channels=num_kp,
                            kernel_size=(7, 7),
                            padding=pad)

        if estimate_jacobian:
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            if mobile_net:
                self.jacobian = nn.Sequential(
                    nn.Conv2D(in_channels=self.predictor.out_filters,
                        out_channels=self.predictor.out_filters,
                        kernel_size=3,
                        padding=pad),
                    nn.Conv2D(in_channels=self.predictor.out_filters,
                            out_channels=self.predictor.out_filters,
                            kernel_size=3,
                            padding=pad),
                    nn.Conv2D(in_channels=self.predictor.out_filters,
                            out_channels=4 * self.num_jacobian_maps,
                            kernel_size=3,
                            padding=pad),
                )
                self.jacobian[0].weight.set_value(
                    paddle.zeros(self.jacobian[0].weight.shape, dtype='float32'))
                self.jacobian[1].weight.set_value(
                    paddle.zeros(self.jacobian[1].weight.shape, dtype='float32'))
                self.jacobian[2].weight.set_value(
                    paddle.zeros(self.jacobian[2].weight.shape, dtype='float32'))
                self.jacobian[2].bias.set_value(
                    paddle.to_tensor([1, 0, 0, 1] *
                                    self.num_jacobian_maps).astype('float32'))
            else:
                self.jacobian = nn.Conv2D(in_channels=self.predictor.out_filters,
                                      out_channels=4 * self.num_jacobian_maps,
                                      kernel_size=(7, 7),
                                      padding=pad)
                self.jacobian.weight.set_value(
                    paddle.zeros(self.jacobian.weight.shape, dtype='float32'))
                self.jacobian.bias.set_value(
                    paddle.to_tensor([1, 0, 0, 1] *
                                 self.num_jacobian_maps).astype('float32'))
        else:
            self.jacobian = None

        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels,
                                                 self.scale_factor,
                                                 mobile_net=mobile_net)

    def gaussian2kp(self, heatmap):
        """
        Extract the mean and from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:]).unsqueeze([0, 1])
        value = (heatmap * grid).sum(axis=(2, 3))
        kp = {'value': value}
        return kp

    def forward(self, x):
        if self.scale_factor != 1:
            x = self.down(x)
        feature_map = self.predictor(x)
        prediction = self.kp(feature_map)

        final_shape = prediction.shape
        heatmap = prediction.reshape([final_shape[0], final_shape[1], -1])
        heatmap = F.softmax(heatmap / self.temperature, axis=2)
        heatmap = heatmap.reshape(final_shape)
        out = self.gaussian2kp(heatmap)

        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map)
            jacobian_map = jacobian_map.reshape([
                final_shape[0], self.num_jacobian_maps, 4, final_shape[2],
                final_shape[3]
            ])
            heatmap = heatmap.unsqueeze(2)
            heatmap = paddle.tile(heatmap, [1, 1, 4, 1, 1])
            jacobian = heatmap * jacobian_map
            jacobian = jacobian.reshape([final_shape[0], final_shape[1], 4, -1])
            jacobian = jacobian.sum(axis=-1)
            jacobian = jacobian.reshape(
                [jacobian.shape[0], jacobian.shape[1], 2, 2])
            out['jacobian'] = jacobian

        return out
