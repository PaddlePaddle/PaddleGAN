# code was heavily based on https://github.com/AliaksandrSiarohin/first-order-model
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/AliaksandrSiarohin/first-order-model/blob/master/LICENSE.md

import paddle
from paddle import nn
import paddle.nn.functional as F
from ...modules.first_order import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d, make_coordinate_grid
from ...modules.first_order import MobileResBlock2d, MobileUpBlock2d, MobileDownBlock2d
from ...modules.dense_motion import DenseMotionNetwork
import numpy as np
import cv2


class OcclusionAwareGenerator(nn.Layer):
    """
    Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints. Generator follows Johnson architecture.
    """
    def __init__(self,
                 num_channels,
                 num_kp,
                 block_expansion,
                 max_features,
                 num_down_blocks,
                 num_bottleneck_blocks,
                 estimate_occlusion_map=False,
                 dense_motion_params=None,
                 estimate_jacobian=False,
                 inference=False,
                 mobile_net=False):
        super(OcclusionAwareGenerator, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(
                num_kp=num_kp,
                num_channels=num_channels,
                estimate_occlusion_map=estimate_occlusion_map,
                **dense_motion_params,
                mobile_net=mobile_net)
        else:
            self.dense_motion_network = None

        if mobile_net:
            self.first = nn.Sequential(
                    SameBlock2d(num_channels,
                                    num_channels,
                                    kernel_size=3,
                                    padding=1,
                                    mobile_net=mobile_net),
                    SameBlock2d(num_channels,
                                    num_channels,
                                    kernel_size=3,
                                    padding=1,
                                    mobile_net=mobile_net),
                    SameBlock2d(num_channels,
                                    block_expansion,
                                    kernel_size=3,
                                    padding=1,
                                    mobile_net=mobile_net)
                )
        else:
            self.first = SameBlock2d(num_channels,
                                 block_expansion,
                                 kernel_size=(7, 7),
                                 padding=(3, 3),
                                 mobile_net=mobile_net)

        down_blocks = []
        if mobile_net:
            for i in range(num_down_blocks):
                in_features = min(max_features, block_expansion * (2**i))
                out_features = min(max_features, block_expansion * (2**(i + 1)))
                down_blocks.append(
                    MobileDownBlock2d(in_features,
                                      out_features,
                                      kernel_size=(3, 3),
                                      padding=(1, 1)))
        else:
            for i in range(num_down_blocks):
                in_features = min(max_features, block_expansion * (2**i))
                out_features = min(max_features, block_expansion * (2**(i + 1)))
                down_blocks.append(
                    DownBlock2d(in_features,
                                out_features,
                                kernel_size=(3, 3),
                                padding=(1, 1)))
        self.down_blocks = nn.LayerList(down_blocks)

        up_blocks = []
        if mobile_net:
            for i in range(num_down_blocks):
                in_features = min(max_features,
                                  block_expansion * (2**(num_down_blocks - i)))
                out_features = min(
                    max_features,
                    block_expansion * (2**(num_down_blocks - i - 1)))
                up_blocks.append(
                    MobileUpBlock2d(in_features,
                                    out_features,
                                    kernel_size=(3, 3),
                                    padding=(1, 1)))
        else:
            for i in range(num_down_blocks):
                in_features = min(max_features,
                                  block_expansion * (2**(num_down_blocks - i)))
                out_features = min(
                    max_features,
                    block_expansion * (2**(num_down_blocks - i - 1)))
                up_blocks.append(
                    UpBlock2d(in_features,
                              out_features,
                              kernel_size=(3, 3),
                              padding=(1, 1)))
        self.up_blocks = nn.LayerList(up_blocks)

        self.bottleneck = paddle.nn.Sequential()
        in_features = min(max_features, block_expansion * (2**num_down_blocks))
        if mobile_net:
            for i in range(num_bottleneck_blocks):
                self.bottleneck.add_sublayer(
                    'r' + str(i),
                    MobileResBlock2d(in_features,
                                     kernel_size=(3, 3),
                                     padding=(1, 1)))
        else:
            for i in range(num_bottleneck_blocks):
                self.bottleneck.add_sublayer(
                    'r' + str(i),
                    ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))
        if mobile_net:
            self.final = nn.Sequential(
                    nn.Conv2D(block_expansion,
                                block_expansion,
                                kernel_size=3,
                                weight_attr=nn.initializer.KaimingUniform(),
                                padding=1),
                    nn.ReLU(),
                    nn.Conv2D(block_expansion,
                                block_expansion,
                                kernel_size=3,
                                weight_attr=nn.initializer.KaimingUniform(),
                                padding=1),
                    nn.ReLU(),
                    nn.Conv2D(block_expansion,
                                num_channels,
                                kernel_size=3,
                                weight_attr=nn.initializer.KaimingUniform(),
                                padding=1)
                )
        else:
            self.final = nn.Conv2D(block_expansion,
                               num_channels,
                               kernel_size=(7, 7),
                               padding=(3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels
        self.inference = inference
        self.pad = 5
        self.mobile_net = mobile_net

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.transpose([0, 3, 1, 2])
            deformation = F.interpolate(deformation,
                                        size=(h, w),
                                        mode='bilinear',
                                        align_corners=False)
            deformation = deformation.transpose([0, 2, 3, 1])
        if self.inference:
            identity_grid = make_coordinate_grid((h, w), type=inp.dtype)
            identity_grid = identity_grid.reshape([1, h, w, 2])
            visualization_matrix = np.zeros((h, w)).astype("float32")
            visualization_matrix[self.pad:h - self.pad,
                                 self.pad:w - self.pad] = 1.0
            gauss_kernel = paddle.to_tensor(
                cv2.GaussianBlur(visualization_matrix, (9, 9),
                                 0.0,
                                 borderType=cv2.BORDER_ISOLATED))
            gauss_kernel = gauss_kernel.unsqueeze(0).unsqueeze(-1)
            deformation = gauss_kernel * deformation + (
                1 - gauss_kernel) * identity_grid

        return F.grid_sample(inp,
                             deformation,
                             mode='bilinear',
                             padding_mode='zeros',
                             align_corners=True)

    def forward(self, source_image, kp_driving, kp_source):
        # Encoding (downsampling) part
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)

        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(source_image=source_image,
                                                     kp_driving=kp_driving,
                                                     kp_source=kp_source)
            output_dict['mask'] = dense_motion['mask']
            output_dict['sparse_deformed'] = dense_motion['sparse_deformed']

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None
            deformation = dense_motion['deformation']
            out = self.deform_input(out, deformation)

            if occlusion_map is not None:
                if out.shape[2] != occlusion_map.shape[2] or out.shape[
                        3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map,
                                                  size=out.shape[2:],
                                                  mode='bilinear',
                                                  align_corners=False)
                if self.inference and not self.mobile_net:
                    h, w = occlusion_map.shape[2:]
                    occlusion_map[:, :, 0:self.pad, :] = 1.0
                    occlusion_map[:, :, :, 0:self.pad] = 1.0
                    occlusion_map[:, :, h - self.pad:h, :] = 1.0
                    occlusion_map[:, :, :, w - self.pad:w] = 1.0
                out = out * occlusion_map

            output_dict["deformed"] = self.deform_input(source_image,
                                                        deformation)

        # Decoding part
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        out = F.sigmoid(out)

        output_dict["prediction"] = out

        return output_dict
