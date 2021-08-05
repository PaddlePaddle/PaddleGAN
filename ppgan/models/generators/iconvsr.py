#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# basicvsr and iconvsr code are heavily based on mmedit
import paddle
import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F

from .builder import GENERATORS
from .edvr import PCDAlign, TSAFusion
from .basicvsr import SPyNet, PixelShufflePack, ResidualBlockNoBN, \
                      ResidualBlocksWithInputConv, flow_warp
from ...utils.download import get_path_from_url


@GENERATORS.register()
class IconVSR(nn.Layer):
    """BasicVSR network structure for video super-resolution.

    Support only x4 upsampling.
    Paper:
        BasicVSR: The Search for Essential Components in Video Super-Resolution
        and Beyond, CVPR, 2021

    Args:
        mid_channels (int): Channel number of the intermediate features.
            Default: 64.
        num_blocks (int): Number of residual blocks in each propagation branch.
            Default: 30.
        padding (int): Number of frames to be padded at two ends of the
            sequence. 2 for REDS and 3 for Vimeo-90K. Default: 2.
        keyframe_stride (int): Number determining the keyframes. If stride=5,
            then the (0, 5, 10, 15, ...)-th frame will be the keyframes.
            Default: 5.
    """
    def __init__(self,
                 mid_channels=64,
                 num_blocks=30,
                 padding=2,
                 keyframe_stride=5):

        super().__init__()

        self.mid_channels = mid_channels
        self.padding = padding
        self.keyframe_stride = keyframe_stride

        # optical flow network for feature alignment
        self.spynet = SPyNet()
        weight_path = get_path_from_url(
            'https://paddlegan.bj.bcebos.com/models/spynet.pdparams')
        self.spynet.set_state_dict(paddle.load(weight_path))

        # information-refill
        self.edvr = EDVRFeatureExtractor(num_frames=padding * 2 + 1,
                                         center_frame_idx=padding)

        edvr_wight_path = get_path_from_url(
            'https://paddlegan.bj.bcebos.com/models/edvrm.pdparams')
        self.edvr.set_state_dict(paddle.load(edvr_wight_path))

        self.backward_fusion = nn.Conv2D(2 * mid_channels,
                                         mid_channels,
                                         3,
                                         1,
                                         1,
                                         bias_attr=True)
        self.forward_fusion = nn.Conv2D(2 * mid_channels,
                                        mid_channels,
                                        3,
                                        1,
                                        1,
                                        bias_attr=True)

        # propagation branches
        self.backward_resblocks = ResidualBlocksWithInputConv(
            mid_channels + 3, mid_channels, num_blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(
            2 * mid_channels + 3, mid_channels, num_blocks)

        # upsample
        # self.fusion = nn.Conv2D(mid_channels * 2, mid_channels, 1, 1, 0)
        self.upsample1 = PixelShufflePack(mid_channels,
                                          mid_channels,
                                          2,
                                          upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels,
                                          64,
                                          2,
                                          upsample_kernel=3)
        self.conv_hr = nn.Conv2D(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2D(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4,
                                        mode='bilinear',
                                        align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

    def spatial_padding(self, lrs):
        """ Apply pdding spatially.

        Since the PCD module in EDVR requires that the resolution is a multiple
        of 4, we apply padding to the input LR images if their resolution is
        not divisible by 4.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Padded LR sequence with shape (n, t, c, h_pad, w_pad).

        """
        n, t, c, h, w = lrs.shape

        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4

        # padding
        lrs = lrs.reshape([-1, c, h, w])
        lrs = F.pad(lrs, [0, pad_w, 0, pad_h], mode='reflect')

        return lrs.reshape([n, t, c, h + pad_h, w + pad_w])

    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)

        Returns:
            bool: whether the input is a mirror-extended sequence
        """

        self.is_mirror_extended = False
        if lrs.shape[1] % 2 == 0:
            lrs_1, lrs_2 = paddle.chunk(lrs, 2, axis=1)
            lrs_2 = paddle.flip(lrs_2, [1])
            if paddle.norm(lrs_1 - lrs_2) == 0:
                self.is_mirror_extended = True

    def compute_refill_features(self, lrs, keyframe_idx):
        """ Compute keyframe features for information-refill.
        Since EDVR-M is used, padding is performed before feature computation.
        Args:
            lrs (Tensor): Input LR images with shape (n, t, c, h, w)
            keyframe_idx (list(int)): The indices specifying the keyframes.
        Return:
            dict(Tensor): The keyframe features. Each key corresponds to the
                indices in keyframe_idx.
        """

        if self.padding == 2:
            lrs = [
                lrs[:, 4:5, :, :], lrs[:, 3:4, :, :], lrs, lrs[:, -4:-3, :, :],
                lrs[:, -5:-4, :, :]
            ]
        elif self.padding == 3:
            lrs = [lrs[:, [6, 5, 4]], lrs, lrs[:, [-5, -6, -7]]]
        lrs = paddle.concat(lrs, axis=1)

        num_frames = 2 * self.padding + 1
        feats_refill = {}
        for i in keyframe_idx:
            feats_refill[i] = self.edvr(lrs[:, i:i + num_frames])
        return feats_refill

    def compute_flow(self, lrs):
        """Compute optical flow using SPyNet for feature warping.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lrs.shape

        lrs_1 = lrs[:, :-1, :, :, :].reshape([-1, c, h, w])
        lrs_2 = lrs[:, 1:, :, :, :].reshape([-1, c, h, w])

        flows_backward = self.spynet(lrs_1, lrs_2).reshape([n, t - 1, 2, h, w])

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lrs_2,
                                        lrs_1).reshape([n, t - 1, 2, h, w])

        return flows_forward, flows_backward

    def forward(self, lrs):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        n, t, c, h_input, w_input = lrs.shape
        assert h_input >= 64 and w_input >= 64, (
            'The height and width of inputs should be at least 64, '
            f'but got {h_input} and {w_input}.')

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lrs)

        lrs = self.spatial_padding(lrs)
        h, w = lrs.shape[3], lrs.shape[4]

        # get the keyframe indices for information-refill
        keyframe_idx = list(range(0, t, self.keyframe_stride))
        if keyframe_idx[-1] != t - 1:
            keyframe_idx.append(t - 1)  # the last frame must be a keyframe

        # compute optical flow and compute features for information-refill
        flows_forward, flows_backward = self.compute_flow(lrs)
        feats_refill = self.compute_refill_features(lrs, keyframe_idx)
        # compute optical flow
        flows_forward, flows_backward = self.compute_flow(lrs)

        # backward-time propgation
        outputs = []

        feat_prop = paddle.to_tensor(
            np.zeros([n, self.mid_channels, h, w], 'float32'))
        for i in range(t - 1, -1, -1):
            # no warping required for the last timestep
            if i < t - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.transpose([0, 2, 3, 1]))

            # information refill
            if i in keyframe_idx:
                feat_prop = paddle.concat([feat_prop, feats_refill[i]], axis=1)
                feat_prop = self.backward_fusion(feat_prop)

            feat_prop = paddle.concat([lrs[:, i, :, :, :], feat_prop], axis=1)
            feat_prop = self.backward_resblocks(feat_prop)

            outputs.append(feat_prop)
        outputs = outputs[::-1]

        # forward-time propagation and upsampling
        feat_prop = paddle.zeros_like(feat_prop)
        for i in range(0, t):
            lr_curr = lrs[:, i, :, :, :]
            if i > 0:  # no warping required for the first timestep
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_backward[:, -i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.transpose([0, 2, 3, 1]))

            # information refill
            if i in keyframe_idx:
                feat_prop = paddle.concat([feat_prop, feats_refill[i]], axis=1)
                feat_prop = self.forward_fusion(feat_prop)

            feat_prop = paddle.concat([lr_curr, outputs[i], feat_prop], axis=1)
            feat_prop = self.forward_resblocks(feat_prop)

            # upsampling given the backward and forward features
            out = self.lrelu(self.upsample1(feat_prop))
            out = self.lrelu(self.upsample2(out))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = self.img_upsample(lr_curr)
            out += base
            outputs[i] = out

        return paddle.stack(outputs, axis=1)


class EDVRFeatureExtractor(nn.Layer):
    """EDVR feature extractor for information-refill in IconVSR.

    We use EDVR-M in IconVSR. To adopt pretrained models, please
    specify "pretrained".

    Paper:
    EDVR: Video Restoration with Enhanced Deformable Convolutional Networks.
    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        num_frames (int): Number of input frames. Default: 5.
        deform_groups (int): Deformable groups. Defaults: 8.
        num_blocks_extraction (int): Number of blocks for feature extraction.
            Default: 5.
        num_blocks_reconstruction (int): Number of blocks for reconstruction.
            Default: 10.
        center_frame_idx (int): The index of center frame. Frame counting from
            0. Default: 2.
        with_tsa (bool): Whether to use TSA module. Default: True.
    """
    def __init__(self,
                 in_channels=3,
                 out_channel=3,
                 mid_channels=64,
                 num_frames=5,
                 deform_groups=8,
                 num_blocks_extraction=5,
                 num_blocks_reconstruction=10,
                 center_frame_idx=2,
                 with_tsa=True):

        super().__init__()

        self.center_frame_idx = center_frame_idx
        self.with_tsa = with_tsa

        self.conv_first = nn.Conv2D(in_channels, mid_channels, 3, 1, 1)
        self.feature_extraction = make_layer(ResidualBlockNoBN,
                                             num_blocks_extraction,
                                             nf=mid_channels)

        # generate pyramid features
        self.feat_l2_conv1 = nn.Conv2D(mid_channels, mid_channels, 3, 2, 1)
        self.feat_l2_conv2 = nn.Conv2D(mid_channels, mid_channels, 3, 1, 1)
        self.feat_l3_conv1 = nn.Conv2D(mid_channels, mid_channels, 3, 2, 1)
        self.feat_l3_conv2 = nn.Conv2D(mid_channels, mid_channels, 3, 1, 1)

        # pcd alignment
        self.pcd_alignment = PCDAlign(nf=mid_channels, groups=deform_groups)
        # fusion
        if self.with_tsa:
            self.fusion = TSAFusion(nf=mid_channels,
                                    nframes=num_frames,
                                    center=self.center_frame_idx)
        else:
            self.fusion = nn.Conv2D(num_frames * mid_channels, mid_channels, 1,
                                    1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        """Forward function for EDVRFeatureExtractor.
        Args:
            x (Tensor): Input tensor with shape (n, t, 3, h, w).
        Returns:
            Tensor: Intermediate feature with shape (n, mid_channels, h, w).
        """

        n, t, c, h, w = x.shape

        # extract LR features
        # L1
        l1_feat = self.lrelu(self.conv_first(x.reshape([-1, c, h, w])))
        l1_feat = self.feature_extraction(l1_feat)
        # L2
        l2_feat = self.lrelu(
            self.feat_l2_conv2(self.lrelu(self.feat_l2_conv1(l1_feat))))
        # L3
        l3_feat = self.lrelu(
            self.feat_l3_conv2(self.lrelu(self.feat_l3_conv1(l2_feat))))

        l1_feat = l1_feat.reshape([n, t, -1, h, w])
        l2_feat = l2_feat.reshape([n, t, -1, h // 2, w // 2])
        l3_feat = l3_feat.reshape([n, t, -1, h // 4, w // 4])

        # pcd alignment
        ref_feats = [  # reference feature list
            l1_feat[:, self.center_frame_idx, :, :, :].clone(),
            l2_feat[:, self.center_frame_idx, :, :, :].clone(),
            l3_feat[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(t):
            neighbor_feats = [
                l1_feat[:, i, :, :, :].clone(), l2_feat[:, i, :, :, :].clone(),
                l3_feat[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_alignment(neighbor_feats, ref_feats))
        aligned_feat = paddle.stack(aligned_feat, axis=1)  # (n, t, c, h, w)

        if self.with_tsa:
            feat = self.fusion(aligned_feat)
        else:
            aligned_feat = aligned_feat.reshape([n, -1, h, w])
            feat = self.fusion(aligned_feat)

        return feat


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        block (nn.Layer): nn.module class for basic block.
        num_blocks (int): number of blocks.
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)
