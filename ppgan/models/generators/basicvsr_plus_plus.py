#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ...utils.download import get_path_from_url
from .basicvsr import PixelShufflePack, flow_warp, SPyNet, \
                      ResidualBlocksWithInputConv, SecondOrderDeformableAlignment
from .builder import GENERATORS


@GENERATORS.register()
class BasicVSRPlusPlus(nn.Layer):
    """BasicVSR++ network structure.
    Support either x4 upsampling or same size output. Since DCN is used in this
    model, it can only be used with CUDA enabled.
    Paper:
        BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation
        and Alignment

    Adapted from 'https://github.com/open-mmlab/mmediting'
    'mmediting/blob/master/mmedit/models/backbones/sr_backbones/basicvsr_pp.py'
    Copyright (c) MMEditing Authors.

    Args:
        mid_channels (int, optional): Channel number of the intermediate
            features. Default: 64.
        num_blocks (int, optional): The number of residual blocks in each
            propagation branch. Default: 7.
        is_low_res_input (bool, optional): Whether the input is low-resolution
            or not. If False, the output resolution is equal to the input
            resolution. Default: True.
    """
    def __init__(self, mid_channels=64, num_blocks=7, is_low_res_input=True):

        super().__init__()

        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input

        # optical flow
        self.spynet = SPyNet()
        weight_path = get_path_from_url(
            'https://paddlegan.bj.bcebos.com/models/spynet.pdparams')
        self.spynet.set_state_dict(paddle.load(weight_path))

        # feature extraction module
        if is_low_res_input:
            self.feat_extract = ResidualBlocksWithInputConv(3, mid_channels, 5)
        else:
            self.feat_extract = nn.Sequential(
                nn.Conv2D(3, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2D(mid_channels, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1),
                ResidualBlocksWithInputConv(mid_channels, mid_channels, 5))

        # propagation branches
        self.deform_align_backward_1 = SecondOrderDeformableAlignment(
            2 * mid_channels, mid_channels, 3, padding=1, deformable_groups=16)
        self.deform_align_forward_1 = SecondOrderDeformableAlignment(
            2 * mid_channels, mid_channels, 3, padding=1, deformable_groups=16)
        self.deform_align_backward_2 = SecondOrderDeformableAlignment(
            2 * mid_channels, mid_channels, 3, padding=1, deformable_groups=16)
        self.deform_align_forward_2 = SecondOrderDeformableAlignment(
            2 * mid_channels, mid_channels, 3, padding=1, deformable_groups=16)
        self.backbone_backward_1 = ResidualBlocksWithInputConv(
            2 * mid_channels, mid_channels, num_blocks)
        self.backbone_forward_1 = ResidualBlocksWithInputConv(
            3 * mid_channels, mid_channels, num_blocks)
        self.backbone_backward_2 = ResidualBlocksWithInputConv(
            4 * mid_channels, mid_channels, num_blocks)
        self.backbone_forward_2 = ResidualBlocksWithInputConv(
            5 * mid_channels, mid_channels, num_blocks)

        # upsampling module
        self.reconstruction = ResidualBlocksWithInputConv(
            5 * mid_channels, mid_channels, 5)
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

    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.
        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.
        Args:
            lqs (tensor): Input LR images with shape (n, t, c, h, w)
        """

        with paddle.no_grad():
            self.is_mirror_extended = False
            if lrs.shape[1] % 2 == 0:
                lrs_1, lrs_2 = paddle.chunk(lrs, 2, axis=1)
                lrs_2 = paddle.flip(lrs_2, [1])
                if paddle.norm(lrs_1 - lrs_2) == 0:
                    self.is_mirror_extended = True

    def compute_flow(self, lrs):
        """Compute optical flow using SPyNet for feature alignment.
        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.
        Args:
            lqs (tensor): Input LR images with shape (n, t, c, h, w)
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

        if self.is_mirror_extended:
            flows_forward = flows_backward.flip(1)
        else:
            flows_forward = self.spynet(lrs_2,
                                        lrs_1).reshape([n, t - 1, 2, h, w])

        return flows_forward, flows_backward

    def upsample(self, lqs, feats):
        """Compute the output image given the features.
        Args:
            lqs (tensor): Input LR images with shape (n, t, c, h, w).
            feats (dict): The features from the propgation branches.
        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        outputs = []
        num_outputs = len(feats['spatial'])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.shape[1]):
            hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            hr.insert(0, feats['spatial'][mapping_idx[i]])
            hr = paddle.concat(hr, axis=1)

            hr = self.reconstruction(hr)
            hr = self.lrelu(self.upsample1(hr))
            hr = self.lrelu(self.upsample2(hr))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            if self.is_low_res_input:
                hr += self.img_upsample(lqs[:, i, :, :, :])
            else:
                hr += lqs[:, i, :, :, :]

            outputs.append(hr)

        return paddle.stack(outputs, axis=1)

    def forward(self, lqs):
        """Forward function for BasicVSR++.
        Args:
            lqs (Tensor): Input LR sequence with shape (n, t, c, h, w).
        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        n, t, c, h, w = lqs.shape

        if self.is_low_res_input:
            lqs_downsample = lqs
        else:
            lqs_downsample = F.interpolate(lqs.reshape([-1, c, h, w]),
                                           scale_factor=0.25,
                                           mode='bicubic').reshape(
                                               [n, t, c, h // 4, w // 4])

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lqs)

        feats = {}
        feats_ = self.feat_extract(lqs.reshape([-1, c, h, w]))
        h, w = feats_.shape[2:]
        feats_ = feats_.reshape([n, t, -1, h, w])
        feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]

        # compute optical flow using the low-res inputs
        assert lqs_downsample.shape[3] >= 64 and lqs_downsample.shape[4] >= 64, (
            'The height and width of low-res inputs must be at least 64, '
            f'but got {h} and {w}.')
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        # feature propgation

        # backward_1
        feats['backward_1'] = []
        flows = flows_backward

        n, t, _, h, w = flows.shape

        frame_idx = range(t, -1, -1)
        flow_idx = range(t, -1, -1)
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]

        feat_prop = paddle.zeros([n, self.mid_channels, h, w])

        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]

            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                cond_n1 = flow_warp(feat_prop, flow_n1.transpose([0, 2, 3, 1]))

                # initialize second-order features
                feat_n2 = paddle.zeros_like(feat_prop)
                flow_n2 = paddle.zeros_like(flow_n1)
                cond_n2 = paddle.zeros_like(cond_n1)

                if i > 1:  # second-order features
                    feat_n2 = feats['backward_1'][-2]

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]

                    flow_n2 = flow_n1 + flow_warp(
                        flow_n2, flow_n1.transpose([0, 2, 3, 1]))

                    cond_n2 = flow_warp(feat_n2, flow_n2.transpose([0, 2, 3,
                                                                    1]))

                # flow-guided deformable convolution
                cond = paddle.concat([cond_n1, feat_current, cond_n2], axis=1)
                feat_prop = paddle.concat([feat_prop, feat_n2], axis=1)

                feat_prop = self.deform_align_backward_1(
                    feat_prop, cond, flow_n1, flow_n2)

            # concatenate and residual blocks
            feat = [feat_current] + [
                feats[k][idx]
                for k in feats if k not in ['spatial', 'backward_1']
            ] + [feat_prop]

            feat = paddle.concat(feat, axis=1)
            feat_prop = feat_prop + self.backbone_backward_1(feat)
            feats['backward_1'].append(feat_prop)

        feats['backward_1'] = feats['backward_1'][::-1]

        # forward_1
        feats['forward_1'] = []
        flows = flows_forward

        n, t, _, h, w = flows.shape

        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]

        feat_prop = paddle.zeros([n, self.mid_channels, h, w])

        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]

            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                cond_n1 = flow_warp(feat_prop, flow_n1.transpose([0, 2, 3, 1]))

                # initialize second-order features
                feat_n2 = paddle.zeros_like(feat_prop)
                flow_n2 = paddle.zeros_like(flow_n1)
                cond_n2 = paddle.zeros_like(cond_n1)

                if i > 1:  # second-order features
                    feat_n2 = feats['forward_1'][-2]

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]

                    flow_n2 = flow_n1 + flow_warp(
                        flow_n2, flow_n1.transpose([0, 2, 3, 1]))

                    cond_n2 = flow_warp(feat_n2, flow_n2.transpose([0, 2, 3,
                                                                    1]))

                # flow-guided deformable convolution
                cond = paddle.concat([cond_n1, feat_current, cond_n2], axis=1)
                feat_prop = paddle.concat([feat_prop, feat_n2], axis=1)

                feat_prop = self.deform_align_forward_1(feat_prop, cond,
                                                        flow_n1, flow_n2)

            # concatenate and residual blocks
            feat = [feat_current] + [
                feats[k][idx]
                for k in feats if k not in ['spatial', 'forward_1']
            ] + [feat_prop]

            feat = paddle.concat(feat, axis=1)
            feat_prop = feat_prop + self.backbone_forward_1(feat)
            feats['forward_1'].append(feat_prop)

        # backward_2
        feats['backward_2'] = []
        flows = flows_backward

        n, t, _, h, w = flows.shape

        frame_idx = range(t, -1, -1)
        flow_idx = range(t, -1, -1)
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]

        feat_prop = paddle.zeros([n, self.mid_channels, h, w])

        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]

            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                cond_n1 = flow_warp(feat_prop, flow_n1.transpose([0, 2, 3, 1]))

                # initialize second-order features
                feat_n2 = paddle.zeros_like(feat_prop)
                flow_n2 = paddle.zeros_like(flow_n1)
                cond_n2 = paddle.zeros_like(cond_n1)

                if i > 1:  # second-order features
                    feat_n2 = feats['backward_2'][-2]

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]

                    flow_n2 = flow_n1 + flow_warp(
                        flow_n2, flow_n1.transpose([0, 2, 3, 1]))

                    cond_n2 = flow_warp(feat_n2, flow_n2.transpose([0, 2, 3,
                                                                    1]))

                # flow-guided deformable convolution
                cond = paddle.concat([cond_n1, feat_current, cond_n2], axis=1)
                feat_prop = paddle.concat([feat_prop, feat_n2], axis=1)

                feat_prop = self.deform_align_backward_2(
                    feat_prop, cond, flow_n1, flow_n2)

            # concatenate and residual blocks
            feat = [feat_current] + [
                feats[k][idx]
                for k in feats if k not in ['spatial', 'backward_2']
            ] + [feat_prop]

            feat = paddle.concat(feat, axis=1)
            feat_prop = feat_prop + self.backbone_backward_2(feat)
            feats['backward_2'].append(feat_prop)

        feats['backward_2'] = feats['backward_2'][::-1]

        # forward_2
        feats['forward_2'] = []
        flows = flows_forward

        n, t, _, h, w = flows.shape

        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]

        feat_prop = paddle.zeros([n, self.mid_channels, h, w])

        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]

            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                cond_n1 = flow_warp(feat_prop, flow_n1.transpose([0, 2, 3, 1]))

                # initialize second-order features
                feat_n2 = paddle.zeros_like(feat_prop)
                flow_n2 = paddle.zeros_like(flow_n1)
                cond_n2 = paddle.zeros_like(cond_n1)

                if i > 1:  # second-order features
                    feat_n2 = feats['forward_2'][-2]

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]

                    flow_n2 = flow_n1 + flow_warp(
                        flow_n2, flow_n1.transpose([0, 2, 3, 1]))

                    cond_n2 = flow_warp(feat_n2, flow_n2.transpose([0, 2, 3,
                                                                    1]))

                # flow-guided deformable convolution
                cond = paddle.concat([cond_n1, feat_current, cond_n2], axis=1)
                feat_prop = paddle.concat([feat_prop, feat_n2], axis=1)

                feat_prop = self.deform_align_forward_2(feat_prop, cond,
                                                        flow_n1, flow_n2)

            # concatenate and residual blocks
            feat = [feat_current] + [
                feats[k][idx]
                for k in feats if k not in ['spatial', 'forward_2']
            ] + [feat_prop]

            feat = paddle.concat(feat, axis=1)
            feat_prop = feat_prop + self.backbone_forward_2(feat)
            feats['forward_2'].append(feat_prop)

        return self.upsample(lqs, feats)
