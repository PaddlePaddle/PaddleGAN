#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.ops import DeformConv2D

from .basicvsr import PixelShufflePack, flow_warp, SPyNet, ResidualBlocksWithInputConv
from ...utils.download import get_path_from_url
from ...modules.init import kaiming_normal_, constant_
from .builder import GENERATORS


@GENERATORS.register()
class MSVSR(nn.Layer):
    """PP-MSVSR network structure for video super-resolution.

    Support only x4 upsampling.
    Paper:
        PP-MSVSR: Multi-Stage Video Super-Resolution, 2021

    Args:
        mid_channels (int): Channel number of the intermediate features.
            Default: 32.
        num_init_blocks (int): Number of residual blocks in feat_extract.
            Default: 2.
        num_blocks (int): Number of residual blocks in each propagation branch.
            Default: 3.
        num_reconstruction_blocks (int): Number of residual blocks in reconstruction.
            Default: 2.
        only_last (bool): Whether the hr feature only do the last convolution.
            Default: True.
        use_tiny_spynet (bool): Whether use tiny spynet.
            Default: True.
        deform_groups (int): Number of deformable_groups in DeformConv2D in stage2 and stage3.
            Defaults: 4.
        stage1_groups (int): Number of deformable_groups in DeformConv2D in stage1.
            Defaults: 8.
        auxiliary_loss (bool): Whether use auxiliary loss.
            Default: True.
        use_refine_align (bool): Whether use refine align.
            Default: True.
        aux_reconstruction_blocks : Number of residual blocks in auxiliary reconstruction.
            Default: 1.
        use_local_connnect (bool): Whether add feature of stage1 after upsample.
            Default: True.
    """
    def __init__(self,
                 mid_channels=32,
                 num_init_blocks=2,
                 num_blocks=3,
                 num_reconstruction_blocks=2,
                 only_last=True,
                 use_tiny_spynet=True,
                 deform_groups=4,
                 stage1_groups=8,
                 auxiliary_loss=True,
                 use_refine_align=True,
                 aux_reconstruction_blocks=1,
                 use_local_connnect=True):

        super().__init__()

        self.mid_channels = mid_channels
        self.only_last = only_last
        self.deform_groups = deform_groups
        self.auxiliary_loss = auxiliary_loss
        self.use_refine_align = use_refine_align
        self.use_local_connnect = use_local_connnect

        # optical flow module
        if use_tiny_spynet:
            self.spynet = ModifiedSPyNet(num_blocks=3, use_tiny_block=True)
            weight_path = get_path_from_url(
                'https://paddlegan.bj.bcebos.com/models/modified_spynet_tiny.pdparams'
            )
            self.spynet.set_state_dict(paddle.load(weight_path))
        else:
            self.spynet = ModifiedSPyNet(num_blocks=6, use_tiny_block=False)
            weight_path = get_path_from_url(
                'https://paddlegan.bj.bcebos.com/models/modified_spynet.pdparams'
            )
            self.spynet.set_state_dict(paddle.load(weight_path))

        # feature extraction module
        self.feat_extract = ResidualBlocksWithInputConv(3, mid_channels,
                                                        num_init_blocks)

        # propagation branches module for stage2 and stage3
        self.deform_align = nn.LayerDict()
        self.backbone = nn.LayerDict()

        prop_names = [
            'stage2_backward', 'stage2_forward', 'stage3_backward',
            'stage3_forward'
        ]

        for i, layer in enumerate(prop_names):
            if i > 1 and self.use_refine_align:
                self.deform_align[layer] = ReAlignmentModule(
                    mid_channels,
                    mid_channels,
                    3,
                    padding=1,
                    deformable_groups=deform_groups)
            else:
                self.deform_align[layer] = AlignmentModule(
                    mid_channels,
                    mid_channels,
                    3,
                    padding=1,
                    deformable_groups=deform_groups)

            self.backbone[layer] = ResidualBlocksWithInputConv(
                (3 + i) * mid_channels, mid_channels, num_blocks)

        # stage1
        self.stage1_align = AlignmentModule(mid_channels,
                                            mid_channels,
                                            3,
                                            padding=1,
                                            deformable_groups=stage1_groups)
        self.stage1_blocks = ResidualBlocksWithInputConv(
            3 * mid_channels, mid_channels, 3)

        # upsampling module
        self.reconstruction = ResidualBlocksWithInputConv(
            6 * mid_channels, mid_channels, num_reconstruction_blocks)

        self.upsample1 = PixelShufflePack(mid_channels,
                                          mid_channels,
                                          2,
                                          upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels,
                                          mid_channels,
                                          2,
                                          upsample_kernel=3)
        if self.only_last:
            self.conv_last = nn.Conv2D(mid_channels, 3, 3, 1, 1)
        else:
            self.conv_hr = nn.Conv2D(mid_channels, mid_channels, 3, 1, 1)
            self.conv_last = nn.Conv2D(mid_channels, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4,
                                        mode='bilinear',
                                        align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

        # auxiliary loss
        if self.auxiliary_loss:
            self.aux_fusion = nn.Conv2D(mid_channels * 2, mid_channels, 3, 1, 1)

            self.aux_reconstruction = ResidualBlocksWithInputConv(
                4 * mid_channels, mid_channels, aux_reconstruction_blocks)

            self.aux_block_down1 = nn.Sequential(
                nn.Conv2D(3 + mid_channels, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2D(mid_channels, mid_channels, 3, 1, 1))
            self.aux_block_down2 = nn.Sequential(
                nn.Conv2D(mid_channels * 2, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2D(mid_channels, mid_channels, 3, 1, 1))

            self.aux_conv_last = nn.Conv2D(mid_channels, 3, 3, 1, 1)

        self.aux_upsample1 = PixelShufflePack(mid_channels,
                                              mid_channels,
                                              2,
                                              upsample_kernel=3)
        self.aux_upsample2 = PixelShufflePack(mid_channels,
                                              mid_channels,
                                              2,
                                              upsample_kernel=3)
        self.hybrid_conv_last = nn.Conv2D(mid_channels, 3, 3, 1, 1)

    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.
        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.
        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)

        Returns:
            Bool: Whether the input is a mirror-extended sequence.
        """

        with paddle.no_grad():
            self.is_mirror_extended = False
            if lrs.shape[1] % 2 == 0:
                lrs_1, lrs_2 = paddle.chunk(lrs, 2, axis=1)
                lrs_2 = paddle.flip(lrs_2, [1])
                if paddle.norm(lrs_1 - lrs_2) == 0:
                    self.is_mirror_extended = True

    def compute_flow(self, lrs):
        """Compute optical flow using pretrained flow network for feature alignment.
        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)

        Returns:
            Tuple: Tensor of forward optical flow and backward optical flow with shape (n, t-1, 2, h, w).
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

    def stage1(self, feats, flows, flows_forward=None):
        """Stage1 of PP-MSVSR network.
        Args:
            feats (dict): Dict with key 'spatial', the value is Array of tensor after feature extraction with shape (n, c, h, w).
            flows (tensor): Backward optical flow with shape (n, t-1, 2, h, w).
            flows_forward (tensor): Forward optical flow with shape (n, t-1, 2, h, w).

        Returns:
            Dict: The input dict with new keys 'feat_stage1', the value of 'feat_stage1' is Array of tensor after Local Fusion Module with shape (n, c, h, w).
        """

        n, t, _, h, w = flows.shape

        frame_idx = range(t, -1, -1)
        flow_idx = range(t, -1, -1)
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]

        # Local Fusion Module
        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]

            # get aligned right adjacent frames
            if i > 0:
                feat_prop = feats['spatial'][mapping_idx[idx + 1]]
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                cond_n1 = flow_warp(feat_prop, flow_n1.transpose([0, 2, 3, 1]))
                cond = paddle.concat([cond_n1, feat_current], axis=1)
                feat_prop, _, _ = self.stage1_align(feat_prop, cond, flow_n1)
            else:
                feat_prop = paddle.zeros([n, self.mid_channels, h, w])

            # get aligned left adjacent frames
            if i < t:
                feat_back = feats['spatial'][mapping_idx[idx - 1]]
                flow_n1_ = flows_forward[:, flow_idx[i] - 1, :, :, :]
                cond_n1_ = flow_warp(feat_back, flow_n1_.transpose([0, 2, 3,
                                                                    1]))
                cond_ = paddle.concat([cond_n1_, feat_current], axis=1)
                feat_back, _, _ = self.stage1_align(feat_back, cond_, flow_n1_)
            else:
                feat_back = paddle.zeros([n, self.mid_channels, h, w])

            # concatenate and residual blocks
            feat = [feat_current] + [feat_prop] + [feat_back]
            feat = paddle.concat(feat, axis=1)
            feat = self.stage1_blocks(feat)

            feats['feat_stage1'].append(feat)

        feats['feat_stage1'] = feats['feat_stage1'][::-1]

        return feats

    def stage2(self, feats, flows):
        """Stage2 of PP-MSVSR network.
        Args:
            feats (dict): Dict with key 'spatial' and 'feat_stage1' after stage1.
            flows (tuple): Tensor of backward optical flow and forward optical flow with shape (n, t-1, 2, h, w).

        Returns:
            feats (dict): The input dict with new keys 'stage2_backward' and 'stage2_forward', the value of both is Array of feature after stage2 with shape (n, c, h, w).
            pre_offset (dict): Dict with keys 'stage2_backward' and 'stage2_forward', the value of both is Array of offset in stage2 with shape (n, 18*deform_groups, h, w).
            pre_mask (dict): Dict with keys 'stage2_backward' and 'stage2_forward', the value of both is Array of mask in stage2 with shape (n, 9*deform_groups, h, w).
        """
        flows_backward, flows_forward = flows
        n, t, _, h, w = flows_backward.shape

        pre_offset = {}
        pre_mask = {}

        # propagation branches module
        prop_names = ['stage2_backward', 'stage2_forward']
        for index in range(2):
            prop_name = prop_names[index]
            pre_offset[prop_name] = [0 for _ in range(t)]
            pre_mask[prop_name] = [0 for _ in range(t)]
            feats[prop_name] = []
            frame_idx = range(0, t + 1)
            flow_idx = range(-1, t)
            mapping_idx = list(range(0, len(feats['spatial'])))
            mapping_idx += mapping_idx[::-1]

            if 'backward' in prop_name:
                frame_idx = frame_idx[::-1]
                flow_idx = frame_idx
                flows = flows_backward
            else:
                flows = flows_forward

            feat_prop = paddle.zeros([n, self.mid_channels, h, w])
            for i, idx in enumerate(frame_idx):
                feat_current = feats['spatial'][mapping_idx[idx]]

                if i > 0:
                    flow_n1 = flows[:, flow_idx[i], :, :, :]

                    cond_n1 = flow_warp(feat_prop,
                                        flow_n1.transpose([0, 2, 3, 1]))
                    cond = paddle.concat([cond_n1, feat_current], axis=1)

                    feat_prop, offset, mask = self.deform_align[prop_name](
                        feat_prop, cond, flow_n1)
                    pre_offset[prop_name][flow_idx[i]] = offset
                    pre_mask[prop_name][flow_idx[i]] = (mask)

                # concatenate and residual blocks
                feat = [feat_current] + [
                    feats[k][idx]
                    for k in feats if k not in ['spatial', prop_name]
                ] + [feat_prop]

                feat = paddle.concat(feat, axis=1)
                feat_prop = feat_prop + self.backbone[prop_name](feat)

                feats[prop_name].append(feat_prop)

            if 'backward' in prop_name:
                feats[prop_name] = feats[prop_name][::-1]

        return feats, pre_offset, pre_mask

    def stage3(self,
               feats,
               flows,
               aux_feats=None,
               pre_offset=None,
               pre_mask=None):
        """Stage3 of PP-MSVSR network.
        Args:
            feats (dict): Dict of features after stage2.
            flows (tuple): Tensor of backward optical flow and forward optical flow with shape (n, t-1, 2, h, w).
            aux_feats (dict): Dict with keys 'outs' and 'feats', the value is Array of tensor after auxiliary_stage with shape (n, 3, 4*h, 4*w) and (n, c, h, w), separately.
            pre_offset (dict): Dict with keys 'stage2_backward' and 'stage2_forward', the value of both is Array of offset in stage2 with shape (n, 18*deform_groups, h, w).
            pre_mask (dict): Dict with keys 'stage2_backward' and 'stage2_forward', the value of both is Array of mask in stage2 with shape (n, 9*deform_groups, h, w).

        Returns:
            feats (dict): The input feats dict with new keys 'stage3_backward' and 'stage3_forward', the value of both is Array of feature after stage3 with shape (n, c, h, w).
            """
        flows_backward, flows_forward = flows
        n, t, _, h, w = flows_backward.shape

        # propagation branches module
        prop_names = ['stage3_backward', 'stage3_forward']
        for index in range(2):
            prop_name = prop_names[index]
            feats[prop_name] = []
            frame_idx = range(0, t + 1)
            flow_idx = range(-1, t)
            mapping_idx = list(range(0, len(feats['spatial'])))
            mapping_idx += mapping_idx[::-1]

            if 'backward' in prop_name:
                frame_idx = frame_idx[::-1]
                flow_idx = frame_idx
                flows = flows_backward
                pre_stage_name = 'stage2_backward'
            else:
                flows = flows_forward
                pre_stage_name = 'stage2_forward'

            feat_prop = paddle.zeros([n, self.mid_channels, h, w])
            for i, idx in enumerate(frame_idx):
                feat_current = feats['spatial'][mapping_idx[idx]]
                if aux_feats is not None and 'feats' in aux_feats:
                    feat_current = aux_feats['feats'][mapping_idx[idx]]

                if i > 0:
                    flow_n1 = flows[:, flow_idx[i], :, :, :]

                    cond_n1 = flow_warp(feat_prop,
                                        flow_n1.transpose([0, 2, 3, 1]))
                    cond = paddle.concat([cond_n1, feat_current], axis=1)

                    feat_prop = self.deform_align[prop_name](
                        feat_prop, cond, flow_n1, feat_current,
                        pre_offset[pre_stage_name][flow_idx[i]],
                        pre_mask[pre_stage_name][flow_idx[i]])

                # concatenate and residual blocks
                feat = [feat_current] + [
                    feats[k][idx]
                    for k in feats if k not in ['spatial', prop_name]
                ] + [feat_prop]

                feat = paddle.concat(feat, axis=1)
                feat_prop = feat_prop + self.backbone[prop_name](feat)

                feats[prop_name].append(feat_prop)

            if 'backward' in prop_name:
                feats[prop_name] = feats[prop_name][::-1]

        return feats

    def auxiliary_stage(self, feats, lqs):
        """Compute the output image and auxiliary feature for Auxiliary Loss in stage2.
        Args:
            feats (dict): Dict of features after stage2.
            lqs (tensor): Input LR images with shape (n, t, c, h, w)

        Returns:
            dict: Dict with keys 'outs' and 'feats', the value is Array of tensor after auxiliary_stage with shape (n, 3, 4*h, 4*w) and (n, c, h, w), separately.
        """
        aux_feats = {}
        aux_feats['outs'] = []
        aux_feats['feats'] = []
        num_outputs = len(feats['spatial'])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        t = lqs.shape[1]
        for i in range(0, t):
            hr = [feats[k][i] for k in feats if (k != 'spatial')]
            feat_current = feats['spatial'][mapping_idx[i]]
            hr.insert(0, feat_current)
            hr = paddle.concat(hr, axis=1)

            hr_low = self.aux_reconstruction(hr)
            hr_mid = self.lrelu(self.aux_upsample1(hr_low))
            hr_high = self.lrelu(self.aux_upsample2(hr_mid))

            hr = self.aux_conv_last(hr_high)
            hr += self.img_upsample(lqs[:, i, :, :, :])

            # output tensor of auxiliary_stage with shape (n, 3, 4*h, 4*w)
            aux_feats['outs'].append(hr)

            aux_feat = self.aux_block_down1(paddle.concat([hr, hr_high],
                                                          axis=1))
            aux_feat = self.aux_block_down2(
                paddle.concat([aux_feat, hr_mid], axis=1))
            aux_feat = self.aux_fusion(paddle.concat([aux_feat, hr_low],
                                                     axis=1))

            # out feature of auxiliary_stage with shape (n, c, h, w)
            aux_feats['feats'].append(aux_feat)

        return aux_feats

    def upsample(self, lqs, feats, aux_feats=None):
        """Compute the output image given the features.
        Args:
            lqs (tensor): Input LR images with shape (n, t, c, h, w).
            feats (dict): Dict of features after stage3.
            aux_feats (dict): Dict with keys 'outs' and 'feats', the value is Array of tensor after auxiliary_stage with shape (n, 3, 4*h, 4*w) and (n, c, h, w), separately.

        Returns:
            Tensor: Output HR sequence with shape (n, t, 3, 4*h, 4*w).
        """

        outputs = []
        num_outputs = len(feats['spatial'])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        t = lqs.shape[1]
        for i in range(0, t):
            hr = [
                feats[k].pop(0) for k in feats
                if (k != 'spatial' and k != 'feat_stage1')
            ]
            if 'feat_stage1' in feats:
                local_feat = feats['feat_stage1'].pop(0)
                hr.insert(0, local_feat)
            hr.insert(0, feats['spatial'][mapping_idx[i]])
            hr = paddle.concat(hr, axis=1)

            hr = self.reconstruction(hr)

            hr = self.lrelu(self.upsample1(hr))
            hr = self.lrelu(self.upsample2(hr))
            if self.only_last:
                hr = self.conv_last(hr)
            else:
                hr = self.lrelu(self.conv_hr(hr))
                hr = self.conv_last(hr)

            hr += self.img_upsample(lqs[:, i, :, :, :])
            if self.use_local_connnect:
                local_head = self.lrelu(self.aux_upsample1(local_feat))
                local_head = self.lrelu(self.aux_upsample2(local_head))
                hr = self.hybrid_conv_last(local_head) + hr

            outputs.append(hr)

        if self.auxiliary_loss:
            return paddle.stack(aux_feats['outs'],
                                axis=1), paddle.stack(outputs, axis=1)
        return paddle.stack(outputs, axis=1)

    def forward(self, lqs):
        """Forward function for PP-MSVSR.
        Args:
            lqs (Tensor): Input LR sequence with shape (n, t, c, h, w).
        Returns:
            Tensor: Output HR sequence with shape (n, t, 3, 4*h, 4*w).
        """

        n, t, c, h, w = lqs.shape

        lqs_downsample = lqs

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
        feats['feat_stage1'] = []
        feats = self.stage1(feats, flows_backward, flows_forward)

        feats, pre_offset, pre_mask = self.stage2(
            feats, (flows_backward, flows_forward))

        if self.auxiliary_loss:
            aux_feats = self.auxiliary_stage(feats, lqs)

        feats = self.stage3(feats, (flows_backward, flows_forward), aux_feats,
                            pre_offset, pre_mask)

        return self.upsample(lqs, feats, aux_feats=aux_feats)


class AlignmentModule(nn.Layer):
    """deformable alignment module.
    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        deformable_groups (int): Number of deformable_groups in DeformConv2D.
    """
    def __init__(self,
                 in_channels=128,
                 out_channels=64,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 deformable_groups=16):
        super(AlignmentModule, self).__init__()

        self.conv_offset = nn.Sequential(
            nn.Conv2D(2 * out_channels + 2, out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(out_channels, 27 * deformable_groups, 3, 1, 1),
        )
        self.dcn = DeformConv2D(in_channels,
                                out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                deformable_groups=deformable_groups)

        self.init_offset()

    def init_offset(self):
        constant_(self.conv_offset[-1].weight, 0)
        constant_(self.conv_offset[-1].bias, 0)

    def forward(self, x, extra_feat, flow_1):
        extra_feat = paddle.concat([extra_feat, flow_1], axis=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = paddle.chunk(out, 3, axis=1)

        # offset
        offset = 10 * paddle.tanh(paddle.concat((o1, o2), axis=1))
        offset = offset + flow_1.flip(1).tile([1, offset.shape[1] // 2, 1, 1])

        # mask
        mask = F.sigmoid(mask)
        out = self.dcn(x, offset, mask)
        return out, offset, mask


class ReAlignmentModule(nn.Layer):
    """refine deformable alignment module.
    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        deformable_groups (int): Number of deformable_groups in DeformConv2D.
    """
    def __init__(self,
                 in_channels=128,
                 out_channels=64,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 deformable_groups=16):
        super(ReAlignmentModule, self).__init__()

        self.mdconv = DeformConv2D(in_channels,
                                   out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   dilation=dilation,
                                   deformable_groups=deformable_groups)
        self.conv_offset = nn.Sequential(
            nn.Conv2D(2 * out_channels + 2, out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2D(out_channels, 27 * deformable_groups, 3, 1, 1),
        )
        self.dcn = DeformConv2D(in_channels,
                                out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                deformable_groups=deformable_groups)

        self.init_offset()

    def init_offset(self):
        constant_(self.conv_offset[-1].weight, 0)
        constant_(self.conv_offset[-1].bias, 0)

    def forward(self,
                x,
                extra_feat,
                flow_1,
                feat_current,
                pre_stage_flow=None,
                pre_stage_mask=None):
        if pre_stage_flow is not None:
            pre_feat = self.mdconv(x, pre_stage_flow, pre_stage_mask)
            extra_feat = paddle.concat([pre_feat, feat_current, flow_1], axis=1)
        else:
            extra_feat = paddle.concat([extra_feat, flow_1], axis=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = paddle.chunk(out, 3, axis=1)

        # offset
        offset = 10 * paddle.tanh(paddle.concat((o1, o2), axis=1))
        if pre_stage_flow is not None:
            offset = offset + pre_stage_flow
        else:
            offset = offset + flow_1.flip(1).tile(
                [1, offset.shape[1] // 2, 1, 1])

        # mask
        if pre_stage_mask is not None:
            mask = (F.sigmoid(mask) + pre_stage_mask) / 2.0
        else:
            mask = F.sigmoid(mask)
        out = self.dcn(x, offset, mask)
        return out


class ModifiedSPyNet(nn.Layer):
    """Modified SPyNet network structure.

    The difference to the SPyNet in paper is that
        1. convolution with kernel_size=7 is replaced by convolution with kernel_size=3 in this version,
        2. less SPyNetBasicModule is used in this version,
        3. no BN is used in this version.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017

    Args:
        act_cfg (dict): Activation function.
            Default: dict(name='LeakyReLU').
        num_blocks (int): Number of SPyNetBlock.
            Default: 6.
        use_tiny_block (bool): Whether use tiny spynet.
            Default: True.
    """
    def __init__(self,
                 act_cfg=dict(name='LeakyReLU'),
                 num_blocks=6,
                 use_tiny_block=False):
        super().__init__()
        self.num_blocks = num_blocks
        self.basic_module = nn.LayerList([
            SPyNetBlock(act_cfg=act_cfg, use_tiny_block=use_tiny_block)
            for _ in range(num_blocks)
        ])

        self.register_buffer(
            'mean',
            paddle.to_tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]))
        self.register_buffer(
            'std',
            paddle.to_tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1]))

    def compute_flow(self, ref, supp):
        """Compute flow from ref to supp.

        Note that in this function, the images are already resized to a
        multiple of 32.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """
        n, _, h, w = ref.shape

        # normalize the input images
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]

        # generate downsampled frames
        for level in range(self.num_blocks - 1):
            ref.append(F.avg_pool2d(ref[-1], kernel_size=2, stride=2))
            supp.append(F.avg_pool2d(supp[-1], kernel_size=2, stride=2))
        ref = ref[::-1]
        supp = supp[::-1]

        # flow computation
        flow = paddle.to_tensor(
            np.zeros([
                n, 2, h // (2**(self.num_blocks - 1)), w //
                (2**(self.num_blocks - 1))
            ], 'float32'))

        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(
                    flow, scale_factor=2, mode='bilinear',
                    align_corners=True) * 2.0

            # add the residue to the upsampled flow
            flow = flow_up + self.basic_module[level](paddle.concat([
                ref[level],
                flow_warp(supp[level],
                          flow_up.transpose([0, 2, 3, 1]),
                          padding_mode='border'), flow_up
            ],
                                                                    axis=1))

        return flow

    def compute_flow_list(self, ref, supp):
        n, _, h, w = ref.shape

        # normalize the input images
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]

        # generate downsampled frames
        for level in range(self.num_blocks - 1):
            ref.append(F.avg_pool2d(ref[-1], kernel_size=2, stride=2))
            supp.append(F.avg_pool2d(supp[-1], kernel_size=2, stride=2))
        ref = ref[::-1]
        supp = supp[::-1]

        # flow computation
        flow_list = []
        flow = paddle.to_tensor(
            np.zeros([
                n, 2, h // (2**(self.num_blocks - 1)), w //
                (2**(self.num_blocks - 1))
            ], 'float32'))
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(
                    flow, scale_factor=2, mode='bilinear',
                    align_corners=True) * 2.0

            # add the residue to the upsampled flow
            flow = flow_up + self.basic_module[level](paddle.concat([
                ref[level],
                flow_warp(supp[level],
                          flow_up.transpose([0, 2, 3, 1]),
                          padding_mode='border'), flow_up
            ],
                                                                    axis=1))
            flow_list.append(flow)
        return flow_list

    def forward(self, ref, supp):
        """Forward function of Modified SPyNet.

        This function computes the optical flow from ref to supp.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """

        # upsize to a multiple of 32
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(ref,
                            size=(h_up, w_up),
                            mode='bilinear',
                            align_corners=False)

        supp = F.interpolate(supp,
                             size=(h_up, w_up),
                             mode='bilinear',
                             align_corners=False)

        ref.stop_gradient = False
        supp.stop_gradient = False

        # compute flow, and resize back to the original resolution
        flow = F.interpolate(self.compute_flow(ref, supp),
                             size=(h, w),
                             mode='bilinear',
                             align_corners=False)

        # adjust the flow values
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)

        return flow


class SPyNetBlock(nn.Layer):
    """Basic Block of Modified SPyNet.
    refer to Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    """
    def __init__(self, act_cfg=dict(name='LeakyReLU'), use_tiny_block=False):
        super().__init__()
        if use_tiny_block:
            self.basic_module = nn.Sequential(
                ConvLayer(in_channels=8,
                          out_channels=16,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          act_cfg=act_cfg),
                ConvLayer(in_channels=16,
                          out_channels=16,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          act_cfg=act_cfg),
                ConvLayer(in_channels=16,
                          out_channels=32,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          act_cfg=act_cfg),
                ConvLayer(in_channels=32,
                          out_channels=32,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          act_cfg=act_cfg),
                ConvLayer(in_channels=32,
                          out_channels=32,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          act_cfg=act_cfg),
                ConvLayer(in_channels=32,
                          out_channels=32,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          act_cfg=act_cfg),
                ConvLayer(in_channels=32,
                          out_channels=16,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          act_cfg=act_cfg),
                ConvLayer(in_channels=16,
                          out_channels=16,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          act_cfg=act_cfg),
                ConvLayer(in_channels=16,
                          out_channels=16,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          act_cfg=act_cfg),
                ConvLayer(in_channels=16,
                          out_channels=8,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          act_cfg=act_cfg),
                ConvLayer(in_channels=8,
                          out_channels=8,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          act_cfg=act_cfg),
                ConvLayer(in_channels=8,
                          out_channels=2,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          act_cfg=None))
        else:
            self.basic_module = nn.Sequential(
                ConvLayer(in_channels=8,
                          out_channels=16,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          act_cfg=act_cfg),
                ConvLayer(in_channels=16,
                          out_channels=16,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          act_cfg=act_cfg),
                ConvLayer(in_channels=16,
                          out_channels=32,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          act_cfg=act_cfg),
                ConvLayer(in_channels=32,
                          out_channels=32,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          act_cfg=act_cfg),
                ConvLayer(in_channels=32,
                          out_channels=32,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          act_cfg=act_cfg),
                ConvLayer(in_channels=32,
                          out_channels=64,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          act_cfg=act_cfg),
                ConvLayer(in_channels=64,
                          out_channels=32,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          act_cfg=act_cfg),
                ConvLayer(in_channels=32,
                          out_channels=32,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          act_cfg=act_cfg),
                ConvLayer(in_channels=32,
                          out_channels=32,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          act_cfg=act_cfg),
                ConvLayer(in_channels=32,
                          out_channels=16,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          act_cfg=act_cfg),
                ConvLayer(in_channels=16,
                          out_channels=16,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          act_cfg=act_cfg),
                ConvLayer(in_channels=16,
                          out_channels=16,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          act_cfg=act_cfg),
                ConvLayer(in_channels=16,
                          out_channels=16,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          act_cfg=act_cfg),
                ConvLayer(in_channels=16,
                          out_channels=16,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          act_cfg=act_cfg),
                ConvLayer(in_channels=16,
                          out_channels=2,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          act_cfg=None))

    def forward(self, tensor_input):
        """Forward function of SPyNetBlock.
        Args:
            tensor_input (Tensor): Input tensor with shape (b, 8, h, w).
                8 channels contain:
                [reference image (3), neighbor image (3), initial flow (2)].

        Returns:
            Tensor: Refined flow with shape (b, 2, h, w)
        """
        return self.basic_module(tensor_input)


class ConvLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 act_cfg=dict(name='ReLU')):
        super(ConvLayer, self).__init__()
        self.act_cfg = act_cfg
        self.with_activation = act_cfg is not None

        self.conv = nn.Conv2D(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups)

        if self.with_activation:
            if act_cfg['name'] == 'ReLU':
                self.act = paddle.nn.ReLU()
            elif act_cfg['name'] == 'LeakyReLU':
                self.act = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, tensor_input):
        out = self.conv(tensor_input)
        if self.with_activation:
            out = self.act(out)
        return out
