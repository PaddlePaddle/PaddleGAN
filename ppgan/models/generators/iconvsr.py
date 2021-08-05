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

import paddle

import numpy as np
import scipy.io as scio

import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import initializer
from ...utils.download import get_path_from_url
from ...modules.init import kaiming_normal_, constant_

from .builder import GENERATORS
from .basicvsr import SPyNet
from .edvr import PCDAlign, TSAFusion


@paddle.no_grad()
def default_init_weights(layer_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        layer_list (list[nn.Layer] | nn.Layer): Layers to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(layer_list, list):
        layer_list = [layer_list]
    for m in layer_list:
        if isinstance(m, nn.Conv2D):
            kaiming_normal_(m.weight, **kwargs)
            scale_weight = scale * m.weight
            m.weight.set_value(scale_weight)
            if m.bias is not None:
                constant_(m.bias, bias_fill)
        elif isinstance(m, nn.Linear):
            kaiming_normal_(m.weight, **kwargs)
            scale_weight = scale * m.weight
            m.weight.set_value(scale_weight)
            if m.bias is not None:
                constant_(m.bias, bias_fill)
        elif isinstance(m, nn.BatchNorm):
            constant_(m.weight, 1)


class PixelShufflePack(nn.Layer):
    """ Pixel Shuffle upsample layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of Conv layer to expand channels.

    Returns:
        Upsampled feature map.
    """
    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2D(self.in_channels,
                                       self.out_channels * scale_factor *
                                       scale_factor,
                                       self.upsample_kernel,
                                       padding=(self.upsample_kernel - 1) // 2)
        self.pixel_shuffle = nn.PixelShuffle(self.scale_factor)
        self.init_weights()

    def init_weights(self):
        """Initialize weights for PixelShufflePack.
        """
        default_init_weights(self, 1)

    def forward(self, x):
        """Forward function for PixelShufflePack.

        Args:
            x (Tensor): Input tensor with shape (in_channels, c, h, w).

        Returns:
            Tensor with shape (out_channels, c, scale_factor*h, scale_factor*w).
        """
        x = self.upsample_conv(x)
        x = self.pixel_shuffle(x)
        return x


def MakeMultiBlocks(func, num_layers, nf=64):
    """Make layers by stacking the same blocks.

    Args:
        func (nn.Layer): nn.Layer class for basic block.
        num_layers (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    Blocks = nn.Sequential()
    for i in range(num_layers):
        Blocks.add_sublayer('block%d' % i, func(nf))
    return Blocks


class ResidualBlockNoBN(nn.Layer):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        nf (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.0.
    """
    def __init__(self, nf=64, res_scale=1.0):
        super(ResidualBlockNoBN, self).__init__()
        self.nf = nf
        self.res_scale = res_scale
        self.conv1 = nn.Conv2D(self.nf, self.nf, 3, 1, 1)
        self.conv2 = nn.Conv2D(self.nf, self.nf, 3, 1, 1)
        self.relu = nn.ReLU()
        if self.res_scale == 1.0:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor with shape (n, c, h, w).
        """
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or a feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    if x.shape[-2:] != flow.shape[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.shape[-2:]}) and '
                         f'flow ({flow.shape[1:3]}) are not the same.')
    _, _, h, w = x.shape
    # create mesh grid
    grid_y, grid_x = paddle.meshgrid(paddle.arange(0, h), paddle.arange(0, w))
    grid = paddle.stack((grid_x, grid_y), axis=2)  # (w, h, 2)
    grid = paddle.cast(grid, 'float32')
    grid.stop_gradient = True

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = paddle.stack((grid_flow_x, grid_flow_y), axis=3)
    output = F.grid_sample(x,
                           grid_flow,
                           mode=interpolation,
                           padding_mode=padding_mode,
                           align_corners=align_corners)
    return output


class ResidualBlocksWithInputConv(nn.Layer):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """
    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        # a convolution used to match the channels of the residual blocks
        self.covn1 = nn.Conv2D(in_channels, out_channels, 3, 1, 1)
        self.Leaky_relu = nn.LeakyReLU(negative_slope=0.1)

        # residual blocks
        self.ResidualBlocks = MakeMultiBlocks(ResidualBlockNoBN,
                                              num_blocks,
                                              nf=out_channels)

    def forward(self, feat):
        """
        Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        out = self.Leaky_relu(self.covn1(feat))
        out = self.ResidualBlocks(out)
        return out


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
            ]  # padding
        elif self.padding == 3:
            lrs = [lrs[:, [6, 5, 4]], lrs, lrs[:, [-5, -6, -7]]]  # padding
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
            if i < t - 1:  # no warping required for the last timestep
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


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)


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
        pretrained (str): The pretrained model path. Default: None.
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
        # act_cfg = dict(type='LeakyReLU', negative_slope=0.1)

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
            self.fusion = nn.Conv2d(num_frames * mid_channels, mid_channels, 1,
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
