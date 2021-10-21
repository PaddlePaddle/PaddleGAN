#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import numpy as np

import paddle
import paddle.nn as nn
from paddle.vision.ops import DeformConv2D

from ...modules.init import kaiming_normal_, constant_, constant_init

from .builder import GENERATORS


@paddle.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for m in module_list:
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


class ResidualBlockNoBN(nn.Layer):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        nf (int): Channel number of intermediate features.
            Default: 64.
    """
    def __init__(self, nf=64):
        super(ResidualBlockNoBN, self).__init__()
        self.nf = nf
        self.conv1 = nn.Conv2D(self.nf, self.nf, 3, 1, 1)
        self.conv2 = nn.Conv2D(self.nf, self.nf, 3, 1, 1)
        self.relu = nn.ReLU()
        default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out


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


class PredeblurResNetPyramid(nn.Layer):
    """Pre-dublur module.

    Args:
        in_nf (int): Channel number of input image. Default: 3.
        nf (int): Channel number of intermediate features. Default: 64.
        HR_in (bool): Whether the input has high resolution. Default: False.
    """
    def __init__(self, in_nf=3, nf=64, HR_in=False):
        super(PredeblurResNetPyramid, self).__init__()
        self.in_nf = in_nf
        self.nf = nf
        self.HR_in = True if HR_in else False
        self.Leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        if self.HR_in:
            self.conv_first_1 = nn.Conv2D(in_channels=self.in_nf,
                                          out_channels=self.nf,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1)
            self.conv_first_2 = nn.Conv2D(in_channels=self.nf,
                                          out_channels=self.nf,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1)
            self.conv_first_3 = nn.Conv2D(in_channels=self.nf,
                                          out_channels=self.nf,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1)
        else:
            self.conv_first = nn.Conv2D(in_channels=self.in_nf,
                                        out_channels=self.nf,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        self.RB_L1_1 = ResidualBlockNoBN(nf=self.nf)
        self.RB_L1_2 = ResidualBlockNoBN(nf=self.nf)
        self.RB_L1_3 = ResidualBlockNoBN(nf=self.nf)
        self.RB_L1_4 = ResidualBlockNoBN(nf=self.nf)
        self.RB_L1_5 = ResidualBlockNoBN(nf=self.nf)
        self.RB_L2_1 = ResidualBlockNoBN(nf=self.nf)
        self.RB_L2_2 = ResidualBlockNoBN(nf=self.nf)
        self.RB_L3_1 = ResidualBlockNoBN(nf=self.nf)
        self.deblur_L2_conv = nn.Conv2D(in_channels=self.nf,
                                        out_channels=self.nf,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1)
        self.deblur_L3_conv = nn.Conv2D(in_channels=self.nf,
                                        out_channels=self.nf,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1)
        self.upsample = nn.Upsample(scale_factor=2,
                                    mode="bilinear",
                                    align_corners=False,
                                    align_mode=0)

    def forward(self, x):
        if self.HR_in:
            L1_fea = self.Leaky_relu(self.conv_first_1(x))
            L1_fea = self.Leaky_relu(self.conv_first_2(L1_fea))
            L1_fea = self.Leaky_relu(self.conv_first_3(L1_fea))
        else:
            L1_fea = self.Leaky_relu(self.conv_first(x))
        L2_fea = self.deblur_L2_conv(L1_fea)
        L2_fea = self.Leaky_relu(L2_fea)
        L3_fea = self.deblur_L3_conv(L2_fea)
        L3_fea = self.Leaky_relu(L3_fea)
        L3_fea = self.RB_L3_1(L3_fea)
        L3_fea = self.upsample(L3_fea)
        L2_fea = self.RB_L2_1(L2_fea) + L3_fea
        L2_fea = self.RB_L2_2(L2_fea)
        L2_fea = self.upsample(L2_fea)
        L1_fea = self.RB_L1_1(L1_fea)
        L1_fea = self.RB_L1_2(L1_fea) + L2_fea
        out = self.RB_L1_3(L1_fea)
        out = self.RB_L1_4(out)
        out = self.RB_L1_5(out)
        return out


class TSAFusion(nn.Layer):
    """Temporal Spatial Attention (TSA) fusion module.

    Temporal: Calculate the correlation between center frame and
        neighboring frames;
    Spatial: It has 3 pyramid levels, the attention is similar to SFT.
        (SFT: Recovering realistic texture in image super-resolution by deep
            spatial feature transform.)

    Args:
        nf (int): Channel number of middle features. Default: 64.
        nframes (int): Number of frames. Default: 5.
        center (int): The index of center frame. Default: 2.
    """
    def __init__(self, nf=64, nframes=5, center=2):
        super(TSAFusion, self).__init__()
        self.nf = nf
        self.nframes = nframes
        self.center = center
        self.sigmoid = nn.Sigmoid()
        self.Leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.tAtt_2 = nn.Conv2D(in_channels=self.nf,
                                out_channels=self.nf,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.tAtt_1 = nn.Conv2D(in_channels=self.nf,
                                out_channels=self.nf,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.fea_fusion = nn.Conv2D(in_channels=self.nf * self.nframes,
                                    out_channels=self.nf,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)
        self.sAtt_1 = nn.Conv2D(in_channels=self.nf * self.nframes,
                                out_channels=self.nf,
                                kernel_size=1,
                                stride=1,
                                padding=0)
        self.max_pool = nn.MaxPool2D(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2D(3, stride=2, padding=1, exclusive=False)
        self.sAtt_2 = nn.Conv2D(in_channels=2 * self.nf,
                                out_channels=self.nf,
                                kernel_size=1,
                                stride=1,
                                padding=0)
        self.sAtt_3 = nn.Conv2D(in_channels=self.nf,
                                out_channels=self.nf,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.sAtt_4 = nn.Conv2D(
            in_channels=self.nf,
            out_channels=self.nf,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.sAtt_5 = nn.Conv2D(in_channels=self.nf,
                                out_channels=self.nf,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.sAtt_add_1 = nn.Conv2D(in_channels=self.nf,
                                    out_channels=self.nf,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)
        self.sAtt_add_2 = nn.Conv2D(in_channels=self.nf,
                                    out_channels=self.nf,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)
        self.sAtt_L1 = nn.Conv2D(in_channels=self.nf,
                                 out_channels=self.nf,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.sAtt_L2 = nn.Conv2D(
            in_channels=2 * self.nf,
            out_channels=self.nf,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.sAtt_L3 = nn.Conv2D(in_channels=self.nf,
                                 out_channels=self.nf,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.upsample = nn.Upsample(scale_factor=2,
                                    mode="bilinear",
                                    align_corners=False,
                                    align_mode=0)

    def forward(self, aligned_fea):
        """
        Args:
            aligned_feat (Tensor): Aligned features with shape (b, n, c, h, w).

        Returns:
            Tensor: Features after TSA with the shape (b, c, h, w).
        """
        B, N, C, H, W = aligned_fea.shape
        x_center = aligned_fea[:, self.center, :, :, :]
        emb_rf = self.tAtt_2(x_center)
        emb = aligned_fea.reshape([-1, C, H, W])
        emb = self.tAtt_1(emb)
        emb = emb.reshape([-1, N, self.nf, H, W])
        cor_l = []
        for i in range(N):
            emb_nbr = emb[:, i, :, :, :]  #[B,C,W,H]
            cor_tmp = paddle.sum(emb_nbr * emb_rf, axis=1)
            cor_tmp = paddle.unsqueeze(cor_tmp, axis=1)
            cor_l.append(cor_tmp)
        cor_prob = paddle.concat(cor_l, axis=1)  #[B,N,H,W]

        cor_prob = self.sigmoid(cor_prob)
        cor_prob = paddle.unsqueeze(cor_prob, axis=2)  #[B,N,1,H,W]
        cor_prob = paddle.expand(cor_prob, [B, N, self.nf, H, W])  #[B,N,C,H,W]
        cor_prob = cor_prob.reshape([B, -1, H, W])
        aligned_fea = aligned_fea.reshape([B, -1, H, W])
        aligned_fea = aligned_fea * cor_prob

        fea = self.fea_fusion(aligned_fea)
        fea = self.Leaky_relu(fea)

        #spatial fusion
        att = self.sAtt_1(aligned_fea)
        att = self.Leaky_relu(att)
        att_max = self.max_pool(att)
        att_avg = self.avg_pool(att)
        att_pool = paddle.concat([att_max, att_avg], axis=1)
        att = self.sAtt_2(att_pool)
        att = self.Leaky_relu(att)

        #pyramid
        att_L = self.sAtt_L1(att)
        att_L = self.Leaky_relu(att_L)
        att_max = self.max_pool(att_L)
        att_avg = self.avg_pool(att_L)
        att_pool = paddle.concat([att_max, att_avg], axis=1)
        att_L = self.sAtt_L2(att_pool)
        att_L = self.Leaky_relu(att_L)
        att_L = self.sAtt_L3(att_L)
        att_L = self.Leaky_relu(att_L)
        att_L = self.upsample(att_L)

        att = self.sAtt_3(att)
        att = self.Leaky_relu(att)
        att = att + att_L
        att = self.sAtt_4(att)
        att = self.Leaky_relu(att)
        att = self.upsample(att)
        att = self.sAtt_5(att)
        att_add = self.sAtt_add_1(att)
        att_add = self.Leaky_relu(att_add)
        att_add = self.sAtt_add_2(att_add)
        att = self.sigmoid(att)

        fea = fea * att * 2 + att_add
        return fea


class DCNPack(nn.Layer):
    """Modulated deformable conv for deformable alignment.

    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """
    def __init__(self,
                 num_filters=64,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 deformable_groups=8,
                 extra_offset_mask=True):
        super(DCNPack, self).__init__()
        self.extra_offset_mask = extra_offset_mask
        self.deformable_groups = deformable_groups
        self.num_filters = num_filters
        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size, kernel_size]
        self.conv_offset_mask = nn.Conv2D(in_channels=self.num_filters,
                                          out_channels=self.deformable_groups *
                                          3 * self.kernel_size[0] *
                                          self.kernel_size[1],
                                          kernel_size=self.kernel_size,
                                          stride=stride,
                                          padding=padding)
        self.total_channels = self.deformable_groups * 3 * self.kernel_size[
            0] * self.kernel_size[1]
        self.split_channels = self.total_channels // 3
        self.dcn = DeformConv2D(in_channels=self.num_filters,
                                out_channels=self.num_filters,
                                kernel_size=self.kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                deformable_groups=self.deformable_groups)
        self.sigmoid = nn.Sigmoid()
        # init conv offset
        constant_init(self.conv_offset_mask, 0., 0.)

    def forward(self, fea_and_offset):
        out = None
        x = None
        if self.extra_offset_mask:
            out = self.conv_offset_mask(fea_and_offset[1])
            x = fea_and_offset[0]
        o1 = out[:, 0:self.split_channels, :, :]
        o2 = out[:, self.split_channels:2 * self.split_channels, :, :]
        mask = out[:, 2 * self.split_channels:, :, :]
        offset = paddle.concat([o1, o2], axis=1)
        mask = self.sigmoid(mask)
        y = self.dcn(x, offset, mask)
        return y


class PCDAlign(nn.Layer):
    """Alignment module using Pyramid, Cascading and Deformable convolution
    (PCD). It is used in EDVR.

    Ref:
        EDVR: Video Restoration with Enhanced Deformable Convolutional Networks

    Args:
        nf (int): Channel number of middle features. Default: 64.
        groups (int): Deformable groups. Defaults: 8.
    """
    def __init__(self, nf=64, groups=8):
        super(PCDAlign, self).__init__()
        self.nf = nf
        self.groups = groups
        self.Leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.upsample = nn.Upsample(scale_factor=2,
                                    mode="bilinear",
                                    align_corners=False,
                                    align_mode=0)
        # Pyramid has three levels:
        # L3: level 3, 1/4 spatial size
        # L2: level 2, 1/2 spatial size
        # L1: level 1, original spatial size

        # L3
        self.PCD_Align_L3_offset_conv1 = nn.Conv2D(in_channels=nf * 2,
                                                   out_channels=nf,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1)
        self.PCD_Align_L3_offset_conv2 = nn.Conv2D(in_channels=nf,
                                                   out_channels=nf,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1)
        self.PCD_Align_L3_dcn = DCNPack(num_filters=nf,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        deformable_groups=groups)
        #L2
        self.PCD_Align_L2_offset_conv1 = nn.Conv2D(in_channels=nf * 2,
                                                   out_channels=nf,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1)
        self.PCD_Align_L2_offset_conv2 = nn.Conv2D(in_channels=nf * 2,
                                                   out_channels=nf,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1)
        self.PCD_Align_L2_offset_conv3 = nn.Conv2D(in_channels=nf,
                                                   out_channels=nf,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1)
        self.PCD_Align_L2_dcn = DCNPack(num_filters=nf,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        deformable_groups=groups)
        self.PCD_Align_L2_fea_conv = nn.Conv2D(in_channels=nf * 2,
                                               out_channels=nf,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1)
        #L1
        self.PCD_Align_L1_offset_conv1 = nn.Conv2D(in_channels=nf * 2,
                                                   out_channels=nf,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1)
        self.PCD_Align_L1_offset_conv2 = nn.Conv2D(in_channels=nf * 2,
                                                   out_channels=nf,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1)
        self.PCD_Align_L1_offset_conv3 = nn.Conv2D(in_channels=nf,
                                                   out_channels=nf,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1)
        self.PCD_Align_L1_dcn = DCNPack(num_filters=nf,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        deformable_groups=groups)
        self.PCD_Align_L1_fea_conv = nn.Conv2D(in_channels=nf * 2,
                                               out_channels=nf,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1)
        #cascade
        self.PCD_Align_cas_offset_conv1 = nn.Conv2D(in_channels=nf * 2,
                                                    out_channels=nf,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=1)
        self.PCD_Align_cas_offset_conv2 = nn.Conv2D(in_channels=nf,
                                                    out_channels=nf,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=1)
        self.PCD_Align_cascade_dcn = DCNPack(num_filters=nf,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1,
                                             deformable_groups=groups)

    def forward(self, nbr_fea_l, ref_fea_l):
        """Align neighboring frame features to the reference frame features.

        Args:
            nbr_fea_l (list[Tensor]): Neighboring feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).
            ref_fea_l (list[Tensor]): Reference feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).

        Returns:
            Tensor: Aligned features.
        """
        #L3
        L3_offset = paddle.concat([nbr_fea_l[2], ref_fea_l[2]], axis=1)
        L3_offset = self.PCD_Align_L3_offset_conv1(L3_offset)
        L3_offset = self.Leaky_relu(L3_offset)
        L3_offset = self.PCD_Align_L3_offset_conv2(L3_offset)
        L3_offset = self.Leaky_relu(L3_offset)

        L3_fea = self.PCD_Align_L3_dcn([nbr_fea_l[2], L3_offset])
        L3_fea = self.Leaky_relu(L3_fea)
        #L2
        L2_offset = paddle.concat([nbr_fea_l[1], ref_fea_l[1]], axis=1)
        L2_offset = self.PCD_Align_L2_offset_conv1(L2_offset)
        L2_offset = self.Leaky_relu(L2_offset)
        L3_offset = self.upsample(L3_offset)
        L2_offset = paddle.concat([L2_offset, L3_offset * 2], axis=1)
        L2_offset = self.PCD_Align_L2_offset_conv2(L2_offset)
        L2_offset = self.Leaky_relu(L2_offset)
        L2_offset = self.PCD_Align_L2_offset_conv3(L2_offset)
        L2_offset = self.Leaky_relu(L2_offset)
        L2_fea = self.PCD_Align_L2_dcn([nbr_fea_l[1], L2_offset])
        L3_fea = self.upsample(L3_fea)
        L2_fea = paddle.concat([L2_fea, L3_fea], axis=1)
        L2_fea = self.PCD_Align_L2_fea_conv(L2_fea)
        L2_fea = self.Leaky_relu(L2_fea)
        #L1
        L1_offset = paddle.concat([nbr_fea_l[0], ref_fea_l[0]], axis=1)
        L1_offset = self.PCD_Align_L1_offset_conv1(L1_offset)
        L1_offset = self.Leaky_relu(L1_offset)
        L2_offset = self.upsample(L2_offset)
        L1_offset = paddle.concat([L1_offset, L2_offset * 2], axis=1)
        L1_offset = self.PCD_Align_L1_offset_conv2(L1_offset)
        L1_offset = self.Leaky_relu(L1_offset)
        L1_offset = self.PCD_Align_L1_offset_conv3(L1_offset)
        L1_offset = self.Leaky_relu(L1_offset)
        L1_fea = self.PCD_Align_L1_dcn([nbr_fea_l[0], L1_offset])
        L2_fea = self.upsample(L2_fea)
        L1_fea = paddle.concat([L1_fea, L2_fea], axis=1)
        L1_fea = self.PCD_Align_L1_fea_conv(L1_fea)
        #cascade
        offset = paddle.concat([L1_fea, ref_fea_l[0]], axis=1)
        offset = self.PCD_Align_cas_offset_conv1(offset)
        offset = self.Leaky_relu(offset)
        offset = self.PCD_Align_cas_offset_conv2(offset)
        offset = self.Leaky_relu(offset)
        L1_fea = self.PCD_Align_cascade_dcn([L1_fea, offset])
        L1_fea = self.Leaky_relu(L1_fea)

        return L1_fea


@GENERATORS.register()
class EDVRNet(nn.Layer):
    """EDVR network structure for video super-resolution.

    Now only support X4 upsampling factor.
    Paper:
        EDVR: Video Restoration with Enhanced Deformable Convolutional Networks

    Args:
        in_nf (int): Channel number of input image. Default: 3.
        out_nf (int): Channel number of output image. Default: 3.
        scale_factor (int): Scale factor from input image to output image. Default: 4.
        nf (int): Channel number of intermediate features. Default: 64.
        nframes (int): Number of input frames. Default: 5.
        groups (int): Deformable groups. Defaults: 8.
        front_RBs (int): Number of blocks for feature extraction. Default: 5.
        back_RBs (int): Number of blocks for reconstruction. Default: 10.
        center (int): The index of center frame. Frame counting from 0. Default: None.
        predeblur (bool): Whether has predeblur module. Default: False.
        HR_in (bool): Whether the input has high resolution. Default: False.
        with_tsa (bool): Whether has TSA module. Default: True.
        TSA_only (bool): Whether only use TSA module. Default: False.
    """
    def __init__(self,
                 in_nf=3,
                 out_nf=3,
                 scale_factor=4,
                 nf=64,
                 nframes=5,
                 groups=8,
                 front_RBs=5,
                 back_RBs=10,
                 center=None,
                 predeblur=False,
                 HR_in=False,
                 w_TSA=True):
        super(EDVRNet, self).__init__()
        self.in_nf = in_nf
        self.out_nf = out_nf
        self.scale_factor = scale_factor
        self.nf = nf
        self.nframes = nframes
        self.groups = groups
        self.front_RBs = front_RBs
        self.back_RBs = back_RBs
        self.center = nframes // 2 if center is None else center
        self.predeblur = True if predeblur else False
        self.HR_in = True if HR_in else False
        self.w_TSA = True if w_TSA else False

        self.Leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        if self.predeblur:
            self.pre_deblur = PredeblurResNetPyramid(in_nf=self.in_nf,
                                                     nf=self.nf,
                                                     HR_in=self.HR_in)
            self.cov_1 = nn.Conv2D(in_channels=self.nf,
                                   out_channels=self.nf,
                                   kernel_size=1,
                                   stride=1)
        else:
            self.conv_first = nn.Conv2D(in_channels=self.in_nf,
                                        out_channels=self.nf,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

        #feature extraction module
        self.feature_extractor = MakeMultiBlocks(ResidualBlockNoBN,
                                                 self.front_RBs, self.nf)
        self.fea_L2_conv1 = nn.Conv2D(in_channels=self.nf,
                                      out_channels=self.nf,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1)
        self.fea_L2_conv2 = nn.Conv2D(in_channels=self.nf,
                                      out_channels=self.nf,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)
        self.fea_L3_conv1 = nn.Conv2D(
            in_channels=self.nf,
            out_channels=self.nf,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.fea_L3_conv2 = nn.Conv2D(in_channels=self.nf,
                                      out_channels=self.nf,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)

        #PCD alignment module
        self.PCDModule = PCDAlign(nf=self.nf, groups=self.groups)

        #TSA Fusion module
        if self.w_TSA:
            self.TSAModule = TSAFusion(nf=self.nf,
                                       nframes=self.nframes,
                                       center=self.center)
        else:
            self.TSAModule = nn.Conv2D(in_channels=self.nframes * self.nf,
                                       out_channels=self.nf,
                                       kernel_size=1,
                                       stride=1)

        #reconstruction module
        self.reconstructor = MakeMultiBlocks(ResidualBlockNoBN, self.back_RBs,
                                             self.nf)
        self.upconv1 = nn.Conv2D(in_channels=self.nf,
                                 out_channels=4 * self.nf,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)

        self.pixel_shuffle = nn.PixelShuffle(2)
        self.upconv2 = nn.Conv2D(in_channels=self.nf,
                                 out_channels=4 * 64,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.HRconv = nn.Conv2D(in_channels=64,
                                out_channels=64,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.conv_last = nn.Conv2D(in_channels=64,
                                   out_channels=self.out_nf,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        if self.scale_factor == 4:
            self.upsample = nn.Upsample(scale_factor=self.scale_factor,
                                        mode="bilinear",
                                        align_corners=False,
                                        align_mode=0)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input features with shape (b, n, c, h, w).

        Returns:
            Tensor: Features after EDVR with the shape (b, c, scale_factor*h, scale_factor*w).
        """
        B, N, C, H, W = x.shape
        x_center = x[:, self.center, :, :, :]
        L1_fea = x.reshape([-1, C, H, W])  #[B*N，C,W,H]
        if self.predeblur:
            L1_fea = self.pre_deblur(L1_fea)
            L1_fea = self.cov_1(L1_fea)
            if self.HR_in:
                H, W = H // 4, W // 4
        else:
            L1_fea = self.conv_first(L1_fea)
            L1_fea = self.Leaky_relu(L1_fea)

        # feature extraction and create Pyramid
        L1_fea = self.feature_extractor(L1_fea)
        # L2
        L2_fea = self.fea_L2_conv1(L1_fea)
        L2_fea = self.Leaky_relu(L2_fea)
        L2_fea = self.fea_L2_conv2(L2_fea)
        L2_fea = self.Leaky_relu(L2_fea)
        # L3
        L3_fea = self.fea_L3_conv1(L2_fea)
        L3_fea = self.Leaky_relu(L3_fea)
        L3_fea = self.fea_L3_conv2(L3_fea)
        L3_fea = self.Leaky_relu(L3_fea)

        L1_fea = L1_fea.reshape([-1, N, self.nf, H, W])
        L2_fea = L2_fea.reshape([-1, N, self.nf, H // 2, W // 2])
        L3_fea = L3_fea.reshape([-1, N, self.nf, H // 4, W // 4])

        # pcd align
        ref_fea_l = [
            L1_fea[:, self.center, :, :, :], L2_fea[:, self.center, :, :, :],
            L3_fea[:, self.center, :, :, :]
        ]
        aligned_fea = []
        for i in range(N):
            nbr_fea_l = [
                L1_fea[:, i, :, :, :], L2_fea[:, i, :, :, :], L3_fea[:,
                                                                     i, :, :, :]
            ]
            aligned_fea.append(self.PCDModule(nbr_fea_l, ref_fea_l))

        # TSA Fusion
        aligned_fea = paddle.stack(aligned_fea, axis=1)  # [B, N, C, H, W]
        fea = None
        if not self.w_TSA:
            aligned_fea = aligned_fea.reshape([B, -1, H, W])
        fea = self.TSAModule(aligned_fea)  # [B, N, C, H, W]

        #Reconstruct
        out = self.reconstructor(fea)

        out = self.upconv1(out)
        out = self.pixel_shuffle(out)
        out = self.Leaky_relu(out)
        out = self.upconv2(out)
        out = self.pixel_shuffle(out)
        out = self.Leaky_relu(out)

        out = self.HRconv(out)
        out = self.Leaky_relu(out)
        out = self.conv_last(out)

        if self.HR_in:
            base = x_center
        else:
            base = self.upsample(x_center)

        out += base
        return out
