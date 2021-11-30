# Copyright (c) MMEditing Authors.

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.ops import DeformConv2D
from ...utils.download import get_path_from_url
from ...modules.init import kaiming_normal_, constant_
from .builder import GENERATORS


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
    x_h, x_w = x.shape[-2:]
    flow_h, flow_w = flow.shape[1:3]
    if x_h != flow_h or x_w != flow_w:
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


class SPyNetBasicModule(nn.Layer):
    """Basic Module for SPyNet.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    """
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2D(in_channels=8,
                               out_channels=32,
                               kernel_size=7,
                               stride=1,
                               padding=3)
        self.conv2 = nn.Conv2D(in_channels=32,
                               out_channels=64,
                               kernel_size=7,
                               stride=1,
                               padding=3)
        self.conv3 = nn.Conv2D(in_channels=64,
                               out_channels=32,
                               kernel_size=7,
                               stride=1,
                               padding=3)
        self.conv4 = nn.Conv2D(in_channels=32,
                               out_channels=16,
                               kernel_size=7,
                               stride=1,
                               padding=3)
        self.conv5 = nn.Conv2D(in_channels=16,
                               out_channels=2,
                               kernel_size=7,
                               stride=1,
                               padding=3)
        self.relu = nn.ReLU()

    def forward(self, tensor_input):
        """
        Args:
            tensor_input (Tensor): Input tensor with shape (b, 8, h, w).
                8 channels contain:
                [reference image (3), neighbor image (3), initial flow (2)].

        Returns:
            Tensor: Refined flow with shape (b, 2, h, w)
        """
        out = self.relu(self.conv1(tensor_input))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        out = self.conv5(out)
        return out


class SPyNet(nn.Layer):
    """SPyNet network structure.

    The difference to the SPyNet in paper is that
        1. more SPyNetBasicModule is used in this version, and
        2. no batch normalization is used in this version.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017

    """
    def __init__(self):
        super().__init__()

        self.basic_module0 = SPyNetBasicModule()
        self.basic_module1 = SPyNetBasicModule()
        self.basic_module2 = SPyNetBasicModule()
        self.basic_module3 = SPyNetBasicModule()
        self.basic_module4 = SPyNetBasicModule()
        self.basic_module5 = SPyNetBasicModule()

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
        for level in range(5):
            ref.append(F.avg_pool2d(ref[-1], kernel_size=2, stride=2))
            supp.append(F.avg_pool2d(supp[-1], kernel_size=2, stride=2))
        ref = ref[::-1]
        supp = supp[::-1]

        # flow computation
        flow = paddle.zeros([n, 2, h // 32, w // 32])

        # level=0
        flow_up = flow
        flow = flow_up + self.basic_module0(
            paddle.concat([
                ref[0],
                flow_warp(supp[0],
                          flow_up.transpose([0, 2, 3, 1]),
                          padding_mode='border'), flow_up
            ], 1))

        # level=1
        flow_up = F.interpolate(
            flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
        flow = flow_up + self.basic_module1(
            paddle.concat([
                ref[1],
                flow_warp(supp[1],
                          flow_up.transpose([0, 2, 3, 1]),
                          padding_mode='border'), flow_up
            ], 1))

        # level=2
        flow_up = F.interpolate(
            flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
        flow = flow_up + self.basic_module2(
            paddle.concat([
                ref[2],
                flow_warp(supp[2],
                          flow_up.transpose([0, 2, 3, 1]),
                          padding_mode='border'), flow_up
            ], 1))

        # level=3
        flow_up = F.interpolate(
            flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
        flow = flow_up + self.basic_module3(
            paddle.concat([
                ref[3],
                flow_warp(supp[3],
                          flow_up.transpose([0, 2, 3, 1]),
                          padding_mode='border'), flow_up
            ], 1))

        # level=4
        flow_up = F.interpolate(
            flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
        flow = flow_up + self.basic_module4(
            paddle.concat([
                ref[4],
                flow_warp(supp[4],
                          flow_up.transpose([0, 2, 3, 1]),
                          padding_mode='border'), flow_up
            ], 1))

        # level=5
        flow_up = F.interpolate(
            flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
        flow = flow_up + self.basic_module5(
            paddle.concat([
                ref[5],
                flow_warp(supp[5],
                          flow_up.transpose([0, 2, 3, 1]),
                          padding_mode='border'), flow_up
            ], 1))

        return flow

    def forward(self, ref, supp):
        """Forward function of SPyNet.

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
        flow_up = self.compute_flow(ref, supp)
        flow = F.interpolate(flow_up,
                             size=(h, w),
                             mode='bilinear',
                             align_corners=False)

        # adjust the flow values
        # todo: grad bug
        # flow[:, 0, :, :] *= (float(w) / float(w_up))
        # flow[:, 1, :, :] *= (float(h) / float(h_up))

        flow_x = flow[:, 0:1, :, :] * (float(w) / float(w_up))
        flow_y = flow[:, 1:2, :, :] * (float(h) / float(h_up))
        flow = paddle.concat([flow_x, flow_y], 1)

        return flow


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
class BasicVSRNet(nn.Layer):
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
    def __init__(self, mid_channels=64, num_blocks=30):

        super().__init__()

        self.mid_channels = mid_channels

        # optical flow network for feature alignment
        self.spynet = SPyNet()
        weight_path = get_path_from_url(
            'https://paddlegan.bj.bcebos.com/models/spynet.pdparams')
        self.spynet.set_state_dict(paddle.load(weight_path))

        # propagation branches
        self.backward_resblocks = ResidualBlocksWithInputConv(
            mid_channels + 3, mid_channels, num_blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(
            mid_channels + 3, mid_channels, num_blocks)

        # upsample
        self.fusion = nn.Conv2D(mid_channels * 2, mid_channels, 1, 1, 0)
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
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        """

        self.is_mirror_extended = False
        if lrs.shape[1] % 2 == 0:
            lrs_1, lrs_2 = paddle.chunk(lrs, 2, axis=1)
            lrs_2 = paddle.flip(lrs_2, [1])
            if paddle.norm(lrs_1 - lrs_2) == 0:
                self.is_mirror_extended = True

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

        n, t, c, h, w = lrs.shape
        t = paddle.to_tensor(t)
        assert h >= 64 and w >= 64, (
            'The height and width of inputs should be at least 64, '
            f'but got {h} and {w}.')

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lrs)

        # compute optical flow
        flows_forward, flows_backward = self.compute_flow(lrs)

        # backward-time propgation
        outputs = []
        feat_prop = paddle.zeros([n, self.mid_channels, h, w])
        for i in range(t - 1, -1, -1):
            if i < t - 1:  # no warping required for the last timestep
                flow1 = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow1.transpose([0, 2, 3, 1]))

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

            feat_prop = paddle.concat([lr_curr, feat_prop], axis=1)
            feat_prop = self.forward_resblocks(feat_prop)

            # upsampling given the backward and forward features
            out = paddle.concat([outputs[i], feat_prop], axis=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.upsample1(out))
            out = self.lrelu(self.upsample2(out))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = self.img_upsample(lr_curr)
            out += base
            outputs[i] = out

        return paddle.stack(outputs, axis=1)


class SecondOrderDeformableAlignment(nn.Layer):
    """Second-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        deformable_groups (int).
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
        super(SecondOrderDeformableAlignment, self).__init__()

        self.conv_offset = nn.Sequential(
            nn.Conv2D(3 * out_channels + 4, out_channels, 3, 1, 1),
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

    def forward(self, x, extra_feat, flow_1, flow_2):
        extra_feat = paddle.concat([extra_feat, flow_1, flow_2], axis=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = paddle.chunk(out, 3, axis=1)

        # offset
        offset = 10 * paddle.tanh(paddle.concat((o1, o2), axis=1))
        offset_1, offset_2 = paddle.chunk(offset, 2, axis=1)
        offset_1 = offset_1 + flow_1.flip(1).tile(
            [1, offset_1.shape[1] // 2, 1, 1])
        offset_2 = offset_2 + flow_2.flip(1).tile(
            [1, offset_2.shape[1] // 2, 1, 1])
        offset = paddle.concat([offset_1, offset_2], axis=1)

        # mask
        mask = F.sigmoid(mask)

        out = self.dcn(x, offset, mask)
        return out
