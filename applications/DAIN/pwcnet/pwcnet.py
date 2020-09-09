# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D, Conv2DTranspose
from paddle.fluid.contrib import correlation

__all__ = ['pwc_dc_net']


class PWCDCNet(fluid.dygraph.Layer):
    def __init__(self, md=4):
        super(PWCDCNet, self).__init__()
        self.md = md
        self.param_attr = fluid.ParamAttr(
            regularizer=fluid.regularizer.L2DecayRegularizer(
                regularization_coeff=0.0004),
            initializer=fluid.initializer.MSRAInitializer(uniform=True,
                                                          fan_in=None,
                                                          seed=0))
        self.conv1a = Conv2D(3, 16, 3, 2, 1, param_attr=self.param_attr)
        self.conv1aa = Conv2D(16, 16, 3, 1, 1, param_attr=self.param_attr)
        self.conv1b = Conv2D(16, 16, 3, 1, 1, param_attr=self.param_attr)
        self.conv2a = Conv2D(16, 32, 3, 2, 1, param_attr=self.param_attr)
        self.conv2aa = Conv2D(32, 32, 3, 1, 1, param_attr=self.param_attr)
        self.conv2b = Conv2D(32, 32, 3, 1, 1, param_attr=self.param_attr)
        self.conv3a = Conv2D(32, 64, 3, 2, 1, param_attr=self.param_attr)
        self.conv3aa = Conv2D(64, 64, 3, 1, 1, param_attr=self.param_attr)
        self.conv3b = Conv2D(64, 64, 3, 1, 1, param_attr=self.param_attr)
        self.conv4a = Conv2D(64, 96, 3, 2, 1, param_attr=self.param_attr)
        self.conv4aa = Conv2D(96, 96, 3, 1, 1, param_attr=self.param_attr)
        self.conv4b = Conv2D(96, 96, 3, 1, 1, param_attr=self.param_attr)
        self.conv5a = Conv2D(96, 128, 3, 2, 1, param_attr=self.param_attr)
        self.conv5aa = Conv2D(128, 128, 3, 1, 1, param_attr=self.param_attr)
        self.conv5b = Conv2D(128, 128, 3, 1, 1, param_attr=self.param_attr)
        self.conv6aa = Conv2D(128, 196, 3, 2, 1, param_attr=self.param_attr)
        self.conv6a = Conv2D(196, 196, 3, 1, 1, param_attr=self.param_attr)
        self.conv6b = Conv2D(196, 196, 3, 1, 1, param_attr=self.param_attr)

        nd = (2 * self.md + 1)**2
        dd = np.cumsum([128, 128, 96, 64, 32], dtype=np.int32).astype(np.int)
        dd = [int(d) for d in dd]
        od = nd
        self.conv6_0 = Conv2D(od, 128, 3, 1, 1, param_attr=self.param_attr)
        self.conv6_1 = Conv2D(od + dd[0],
                              128,
                              3,
                              1,
                              1,
                              param_attr=self.param_attr)
        self.conv6_2 = Conv2D(od + dd[1],
                              96,
                              3,
                              1,
                              1,
                              param_attr=self.param_attr)
        self.conv6_3 = Conv2D(od + dd[2],
                              64,
                              3,
                              1,
                              1,
                              param_attr=self.param_attr)
        self.conv6_4 = Conv2D(od + dd[3],
                              32,
                              3,
                              1,
                              1,
                              param_attr=self.param_attr)
        self.predict_flow6 = Conv2D(od + dd[4],
                                    2,
                                    3,
                                    1,
                                    1,
                                    param_attr=self.param_attr)
        self.deconv6 = Conv2DTranspose(2,
                                       2,
                                       4,
                                       stride=2,
                                       padding=1,
                                       param_attr=self.param_attr)
        self.upfeat6 = Conv2DTranspose(od + dd[4],
                                       2,
                                       4,
                                       stride=2,
                                       padding=1,
                                       param_attr=self.param_attr)

        od = nd + 128 + 4
        self.conv5_0 = Conv2D(od, 128, 3, 1, 1, param_attr=self.param_attr)
        self.conv5_1 = Conv2D(od + dd[0],
                              128,
                              3,
                              1,
                              1,
                              param_attr=self.param_attr)
        self.conv5_2 = Conv2D(od + dd[1],
                              96,
                              3,
                              1,
                              1,
                              param_attr=self.param_attr)
        self.conv5_3 = Conv2D(od + dd[2],
                              64,
                              3,
                              1,
                              1,
                              param_attr=self.param_attr)
        self.conv5_4 = Conv2D(od + dd[3],
                              32,
                              3,
                              1,
                              1,
                              param_attr=self.param_attr)
        self.predict_flow5 = Conv2D(od + dd[4],
                                    2,
                                    3,
                                    1,
                                    1,
                                    param_attr=self.param_attr)
        self.deconv5 = Conv2DTranspose(2,
                                       2,
                                       4,
                                       stride=2,
                                       padding=1,
                                       param_attr=self.param_attr)
        self.upfeat5 = Conv2DTranspose(od + dd[4],
                                       2,
                                       4,
                                       stride=2,
                                       padding=1,
                                       param_attr=self.param_attr)

        od = nd + 96 + 4
        self.conv4_0 = Conv2D(od, 128, 3, 1, 1, param_attr=self.param_attr)
        self.conv4_1 = Conv2D(od + dd[0],
                              128,
                              3,
                              1,
                              1,
                              param_attr=self.param_attr)
        self.conv4_2 = Conv2D(od + dd[1],
                              96,
                              3,
                              1,
                              1,
                              param_attr=self.param_attr)
        self.conv4_3 = Conv2D(od + dd[2],
                              64,
                              3,
                              1,
                              1,
                              param_attr=self.param_attr)
        self.conv4_4 = Conv2D(od + dd[3],
                              32,
                              3,
                              1,
                              1,
                              param_attr=self.param_attr)
        self.predict_flow4 = Conv2D(od + dd[4],
                                    2,
                                    3,
                                    1,
                                    1,
                                    param_attr=self.param_attr)
        self.deconv4 = Conv2DTranspose(2,
                                       2,
                                       4,
                                       stride=2,
                                       padding=1,
                                       param_attr=self.param_attr)
        self.upfeat4 = Conv2DTranspose(od + dd[4],
                                       2,
                                       4,
                                       stride=2,
                                       padding=1,
                                       param_attr=self.param_attr)

        od = nd + 64 + 4
        self.conv3_0 = Conv2D(od, 128, 3, 1, 1, param_attr=self.param_attr)
        self.conv3_1 = Conv2D(od + dd[0],
                              128,
                              3,
                              1,
                              1,
                              param_attr=self.param_attr)
        self.conv3_2 = Conv2D(od + dd[1],
                              96,
                              3,
                              1,
                              1,
                              param_attr=self.param_attr)
        self.conv3_3 = Conv2D(od + dd[2],
                              64,
                              3,
                              1,
                              1,
                              param_attr=self.param_attr)
        self.conv3_4 = Conv2D(od + dd[3],
                              32,
                              3,
                              1,
                              1,
                              param_attr=self.param_attr)
        self.predict_flow3 = Conv2D(od + dd[4],
                                    2,
                                    3,
                                    1,
                                    1,
                                    param_attr=self.param_attr)
        self.deconv3 = Conv2DTranspose(2,
                                       2,
                                       4,
                                       stride=2,
                                       padding=1,
                                       param_attr=self.param_attr)
        self.upfeat3 = Conv2DTranspose(od + dd[4],
                                       2,
                                       4,
                                       stride=2,
                                       padding=1,
                                       param_attr=self.param_attr)

        od = nd + 32 + 4
        self.conv2_0 = Conv2D(od, 128, 3, 1, 1, param_attr=self.param_attr)
        self.conv2_1 = Conv2D(od + dd[0],
                              128,
                              3,
                              1,
                              1,
                              param_attr=self.param_attr)
        self.conv2_2 = Conv2D(od + dd[1],
                              96,
                              3,
                              1,
                              1,
                              param_attr=self.param_attr)
        self.conv2_3 = Conv2D(od + dd[2],
                              64,
                              3,
                              1,
                              1,
                              param_attr=self.param_attr)
        self.conv2_4 = Conv2D(od + dd[3],
                              32,
                              3,
                              1,
                              1,
                              param_attr=self.param_attr)
        self.predict_flow2 = Conv2D(od + dd[4],
                                    2,
                                    3,
                                    1,
                                    1,
                                    param_attr=self.param_attr)
        #        self.deconv2 = Conv2DTranspose(2, 2, 4, stride=2, padding=1, param_attr=self.param_attr)

        self.dc_conv1 = Conv2D(od + dd[4],
                               128,
                               3,
                               1,
                               1,
                               dilation=1,
                               param_attr=self.param_attr)
        self.dc_conv2 = Conv2D(128,
                               128,
                               3,
                               1,
                               2,
                               dilation=2,
                               param_attr=self.param_attr)
        self.dc_conv3 = Conv2D(128,
                               128,
                               3,
                               1,
                               4,
                               dilation=4,
                               param_attr=self.param_attr)
        self.dc_conv4 = Conv2D(128,
                               96,
                               3,
                               1,
                               8,
                               dilation=8,
                               param_attr=self.param_attr)
        self.dc_conv5 = Conv2D(96,
                               64,
                               3,
                               1,
                               16,
                               dilation=16,
                               param_attr=self.param_attr)
        self.dc_conv6 = Conv2D(64,
                               32,
                               3,
                               1,
                               1,
                               dilation=1,
                               param_attr=self.param_attr)
        self.dc_conv7 = Conv2D(32, 2, 3, 1, 1, param_attr=self.param_attr)

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        x_shape = fluid.layers.shape(x)
        B, H, W = x_shape[0], x_shape[2], x_shape[3]
        bb = fluid.layers.range(0, B, 1, 'float32')
        xx = fluid.layers.range(0, W, 1, 'float32')
        yy = fluid.layers.range(0, H, 1, 'float32')
        _, yy, xx = paddle.tensor.meshgrid(bb, yy, xx)
        yy = fluid.layers.unsqueeze(yy, [1])
        xx = fluid.layers.unsqueeze(xx, [1])
        grid = fluid.layers.concat(input=[xx, yy], axis=1)
        flo = flo
        vgrid = fluid.layers.elementwise_add(grid, flo)

        vgrid_0 = 2.0 * fluid.layers.slice(
            vgrid, axes=[1], starts=[0], ends=[1]) / (W - 1.) - 1.0
        vgrid_1 = 2.0 * fluid.layers.slice(
            vgrid, axes=[1], starts=[1], ends=[2]) / (H - 1.) - 1.0

        vgrid = fluid.layers.concat(input=[vgrid_0, vgrid_1], axis=1)
        vgrid = fluid.layers.transpose(vgrid, [0, 2, 3, 1])
        output = fluid.layers.grid_sampler(name='grid_sample', x=x, grid=vgrid)

        mask = fluid.layers.zeros_like(x)
        mask = mask + 1.0
        mask = fluid.layers.grid_sampler(name='grid_sample', x=mask, grid=vgrid)
        mask_temp1 = fluid.layers.cast(mask < 0.9990, 'float32')
        mask = mask * (1 - mask_temp1)
        mask = fluid.layers.cast(mask > 0, 'float32')
        outwarp = fluid.layers.elementwise_mul(output, mask)

        return outwarp

    def warp_nomask(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """

        B, C, H, W = x.shape
        # mesh grid
        #        xx = fluid.layers.range(0, W, 1, 'float32')
        #        xx = fluid.layers.reshape(xx, shape=[1, -1])
        #        xx = fluid.layers.expand(x=xx, expand_times=[H, 1])
        #        xx = fluid.layers.reshape(xx, shape=[1, 1, H, W])
        #        xx = fluid.layers.expand(x=xx, expand_times=[B, 1, 1, 1])
        #
        #        yy = fluid.layers.range(0, H, 1, 'float32')
        #        yy = fluid.layers.reshape(yy, shape=[-1, 1])
        #        yy = fluid.layers.expand(x=yy, expand_times=[1, W])
        #        yy = fluid.layers.reshape(x=yy, shape=[1, 1, H, W])
        #        yy = fluid.layers.expand(x=yy, expand_times=[B, 1, 1, 1])

        x_shape = fluid.layers.shape(x)
        B, H, W = x_shape[0], x_shape[2], x_shape[3]
        bb = fluid.layers.range(0, B, 1, 'float32')
        xx = fluid.layers.range(0, W, 1, 'float32')
        #        xx = fluid.layers.reshape(xx, shape=[1, -1])
        yy = fluid.layers.range(0, H, 1, 'float32')
        #        yy = fluid.layers.reshape(yy, shape=[1, -1])
        _, yy, xx = paddle.tensor.meshgrid(bb, yy, xx)
        yy = fluid.layers.unsqueeze(yy, [1])
        xx = fluid.layers.unsqueeze(xx, [1])

        grid = fluid.layers.concat(input=[xx, yy], axis=1)
        flo = flo
        vgrid = fluid.layers.elementwise_add(grid, flo)
        #vgrid_0 = 2.0 * fluid.layers.slice(vgrid, axes=[1], starts=[0], ends=[1]) / max(W - 1, 1) - 1.0
        #vgrid_1 = 2.0 * fluid.layers.slice(vgrid, axes=[1], starts=[1], ends=[2]) / max(H - 1, 1) - 1.0
        vgrid_0 = 2.0 * fluid.layers.slice(
            vgrid, axes=[1], starts=[0], ends=[1]) / (W - 1.) - 1.0
        vgrid_1 = 2.0 * fluid.layers.slice(
            vgrid, axes=[1], starts=[1], ends=[2]) / (H - 1.) - 1.0
        vgrid = fluid.layers.concat(input=[vgrid_0, vgrid_1], axis=1)
        vgrid = fluid.layers.transpose(vgrid, [0, 2, 3, 1])
        output = fluid.layers.grid_sampler(name='grid_sample', x=x, grid=vgrid)

        return output

    def corr(self, x_1, x_2):
        out = correlation(x_1,
                          x_2,
                          pad_size=self.md,
                          kernel_size=1,
                          max_displacement=self.md,
                          stride1=1,
                          stride2=1,
                          corr_type_multiply=1)
        return out

    def forward(self, x, output_more=False):
        im1 = fluid.layers.slice(x, axes=[1], starts=[0], ends=[3])
        im2 = fluid.layers.slice(x, axes=[1], starts=[3], ends=[6])
        # print("\n\n********************PWC Net details *************** \n\n")
        c11 = fluid.layers.leaky_relu(self.conv1a(im1), 0.1)
        c11 = fluid.layers.leaky_relu(self.conv1aa(c11), 0.1)
        c11 = fluid.layers.leaky_relu(self.conv1b(c11), 0.1)

        c21 = fluid.layers.leaky_relu(self.conv1a(im2), 0.1)
        c21 = fluid.layers.leaky_relu(self.conv1aa(c21), 0.1)
        c21 = fluid.layers.leaky_relu(self.conv1b(c21), 0.1)
        c12 = fluid.layers.leaky_relu(self.conv2a(c11), 0.1)
        c12 = fluid.layers.leaky_relu(self.conv2aa(c12), 0.1)
        c12 = fluid.layers.leaky_relu(self.conv2b(c12), 0.1)

        c22 = fluid.layers.leaky_relu(self.conv2a(c21), 0.1)
        c22 = fluid.layers.leaky_relu(self.conv2aa(c22), 0.1)
        c22 = fluid.layers.leaky_relu(self.conv2b(c22), 0.1)

        c13 = fluid.layers.leaky_relu(self.conv3a(c12), 0.1)
        c13 = fluid.layers.leaky_relu(self.conv3aa(c13), 0.1)
        c13 = fluid.layers.leaky_relu(self.conv3b(c13), 0.1)

        c23 = fluid.layers.leaky_relu(self.conv3a(c22), 0.1)
        c23 = fluid.layers.leaky_relu(self.conv3aa(c23), 0.1)
        c23 = fluid.layers.leaky_relu(self.conv3b(c23), 0.1)

        c14 = fluid.layers.leaky_relu(self.conv4a(c13), 0.1)
        c14 = fluid.layers.leaky_relu(self.conv4aa(c14), 0.1)
        c14 = fluid.layers.leaky_relu(self.conv4b(c14), 0.1)

        c24 = fluid.layers.leaky_relu(self.conv4a(c23), 0.1)
        c24 = fluid.layers.leaky_relu(self.conv4aa(c24), 0.1)
        c24 = fluid.layers.leaky_relu(self.conv4b(c24), 0.1)

        c15 = fluid.layers.leaky_relu(self.conv5a(c14), 0.1)
        c15 = fluid.layers.leaky_relu(self.conv5aa(c15), 0.1)
        c15 = fluid.layers.leaky_relu(self.conv5b(c15), 0.1)

        c25 = fluid.layers.leaky_relu(self.conv5a(c24), 0.1)
        c25 = fluid.layers.leaky_relu(self.conv5aa(c25), 0.1)
        c25 = fluid.layers.leaky_relu(self.conv5b(c25), 0.1)

        c16 = fluid.layers.leaky_relu(self.conv6aa(c15), 0.1)
        c16 = fluid.layers.leaky_relu(self.conv6a(c16), 0.1)
        c16 = fluid.layers.leaky_relu(self.conv6b(c16), 0.1)

        c26 = fluid.layers.leaky_relu(self.conv6aa(c25), 0.1)
        c26 = fluid.layers.leaky_relu(self.conv6a(c26), 0.1)
        c26 = fluid.layers.leaky_relu(self.conv6b(c26), 0.1)

        corr6 = self.corr(c16, c26)
        corr6 = fluid.layers.leaky_relu(corr6, alpha=0.1)

        x = fluid.layers.concat(
            input=[fluid.layers.leaky_relu(self.conv6_0(corr6), 0.1), corr6],
            axis=1)
        x = fluid.layers.concat(
            input=[fluid.layers.leaky_relu(self.conv6_1(x), 0.1), x], axis=1)
        x = fluid.layers.concat(
            input=[fluid.layers.leaky_relu(self.conv6_2(x), 0.1), x], axis=1)
        x = fluid.layers.concat(
            input=[fluid.layers.leaky_relu(self.conv6_3(x), 0.1), x], axis=1)
        x = fluid.layers.concat(
            input=[fluid.layers.leaky_relu(self.conv6_4(x), 0.1), x], axis=1)

        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)

        warp5 = self.warp(c25, up_flow6 * 0.625)
        corr5 = self.corr(c15, warp5)
        corr5 = fluid.layers.leaky_relu(corr5, alpha=0.1)

        x = fluid.layers.concat(input=[corr5, c15, up_flow6, up_feat6], axis=1)
        x = fluid.layers.concat(
            input=[fluid.layers.leaky_relu(self.conv5_0(x), 0.1), x], axis=1)
        x = fluid.layers.concat(
            input=[fluid.layers.leaky_relu(self.conv5_1(x), 0.1), x], axis=1)
        x = fluid.layers.concat(
            input=[fluid.layers.leaky_relu(self.conv5_2(x), 0.1), x], axis=1)
        x = fluid.layers.concat(
            input=[fluid.layers.leaky_relu(self.conv5_3(x), 0.1), x], axis=1)
        x = fluid.layers.concat(
            input=[fluid.layers.leaky_relu(self.conv5_4(x), 0.1), x], axis=1)

        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)

        warp4 = self.warp(c24, up_flow5 * 1.25)
        corr4 = self.corr(c14, warp4)
        corr4 = fluid.layers.leaky_relu(corr4, alpha=0.1)

        x = fluid.layers.concat(input=[corr4, c14, up_flow5, up_feat5], axis=1)
        x = fluid.layers.concat(
            input=[fluid.layers.leaky_relu(self.conv4_0(x), 0.1), x], axis=1)
        x = fluid.layers.concat(
            input=[fluid.layers.leaky_relu(self.conv4_1(x), 0.1), x], axis=1)
        x = fluid.layers.concat(
            input=[fluid.layers.leaky_relu(self.conv4_2(x), 0.1), x], axis=1)
        x = fluid.layers.concat(
            input=[fluid.layers.leaky_relu(self.conv4_3(x), 0.1), x], axis=1)
        x = fluid.layers.concat(
            input=[fluid.layers.leaky_relu(self.conv4_4(x), 0.1), x], axis=1)

        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)

        warp3 = self.warp(c23, up_flow4 * 2.5)
        corr3 = self.corr(c13, warp3)
        corr3 = fluid.layers.leaky_relu(corr3, alpha=0.1)

        x = fluid.layers.concat(input=[corr3, c13, up_flow4, up_feat4], axis=1)
        x = fluid.layers.concat(
            input=[fluid.layers.leaky_relu(self.conv3_0(x), 0.1), x], axis=1)
        x = fluid.layers.concat(
            input=[fluid.layers.leaky_relu(self.conv3_1(x), 0.1), x], axis=1)
        x = fluid.layers.concat(
            input=[fluid.layers.leaky_relu(self.conv3_2(x), 0.1), x], axis=1)
        x = fluid.layers.concat(
            input=[fluid.layers.leaky_relu(self.conv3_3(x), 0.1), x], axis=1)
        x = fluid.layers.concat(
            input=[fluid.layers.leaky_relu(self.conv3_4(x), 0.1), x], axis=1)

        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)

        warp2 = self.warp(c22, up_flow3 * 5.0)
        corr2 = self.corr(c12, warp2)
        corr2 = fluid.layers.leaky_relu(corr2, alpha=0.1)

        x = fluid.layers.concat(input=[corr2, c12, up_flow3, up_feat3], axis=1)
        x = fluid.layers.concat(
            input=[fluid.layers.leaky_relu(self.conv2_0(x), 0.1), x], axis=1)
        x = fluid.layers.concat(
            input=[fluid.layers.leaky_relu(self.conv2_1(x), 0.1), x], axis=1)
        x = fluid.layers.concat(
            input=[fluid.layers.leaky_relu(self.conv2_2(x), 0.1), x], axis=1)
        x = fluid.layers.concat(
            input=[fluid.layers.leaky_relu(self.conv2_3(x), 0.1), x], axis=1)
        x = fluid.layers.concat(
            input=[fluid.layers.leaky_relu(self.conv2_4(x), 0.1), x], axis=1)

        flow2 = self.predict_flow2(x)

        x = fluid.layers.leaky_relu(
            self.dc_conv4(
                fluid.layers.leaky_relu(
                    self.dc_conv3(
                        fluid.layers.leaky_relu(
                            self.dc_conv2(
                                fluid.layers.leaky_relu(self.dc_conv1(x), 0.1)),
                            0.1)), 0.1)), 0.1)
        flow2 += self.dc_conv7(
            fluid.layers.leaky_relu(
                self.dc_conv6(fluid.layers.leaky_relu(self.dc_conv5(x), 0.1)),
                0.1))

        if not output_more:
            return flow2
        else:
            return [flow2, flow3, flow4, flow5, flow6]


def pwc_dc_net(path=None):
    model = PWCDCNet()
    if path is not None:
        import pickle
        data = pickle.load(open(path, 'rb'))
        weight_list = []
        for k, v in data.items():
            weight_list.append(v)
        param_dict = {}
        for i, param in enumerate(model.parameters()):
            param_dict[param.name] = weight_list[i]
        model.load_dict(param_dict)

    return model
