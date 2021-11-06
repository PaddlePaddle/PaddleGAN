#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class TempConv(nn.Layer):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=(1, 3, 3),
                 stride=(1, 1, 1),
                 padding=(0, 1, 1)):
        super(TempConv, self).__init__()
        self.conv3d = nn.Conv3D(in_planes,
                                out_planes,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding)
        self.bn = nn.BatchNorm(out_planes)

    def forward(self, x):
        return F.elu(self.bn(self.conv3d(x)))


class Upsample(nn.Layer):
    def __init__(self, in_planes, out_planes, scale_factor=(1, 2, 2)):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.conv3d = nn.Conv3D(in_planes,
                                out_planes,
                                kernel_size=(3, 3, 3),
                                stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.bn = nn.BatchNorm(out_planes)

    def forward(self, x):
        out_size = x.shape[2:]
        for i in range(3):
            out_size[i] = self.scale_factor[i] * out_size[i]

        return F.elu(
            self.bn(
                self.conv3d(
                    F.interpolate(x,
                                  size=out_size,
                                  mode='trilinear',
                                  align_corners=False,
                                  data_format='NCDHW',
                                  align_mode=0))))


class UpsampleConcat(nn.Layer):
    def __init__(self, in_planes_up, in_planes_flat, out_planes):
        super(UpsampleConcat, self).__init__()
        self.conv3d = TempConv(in_planes_up + in_planes_flat,
                               out_planes,
                               kernel_size=(3, 3, 3),
                               stride=(1, 1, 1),
                               padding=(1, 1, 1))

    def forward(self, x1, x2):
        scale_factor = (1, 2, 2)
        out_size = x1.shape[2:]
        for i in range(3):
            out_size[i] = scale_factor[i] * out_size[i]

        x1 = F.interpolate(x1,
                           size=out_size,
                           mode='trilinear',
                           align_corners=False,
                           data_format='NCDHW',
                           align_mode=0)
        x = paddle.concat([x1, x2], axis=1)
        return self.conv3d(x)


class SourceReferenceAttention(nn.Layer):
    """
    Source-Reference Attention Layer


    Args:
        in_planes_s (int): Number of input source feature vector channels.
        in_planes_r (int): Number of input reference feature vector channels.

    """
    def __init__(self, in_planes_s, in_planes_r):
        super(SourceReferenceAttention, self).__init__()
        self.query_conv = nn.Conv3D(in_channels=in_planes_s,
                                    out_channels=in_planes_s // 8,
                                    kernel_size=1)
        self.key_conv = nn.Conv3D(in_channels=in_planes_r,
                                  out_channels=in_planes_r // 8,
                                  kernel_size=1)
        self.value_conv = nn.Conv3D(in_channels=in_planes_r,
                                    out_channels=in_planes_r,
                                    kernel_size=1)
        self.gamma = self.create_parameter(
            shape=[1],
            dtype=self.query_conv.weight.dtype,
            default_initializer=nn.initializer.Constant(0.0))

    def forward(self, source, reference):
        s_batchsize, sC, sT, sH, sW = source.shape
        r_batchsize, rC, rT, rH, rW = reference.shape

        proj_query = paddle.reshape(self.query_conv(source),
                                    [s_batchsize, -1, sT * sH * sW])
        proj_query = paddle.transpose(proj_query, [0, 2, 1])
        proj_key = paddle.reshape(self.key_conv(reference),
                                  [r_batchsize, -1, rT * rW * rH])
        energy = paddle.bmm(proj_query, proj_key)
        attention = F.softmax(energy)

        proj_value = paddle.reshape(self.value_conv(reference),
                                    [r_batchsize, -1, rT * rH * rW])

        out = paddle.bmm(proj_value, paddle.transpose(attention, [0, 2, 1]))
        out = paddle.reshape(out, [s_batchsize, sC, sT, sH, sW])
        out = self.gamma * out + source
        return out, attention


class NetworkR(nn.Layer):
    def __init__(self):
        super(NetworkR, self).__init__()

        self.layers = nn.Sequential(
            nn.Pad3D((1, 1, 1, 1, 1, 1), mode='replicate'),
            TempConv(1,
                     64,
                     kernel_size=(3, 3, 3),
                     stride=(1, 2, 2),
                     padding=(0, 0, 0)),
            TempConv(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            TempConv(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            TempConv(128,
                     256,
                     kernel_size=(3, 3, 3),
                     stride=(1, 2, 2),
                     padding=(1, 1, 1)),
            TempConv(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            TempConv(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            TempConv(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            TempConv(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            Upsample(256, 128),
            TempConv(128, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            TempConv(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            Upsample(64, 16),
            nn.Conv3D(16,
                      1,
                      kernel_size=(3, 3, 3),
                      stride=(1, 1, 1),
                      padding=(1, 1, 1)))

    def forward(self, x):
        return paddle.clip(
            (x + F.tanh(self.layers(((x * 1).detach()) - 0.4462414))), 0.0, 1.0)


class NetworkC(nn.Layer):
    def __init__(self):
        super(NetworkC, self).__init__()

        self.down1 = nn.Sequential(
            nn.Pad3D((1, 1, 1, 1, 0, 0), mode='replicate'),
            TempConv(1, 64, stride=(1, 2, 2), padding=(0, 0, 0)),
            TempConv(64, 128), TempConv(128, 128),
            TempConv(128, 256, stride=(1, 2, 2)), TempConv(256, 256),
            TempConv(256, 256), TempConv(256, 512, stride=(1, 2, 2)),
            TempConv(512, 512), TempConv(512, 512))
        self.flat = nn.Sequential(TempConv(512, 512), TempConv(512, 512))
        self.down2 = nn.Sequential(
            TempConv(512, 512, stride=(1, 2, 2)),
            TempConv(512, 512),
        )
        self.stattn1 = SourceReferenceAttention(
            512, 512)  # Source-Reference Attention
        self.stattn2 = SourceReferenceAttention(
            512, 512)  # Source-Reference Attention
        self.selfattn1 = SourceReferenceAttention(512, 512)  # Self Attention
        self.conv1 = TempConv(512, 512)
        self.up1 = UpsampleConcat(512, 512, 512)  # 1/8
        self.selfattn2 = SourceReferenceAttention(512, 512)  # Self Attention
        self.conv2 = TempConv(512,
                              256,
                              kernel_size=(3, 3, 3),
                              stride=(1, 1, 1),
                              padding=(1, 1, 1))
        self.up2 = nn.Sequential(
            Upsample(256, 128),  # 1/4
            TempConv(128,
                     64,
                     kernel_size=(3, 3, 3),
                     stride=(1, 1, 1),
                     padding=(1, 1, 1)))
        self.up3 = nn.Sequential(
            Upsample(64, 32),  # 1/2
            TempConv(32,
                     16,
                     kernel_size=(3, 3, 3),
                     stride=(1, 1, 1),
                     padding=(1, 1, 1)))
        self.up4 = nn.Sequential(
            Upsample(16, 8),  # 1/1
            nn.Conv3D(8,
                      2,
                      kernel_size=(3, 3, 3),
                      stride=(1, 1, 1),
                      padding=(1, 1, 1)))
        self.reffeatnet1 = nn.Sequential(
            TempConv(3, 64, stride=(1, 2, 2)),
            TempConv(64, 128),
            TempConv(128, 128),
            TempConv(128, 256, stride=(1, 2, 2)),
            TempConv(256, 256),
            TempConv(256, 256),
            TempConv(256, 512, stride=(1, 2, 2)),
            TempConv(512, 512),
            TempConv(512, 512),
        )
        self.reffeatnet2 = nn.Sequential(
            TempConv(512, 512, stride=(1, 2, 2)),
            TempConv(512, 512),
            TempConv(512, 512),
        )

    def forward(self, x, x_refs=None):
        x1 = self.down1(x - 0.4462414)
        if x_refs is not None:
            x_refs = paddle.transpose(
                x_refs, [0, 2, 1, 3, 4])  # [B,T,C,H,W] --> [B,C,T,H,W]
            reffeat = self.reffeatnet1(x_refs - 0.48)
            x1, _ = self.stattn1(x1, reffeat)

        x2 = self.flat(x1)
        out = self.down2(x1)
        if x_refs is not None:
            reffeat2 = self.reffeatnet2(reffeat)
            out, _ = self.stattn2(out, reffeat2)
        out = self.conv1(out)
        out, _ = self.selfattn1(out, out)
        out = self.up1(out, x2)
        out, _ = self.selfattn2(out, out)
        out = self.conv2(out)
        out = self.up2(out)
        out = self.up3(out)
        out = self.up4(out)

        return F.sigmoid(out)
