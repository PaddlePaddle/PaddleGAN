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


class L2Norm(nn.Layer):
    def __init__(self, n_channels, scale=1.0):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.scale = scale
        self.eps = 1e-10
        self.weight = paddle.create_parameter(shape=[self.n_channels],
                                              dtype='float32')
        self.weight.set_value(paddle.zeros([self.n_channels]) + self.scale)

    def forward(self, x):
        norm = x.pow(2).sum(axis=1, keepdim=True).sqrt() + self.eps
        x = x / norm * self.weight.reshape([1, -1, 1, 1])
        return x


class s3fd(nn.Layer):
    def __init__(self):
        super(s3fd, self).__init__()
        self.conv1_1 = nn.Conv2D(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2D(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2D(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2D(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2D(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2D(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2D(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2D(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2D(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2D(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2D(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2D(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2D(512, 512, kernel_size=3, stride=1, padding=1)

        self.fc6 = nn.Conv2D(512, 1024, kernel_size=3, stride=1, padding=3)
        self.fc7 = nn.Conv2D(1024, 1024, kernel_size=1, stride=1, padding=0)

        self.conv6_1 = nn.Conv2D(1024, 256, kernel_size=1, stride=1, padding=0)
        self.conv6_2 = nn.Conv2D(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv7_1 = nn.Conv2D(512, 128, kernel_size=1, stride=1, padding=0)
        self.conv7_2 = nn.Conv2D(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv3_3_norm = L2Norm(256, scale=10)
        self.conv4_3_norm = L2Norm(512, scale=8)
        self.conv5_3_norm = L2Norm(512, scale=5)

        self.conv3_3_norm_mbox_conf = nn.Conv2D(256,
                                                4,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1)
        self.conv3_3_norm_mbox_loc = nn.Conv2D(256,
                                               4,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1)
        self.conv4_3_norm_mbox_conf = nn.Conv2D(512,
                                                2,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1)
        self.conv4_3_norm_mbox_loc = nn.Conv2D(512,
                                               4,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1)
        self.conv5_3_norm_mbox_conf = nn.Conv2D(512,
                                                2,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1)
        self.conv5_3_norm_mbox_loc = nn.Conv2D(512,
                                               4,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1)

        self.fc7_mbox_conf = nn.Conv2D(1024,
                                       2,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        self.fc7_mbox_loc = nn.Conv2D(1024,
                                      4,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)
        self.conv6_2_mbox_conf = nn.Conv2D(512,
                                           2,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1)
        self.conv6_2_mbox_loc = nn.Conv2D(512,
                                          4,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1)
        self.conv7_2_mbox_conf = nn.Conv2D(256,
                                           2,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1)
        self.conv7_2_mbox_loc = nn.Conv2D(256,
                                          4,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1)

    def forward(self, x):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        f3_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        f4_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        f5_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        ffc7 = h
        h = F.relu(self.conv6_1(h))
        h = F.relu(self.conv6_2(h))
        f6_2 = h
        h = F.relu(self.conv7_1(h))
        h = F.relu(self.conv7_2(h))
        f7_2 = h

        f3_3 = self.conv3_3_norm(f3_3)
        f4_3 = self.conv4_3_norm(f4_3)
        f5_3 = self.conv5_3_norm(f5_3)

        cls1 = self.conv3_3_norm_mbox_conf(f3_3)
        reg1 = self.conv3_3_norm_mbox_loc(f3_3)
        cls2 = self.conv4_3_norm_mbox_conf(f4_3)
        reg2 = self.conv4_3_norm_mbox_loc(f4_3)
        cls3 = self.conv5_3_norm_mbox_conf(f5_3)
        reg3 = self.conv5_3_norm_mbox_loc(f5_3)
        cls4 = self.fc7_mbox_conf(ffc7)
        reg4 = self.fc7_mbox_loc(ffc7)
        cls5 = self.conv6_2_mbox_conf(f6_2)
        reg5 = self.conv6_2_mbox_loc(f6_2)
        cls6 = self.conv7_2_mbox_conf(f7_2)
        reg6 = self.conv7_2_mbox_loc(f7_2)

        # max-out background label
        chunk = paddle.chunk(cls1, 4, 1)
        tmp_max = paddle.where(chunk[0] > chunk[1], chunk[0], chunk[1])
        bmax = paddle.where(tmp_max > chunk[2], tmp_max, chunk[2])
        cls1 = paddle.concat([bmax, chunk[3]], axis=1)

        return [
            cls1, reg1, cls2, reg2, cls3, reg3, cls4, reg4, cls5, reg5, cls6,
            reg6
        ]
