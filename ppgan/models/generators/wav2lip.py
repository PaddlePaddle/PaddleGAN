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
from paddle import nn
from paddle.nn import functional as F

from .builder import GENERATORS
from ...modules.conv import ConvBNRelu
from ...modules.conv import Conv2dTransposeRelu


@GENERATORS.register()
class Wav2Lip(nn.Layer):
    def __init__(self):
        super(Wav2Lip, self).__init__()

        self.face_encoder_blocks = nn.LayerList([
            nn.Sequential(ConvBNRelu(6, 16, kernel_size=7, stride=1,
                                     padding=3)),  # 96,96
            nn.Sequential(
                ConvBNRelu(16, 32, kernel_size=3, stride=2, padding=1),  # 48,48
                ConvBNRelu(32,
                           32,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           residual=True),
                ConvBNRelu(32,
                           32,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           residual=True)),
            nn.Sequential(
                ConvBNRelu(32, 64, kernel_size=3, stride=2, padding=1),  # 24,24
                ConvBNRelu(64,
                           64,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           residual=True),
                ConvBNRelu(64,
                           64,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           residual=True),
                ConvBNRelu(64,
                           64,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           residual=True)),
            nn.Sequential(
                ConvBNRelu(64, 128, kernel_size=3, stride=2,
                           padding=1),  # 12,12
                ConvBNRelu(128,
                           128,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           residual=True),
                ConvBNRelu(128,
                           128,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           residual=True)),
            nn.Sequential(
                ConvBNRelu(128, 256, kernel_size=3, stride=2, padding=1),  # 6,6
                ConvBNRelu(256,
                           256,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           residual=True),
                ConvBNRelu(256,
                           256,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           residual=True)),
            nn.Sequential(
                ConvBNRelu(256, 512, kernel_size=3, stride=2, padding=1),  # 3,3
                ConvBNRelu(512,
                           512,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           residual=True),
            ),
            nn.Sequential(
                ConvBNRelu(512, 512, kernel_size=3, stride=1,
                           padding=0),  # 1, 1
                ConvBNRelu(512, 512, kernel_size=1, stride=1, padding=0)),
        ])

        self.audio_encoder = nn.Sequential(
            ConvBNRelu(1, 32, kernel_size=3, stride=1, padding=1),
            ConvBNRelu(32,
                       32,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       residual=True),
            ConvBNRelu(32,
                       32,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       residual=True),
            ConvBNRelu(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            ConvBNRelu(64,
                       64,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       residual=True),
            ConvBNRelu(64,
                       64,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       residual=True),
            ConvBNRelu(64, 128, kernel_size=3, stride=3, padding=1),
            ConvBNRelu(128,
                       128,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       residual=True),
            ConvBNRelu(128,
                       128,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       residual=True),
            ConvBNRelu(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            ConvBNRelu(256,
                       256,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       residual=True),
            ConvBNRelu(256, 512, kernel_size=3, stride=1, padding=0),
            ConvBNRelu(512, 512, kernel_size=1, stride=1, padding=0),
        )

        self.face_decoder_blocks = nn.LayerList([
            nn.Sequential(
                ConvBNRelu(512, 512, kernel_size=1, stride=1, padding=0), ),
            nn.Sequential(
                Conv2dTransposeRelu(1024,
                                    512,
                                    kernel_size=3,
                                    stride=1,
                                    padding=0),  # 3,3
                ConvBNRelu(512,
                           512,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           residual=True),
            ),
            nn.Sequential(
                Conv2dTransposeRelu(1024,
                                    512,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    output_padding=1),
                ConvBNRelu(512,
                           512,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           residual=True),
                ConvBNRelu(512,
                           512,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           residual=True),
            ),  # 6, 6
            nn.Sequential(
                Conv2dTransposeRelu(768,
                                    384,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    output_padding=1),
                ConvBNRelu(384,
                           384,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           residual=True),
                ConvBNRelu(384,
                           384,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           residual=True),
            ),  # 12, 12
            nn.Sequential(
                Conv2dTransposeRelu(512,
                                    256,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    output_padding=1),
                ConvBNRelu(256,
                           256,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           residual=True),
                ConvBNRelu(256,
                           256,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           residual=True),
            ),  # 24, 24
            nn.Sequential(
                Conv2dTransposeRelu(320,
                                    128,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    output_padding=1),
                ConvBNRelu(128,
                           128,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           residual=True),
                ConvBNRelu(128,
                           128,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           residual=True),
            ),  # 48, 48
            nn.Sequential(
                Conv2dTransposeRelu(160,
                                    64,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    output_padding=1),
                ConvBNRelu(64,
                           64,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           residual=True),
                ConvBNRelu(64,
                           64,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           residual=True),
            ),
        ])  # 96,96

        self.output_block = nn.Sequential(
            ConvBNRelu(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(32, 3, kernel_size=1, stride=1, padding=0), nn.Sigmoid())

    def forward(self, audio_sequences, face_sequences):
        # audio_sequences = (B, T, 1, 80, 16)
        B = audio_sequences.shape[0]

        input_dim_size = len(face_sequences.shape)
        if input_dim_size > 4:
            audio_sequences = paddle.concat([
                audio_sequences[:, i] for i in range(audio_sequences.shape[1])
            ],
                                            axis=0)
            face_sequences = paddle.concat([
                face_sequences[:, :, i] for i in range(face_sequences.shape[2])
            ],
                                           axis=0)

        audio_embedding = self.audio_encoder(audio_sequences)  # B, 512, 1, 1

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = paddle.concat((x, feats[-1]), axis=1)
            except Exception as e:
                print(x.shape)
                print(feats[-1].shape)
                raise e

            feats.pop()

        x = self.output_block(x)

        if input_dim_size > 4:
            x = paddle.split(x, int(x.shape[0] / B), axis=0)  # [(B, C, H, W)]
            outputs = paddle.stack(x, axis=2)  # (B, C, T, H, W)

        else:
            outputs = x

        return outputs
