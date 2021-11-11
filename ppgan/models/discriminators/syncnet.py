# code was heavily based on https://github.com/Rudrabha/Wav2Lip
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/Rudrabha/Wav2Lip#license-and-citation

import paddle
from paddle import nn
from paddle.nn import functional as F
import sys
from ...modules.conv import ConvBNRelu
#from conv import ConvBNRelu
from .builder import DISCRIMINATORS


@DISCRIMINATORS.register()
class SyncNetColor(nn.Layer):
    def __init__(self):
        super(SyncNetColor, self).__init__()

        self.face_encoder = nn.Sequential(
            ConvBNRelu(15, 32, kernel_size=(7, 7), stride=1, padding=3),
            ConvBNRelu(32, 64, kernel_size=5, stride=(1, 2), padding=1),
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
            ConvBNRelu(64, 128, kernel_size=3, stride=2, padding=1),
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
            ConvBNRelu(128,
                       128,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       residual=True),
            ConvBNRelu(128, 256, kernel_size=3, stride=2, padding=1),
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
            ConvBNRelu(256, 512, kernel_size=3, stride=2, padding=1),
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
            ConvBNRelu(512, 512, kernel_size=3, stride=2, padding=1),
            ConvBNRelu(512, 512, kernel_size=3, stride=1, padding=0),
            ConvBNRelu(512, 512, kernel_size=1, stride=1, padding=0),
        )

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
            ConvBNRelu(256,
                       256,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       residual=True),
            ConvBNRelu(256, 512, kernel_size=3, stride=1, padding=0),
            ConvBNRelu(512, 512, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, audio_sequences,
                face_sequences):  # audio_sequences := (B, dim, T)
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.reshape(
            [audio_embedding.shape[0], -1])
        face_embedding = face_embedding.reshape([face_embedding.shape[0], -1])

        audio_embedding = F.normalize(audio_embedding, p=2, axis=1)
        face_embedding = F.normalize(face_embedding, p=2, axis=1)

        return audio_embedding, face_embedding
