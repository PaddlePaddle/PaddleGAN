# code was heavily based on https://github.com/Rudrabha/Wav2Lip
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/Rudrabha/Wav2Lip#license-and-citation

import paddle
from paddle import nn
from paddle.nn import functional as F

from ...modules.conv import ConvBNRelu, NonNormConv2d, Conv2dTransposeRelu
from .builder import DISCRIMINATORS


@DISCRIMINATORS.register()
class Wav2LipDiscQual(nn.Layer):
    def __init__(self):
        super(Wav2LipDiscQual, self).__init__()

        self.face_encoder_blocks = nn.LayerList([
            nn.Sequential(
                NonNormConv2d(3, 32, kernel_size=7, stride=1,
                              padding=3)),  # 48,96
            nn.Sequential(
                NonNormConv2d(32, 64, kernel_size=5, stride=(1, 2),
                              padding=2),  # 48,48
                NonNormConv2d(64, 64, kernel_size=5, stride=1, padding=2)),
            nn.Sequential(
                NonNormConv2d(64, 128, kernel_size=5, stride=2,
                              padding=2),  # 24,24
                NonNormConv2d(128, 128, kernel_size=5, stride=1, padding=2)),
            nn.Sequential(
                NonNormConv2d(128, 256, kernel_size=5, stride=2,
                              padding=2),  # 12,12
                NonNormConv2d(256, 256, kernel_size=5, stride=1, padding=2)),
            nn.Sequential(
                NonNormConv2d(256, 512, kernel_size=3, stride=2,
                              padding=1),  # 6,6
                NonNormConv2d(512, 512, kernel_size=3, stride=1, padding=1)),
            nn.Sequential(
                NonNormConv2d(512, 512, kernel_size=3, stride=2,
                              padding=1),  # 3,3
                NonNormConv2d(512, 512, kernel_size=3, stride=1, padding=1),
            ),
            nn.Sequential(
                NonNormConv2d(512, 512, kernel_size=3, stride=1,
                              padding=0),  # 1, 1
                NonNormConv2d(512, 512, kernel_size=1, stride=1, padding=0)),
        ])

        self.binary_pred = nn.Sequential(
            nn.Conv2D(512, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())
        self.label_noise = .0

    def get_lower_half(self, face_sequences):
        return face_sequences[:, :, face_sequences.shape[2] // 2:]

    def to_2d(self, face_sequences):
        B = face_sequences.shape[0]
        face_sequences = paddle.concat(
            [face_sequences[:, :, i] for i in range(face_sequences.shape[2])],
            axis=0)
        return face_sequences

    def perceptual_forward(self, false_face_sequences):
        false_face_sequences = self.to_2d(false_face_sequences)
        false_face_sequences = self.get_lower_half(false_face_sequences)

        false_feats = false_face_sequences
        for f in self.face_encoder_blocks:
            false_feats = f(false_feats)

        binary_pred = self.binary_pred(false_feats).reshape(
            (len(false_feats), -1))

        false_pred_loss = F.binary_cross_entropy(
            binary_pred, paddle.ones((len(false_feats), 1)))

        return false_pred_loss

    def forward(self, face_sequences):
        face_sequences = self.to_2d(face_sequences)
        face_sequences = self.get_lower_half(face_sequences)

        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)

        return paddle.reshape(self.binary_pred(x), (len(x), -1))
