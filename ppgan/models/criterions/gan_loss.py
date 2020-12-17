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

import numpy as np

import paddle
import paddle.nn as nn
from .builder import CRITERIONS
import paddle.nn.functional as F


@CRITERIONS.register()
class GANLoss(nn.Layer):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """
    def __init__(self,
                 gan_mode,
                 target_real_label=1.0,
                 target_fake_label=0.0,
                 loss_weight=1.0):
        """ Initialize the GANLoss class.

        Args:
            gan_mode (str): the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool): label for a real image
            target_fake_label (bool): label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        # when loss weight less than zero return None
        if loss_weight <= 0:
            return None

        self.target_real_label = target_real_label
        self.target_fake_label = target_fake_label
        self.loss_weight = loss_weight

        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgan', 'wgangp', 'hinge', 'logistic']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Args:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
        if target_is_real:
            if not hasattr(self, 'target_real_tensor'):
                self.target_real_tensor = paddle.full(
                    shape=paddle.shape(prediction),
                    fill_value=self.target_real_label,
                    dtype='float32')
            target_tensor = self.target_real_tensor
        else:
            if not hasattr(self, 'target_fake_tensor'):
                self.target_fake_tensor = paddle.full(
                    shape=paddle.shape(prediction),
                    fill_value=self.target_fake_label,
                    dtype='float32')
            target_tensor = self.target_fake_tensor

        return target_tensor

    def __call__(self,
                 prediction,
                 target_is_real,
                 is_disc=False,
                 is_updating_D=None):
        """Calculate loss given Discriminator's output and grount truth labels.

        Args:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
            is_updating_D (bool)  - - if we are in updating D step or not

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode.find('wgan') != -1:
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'hinge':
            if target_is_real:
                loss = F.relu(1 - prediction) if is_updating_D else -prediction
            else:
                loss = F.relu(1 + prediction) if is_updating_D else prediction
            loss = loss.mean()
        elif self.gan_mode == 'logistic':
            if target_is_real:
                loss = F.softplus(-prediction).mean()
            else:
                loss = F.softplus(prediction).mean()

        return loss if is_disc else loss * self.loss_weight
