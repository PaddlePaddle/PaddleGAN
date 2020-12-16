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


@CRITERIONS.register()
class L1Loss():
    """L1 (mean absolute error, MAE) loss.

    Args:
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.

    """
    def __init__(self, reduction='mean', loss_weight=1.0):
        # when loss weight less than zero return None
        if loss_weight <= 0:
            return None
        self._l1_loss = nn.L1Loss(reduction)
        self.loss_weight = loss_weight
        self.reduction = reduction

    def __call__(self, pred, target, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * self._l1_loss(pred, target)


@CRITERIONS.register()
class MSELoss():
    """MSE (L2) loss.

    Args:
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.

    """
    def __init__(self, reduction='mean', loss_weight=1.0):
        # when loss weight less than zero return None
        if loss_weight <= 0:
            return None
        self._l2_loss = nn.MSELoss(reduction)
        self.loss_weight = loss_weight
        self.reduction = reduction

    def __call__(self, pred, target, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * self._l2_loss(pred, target)


@CRITERIONS.register()
class BCEWithLogitsLoss():
    """BCE loss.

    Args:
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
    """
    def __init__(self, reduction='mean', loss_weight=1.0):
        # when loss weight less than zero return None
        if loss_weight <= 0:
            return None
        self._bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)
        self.loss_weight = loss_weight
        self.reduction = reduction

    def __call__(self, pred, target, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * self._bce_loss(pred, target)
