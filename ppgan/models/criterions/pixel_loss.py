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
from ..generators.generater_lapstyle import calc_mean_std, mean_variance_norm

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
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
class CharbonnierLoss():
    """Charbonnier Loss (L1).

    Args:
        eps (float): Default: 1e-12.

    """
    def __init__(self, eps=1e-12, reduction='sum'):
        self.eps = eps
        self.reduction = reduction

    def __call__(self, pred, target, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        if self.reduction == 'sum':
            out = paddle.sum(paddle.sqrt((pred - target)**2 + self.eps))
        elif self.reduction == 'mean':
            out = paddle.mean(paddle.sqrt((pred - target)**2 + self.eps))
        else:
            raise NotImplementedError('CharbonnierLoss %s not implemented' %
                                      self.reduction)
        return out


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


def calc_emd_loss(pred, target):
    """calculate emd loss.

    Args:
        pred (Tensor): of shape (N, C, H, W). Predicted tensor.
        target (Tensor): of shape (N, C, H, W). Ground truth tensor.
    """
    b, _, h, w = pred.shape
    pred = pred.reshape([b, -1, w * h])
    pred_norm = paddle.sqrt((pred**2).sum(1).reshape([b, -1, 1]))
    pred = pred.transpose([0, 2, 1])
    target_t = target.reshape([b, -1, w * h])
    target_norm = paddle.sqrt((target**2).sum(1).reshape([b, 1, -1]))
    similarity = paddle.bmm(pred, target_t) / pred_norm / target_norm
    dist = 1. - similarity
    return dist


@CRITERIONS.register()
class CalcStyleEmdLoss():
    """Calc Style Emd Loss.
    """
    def __init__(self):
        super(CalcStyleEmdLoss, self).__init__()

    def __call__(self, pred, target):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        CX_M = calc_emd_loss(pred, target)
        m1 = CX_M.min(2)
        m2 = CX_M.min(1)
        m = paddle.concat([m1.mean(), m2.mean()])
        loss_remd = paddle.max(m)
        return loss_remd


@CRITERIONS.register()
class CalcContentReltLoss():
    """Calc Content Relt Loss.
    """
    def __init__(self):
        super(CalcContentReltLoss, self).__init__()

    def __call__(self, pred, target):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        dM = 1.
        Mx = calc_emd_loss(pred, pred)
        Mx = Mx / Mx.sum(1, keepdim=True)
        My = calc_emd_loss(target, target)
        My = My / My.sum(1, keepdim=True)
        loss_content = paddle.abs(
            dM * (Mx - My)).mean() * pred.shape[2] * pred.shape[3]
        return loss_content


@CRITERIONS.register()
class CalcContentLoss():
    """Calc Content Loss.
    """
    def __init__(self):
        self.mse_loss = nn.MSELoss()

    def __call__(self, pred, target, norm=False):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            norm(Bool): whether use mean_variance_norm for pred and target
        """
        if (norm == False):
            return self.mse_loss(pred, target)
        else:
            return self.mse_loss(mean_variance_norm(pred),
                                 mean_variance_norm(target))


@CRITERIONS.register()
class CalcStyleLoss():
    """Calc Style Loss.
    """
    def __init__(self):
        self.mse_loss = nn.MSELoss()

    def __call__(self, pred, target):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        pred_mean, pred_std = calc_mean_std(pred)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(pred_mean, target_mean) + self.mse_loss(
            pred_std, target_std)


@CRITERIONS.register()
class EdgeLoss():
    def __init__(self):
        k = paddle.to_tensor([[.05, .25, .4, .25, .05]])
        self.kernel = paddle.matmul(k.t(), k).unsqueeze(0).tile([3, 1, 1, 1])
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, [kw // 2, kh // 2, kw // 2, kh // 2], mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = paddle.zeros_like(filtered)
        new_filter.stop_gradient = True
        new_filter[:, :, ::2, ::2] = down * 4  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def __call__(self, x, y):
        y.stop_gradient = True
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss
