#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
from .builder import CRITERIONS


@CRITERIONS.register()
class GradientPenalty():
    def __init__(self, loss_weight=1.0):
        self.loss_weight = loss_weight
        
    def __call__(self, net, real, fake):
        batch_size = real.shape[0]
        alpha = paddle.rand([batch_size])
        for _ in range(real.ndim - 1):
            alpha = paddle.unsqueeze(alpha, -1)
        interpolate = alpha * real + (1 - alpha) * fake
        interpolate.stop_gradient = False
        interpolate_pred = net(interpolate)
        gradient = paddle.grad(outputs=interpolate_pred, 
                               inputs=interpolate,
                               grad_outputs=paddle.ones_like(interpolate_pred),
                               create_graph=True, 
                               retain_graph=True, 
                               only_inputs=True)[0]
        gradient_penalty = ((gradient.norm(2, 1) - 1) ** 2).mean()
        return gradient_penalty * self.loss_weight
