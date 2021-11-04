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

import math
import paddle

from paddle.optimizer.lr import LRScheduler, MultiStepDecay, LambdaDecay
from .builder import LRSCHEDULERS

LRSCHEDULERS.register(MultiStepDecay)


@LRSCHEDULERS.register()
class NonLinearDecay(LRScheduler):
    def __init__(self, learning_rate, lr_decay, last_epoch=-1):
        self.lr_decay = lr_decay
        super(NonLinearDecay, self).__init__(learning_rate, last_epoch)

    def get_lr(self):
        lr = self.base_lr / (1.0 + self.lr_decay * self.last_epoch)
        return lr


@LRSCHEDULERS.register()
class LinearDecay(LambdaDecay):
    def __init__(self, learning_rate, start_epoch, decay_epochs,
                 iters_per_epoch):
        def lambda_rule(epoch):
            epoch = epoch // iters_per_epoch
            lr_l = 1.0 - max(0,
                             epoch + 1 - start_epoch) / float(decay_epochs + 1)
            return lr_l

        super().__init__(learning_rate, lambda_rule)


@LRSCHEDULERS.register()
class CosineAnnealingRestartLR(LRScheduler):
    """ Cosine annealing with restarts learning rate scheme.

    An example config from configs/edvr_l_blur_wo_tsa.yaml:
    learning_rate: !!float 4e-4
    periods: [150000, 150000, 150000, 150000]
    restart_weights: [1, 1, 1, 1]
    eta_min: !!float 1e-7

    It has four cycles, each has 150000 iterations. At 150000th, 300000th,
    450000th, the scheduler will restart with the weights in restart_weights.

    Args:
        learning_rate (float): Base learning rate.
        periods (list): Period for each cosine anneling cycle.
        restart_weights (list): Restart weights at each restart iteration.
            Default: [1].
        eta_min (float): The mimimum learning rate of the cosine anneling cycle. Default: 0.
        last_epoch (int): Used in paddle.nn._LRScheduler. Default: -1.
    """
    def __init__(self,
                 learning_rate,
                 periods,
                 restart_weights=[1],
                 eta_min=0,
                 last_epoch=-1):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_min = eta_min
        assert (len(self.periods) == len(self.restart_weights)
                ), 'periods and restart_weights should have the same length.'
        self.cumulative_period = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]
        super(CosineAnnealingRestartLR, self).__init__(learning_rate,
                                                       last_epoch)

    def get_lr(self):
        for i, period in enumerate(self.cumulative_period):
            if self.last_epoch <= period:
                index = i
                break

        current_weight = self.restart_weights[index]
        nearest_restart = 0 if index == 0 else self.cumulative_period[index - 1]
        current_period = self.periods[index]

        lr = self.eta_min + current_weight * 0.5 * (
            self.base_lr - self.eta_min) * (1 + math.cos(math.pi * (
                (self.last_epoch - nearest_restart) / current_period)))
        return lr
