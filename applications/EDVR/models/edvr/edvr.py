#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
import math
import paddle.fluid as fluid
from paddle.fluid import ParamAttr
from ..model import ModelBase
from .edvr_model import EDVRModel
import logging
logger = logging.getLogger(__name__)

__all__ = ["EDVR"]


class EDVR(ModelBase):
    def __init__(self, name, cfg, mode='train'):
        super(EDVR, self).__init__(name, cfg, mode=mode)
        self.get_config()

    def get_config(self):
        self.num_filters = self.get_config_from_sec('model', 'num_filters')
        self.num_frames = self.get_config_from_sec('model', 'num_frames')
        self.dcn_groups = self.get_config_from_sec('model', 'deform_conv_groups')
        self.front_RBs = self.get_config_from_sec('model', 'front_RBs')
        self.back_RBs = self.get_config_from_sec('model', 'back_RBs')
        self.center = self.get_config_from_sec('model', 'center', 2)
        self.predeblur = self.get_config_from_sec('model', 'predeblur', False)
        self.HR_in = self.get_config_from_sec('model', 'HR_in', False)
        self.w_TSA = self.get_config_from_sec('model', 'w_TSA', True)

        self.crop_size = self.get_config_from_sec(self.mode, 'crop_size')
        self.scale = self.get_config_from_sec(self.mode, 'scale', 1)
        self.num_gpus = self.get_config_from_sec(self.mode, 'num_gpus', 8)
        self.batch_size = self.get_config_from_sec(self.mode, 'batch_size', 256)

        # get optimizer related parameters
        self.base_learning_rate = self.get_config_from_sec('train', 'base_learning_rate')
        self.l2_weight_decay = self.get_config_from_sec('train', 'l2_weight_decay')
        self.T_periods = self.get_config_from_sec('train', 'T_periods')
        self.restarts = self.get_config_from_sec('train', 'restarts')
        self.weights = self.get_config_from_sec('train', 'weights')
        self.eta_min = self.get_config_from_sec('train', 'eta_min')
        self.TSA_only = self.get_config_from_sec('train', 'TSA_only', False)

    def build_input(self, use_dataloader=True):
        if self.mode != 'test':
            gt_shape = [None, 3, self.crop_size, self.crop_size]
        else:
            gt_shape = [None, 3, 720, 1280]
        if self.HR_in:
            img_shape = [-1, self.num_frames, 3, self.crop_size, self.crop_size]
        else:
            if (self.mode != 'test') and (self.mode != 'infer') :
                img_shape = [None, self.num_frames, 3, \
                      int(self.crop_size / self.scale), int(self.crop_size / self.scale)]
            else:
                img_shape = [None, self.num_frames, 3, 360, 472] #180, 320]

        self.use_dataloader = use_dataloader

        image = fluid.data(name='LQ_IMGs', shape=img_shape, dtype='float32')
        if self.mode != 'infer':
            label = fluid.data(name='GT_IMG', shape=gt_shape, dtype='float32')
        else:
            label = None

        if use_dataloader:
            assert self.mode != 'infer', \
                        'dataloader is not recommendated when infer, please set use_dataloader to be false.'
            self.dataloader = fluid.io.DataLoader.from_generator(
                feed_list=[image, label], capacity=4, iterable=True)

        self.feature_input = [image]
        self.label_input = label

    def create_model_args(self):
        cfg = {}
        cfg['nf'] = self.num_filters
        cfg['nframes'] = self.num_frames
        cfg['groups'] = self.dcn_groups
        cfg['front_RBs'] = self.front_RBs
        cfg['back_RBs'] = self.back_RBs
        cfg['center'] = self.center
        cfg['predeblur'] = self.predeblur
        cfg['HR_in'] = self.HR_in
        cfg['w_TSA'] = self.w_TSA
        cfg['mode'] = self.mode
        cfg['TSA_only'] = self.TSA_only
        return cfg

    def build_model(self):
        cfg = self.create_model_args()
        videomodel = EDVRModel(**cfg)
        out = videomodel.net(self.feature_input[0])
        self.network_outputs = [out]

    def optimizer(self):
        assert self.mode == 'train', "optimizer only can be get in train mode"
        learning_rate = get_lr(base_lr = self.base_learning_rate,
                               T_periods=self.T_periods,
                               restarts=self.restarts,
                               weights=self.weights,
                               eta_min=self.eta_min)

        l2_weight_decay = self.l2_weight_decay
        optimizer = fluid.optimizer.Adam(
            learning_rate = learning_rate,
            beta1 = 0.9,
            beta2 = 0.99,
            regularization=fluid.regularizer.L2Decay(l2_weight_decay))

        return optimizer

    def loss(self):
        assert self.mode != 'infer', "invalid loss calculationg in infer mode"
        pred = self.network_outputs[0]
        label = self.label_input
        epsilon = 1e-6
        diff = pred - label
        diff = diff * diff + epsilon
        diff = fluid.layers.sqrt(diff)
        self.loss_ = fluid.layers.reduce_sum(diff)
        return self.loss_

    def outputs(self):
        return self.network_outputs

    def feeds(self):
        return self.feature_input if self.mode == 'infer' else self.feature_input + [
            self.label_input
        ]

    def fetches(self):
        if self.mode == 'train' or self.mode == 'valid':
            losses = self.loss()
            fetch_list = [losses, self.network_outputs[0], self.label_input]
        elif self.mode == 'test':
            losses = self.loss()
            fetch_list = [losses, self.network_outputs[0], self.label_input]
        elif self.mode == 'infer':
            fetch_list = self.network_outputs
        else:
            raise NotImplementedError('mode {} not implemented'.format(
                self.mode))

        return fetch_list

    def pretrain_info(self):
        return (
            None,
            None
        )

    def weights_info(self):
        return (
            None,
            None
        )

    def load_pretrain_params0(self, exe, pretrain, prog, place):
        """load pretrain form .npz which is created by torch"""
        def is_parameter(var):
            return isinstance(var, fluid.framework.Parameter)
        params_list = list(filter(is_parameter, prog.list_vars()))

        import numpy as np
        state_dict = np.load(pretrain)
        for p in params_list:
            if p.name in state_dict.keys():
                print('########### load param {} from file'.format(p.name))
            else:
                print('----------- param {} not in file'.format(p.name))
        fluid.set_program_state(prog, state_dict)
        print('load pretrain from ', pretrain)

    def load_test_weights(self, exe, weights, prog, place):
        """load weights from .npz which is created by torch"""
        def is_parameter(var):
            return isinstance(var, fluid.framework.Parameter)
        params_list = list(filter(is_parameter, prog.list_vars()))

        import numpy as np
        state_dict = np.load(weights)
        for p in params_list:
            if p.name in state_dict.keys():
                print('########### load param {} from file'.format(p.name))
            else:
                print('----------- param {} not in file'.format(p.name))
        fluid.set_program_state(prog, state_dict)
        print('load weights from ', weights)


# This is for learning rate cosine annealing restart

Dtype='float32'

def decay_step_counter(begin=0):
    # the first global step is zero in learning rate decay
    global_step = fluid.layers.autoincreased_step_counter(
        counter_name='@LR_DECAY_COUNTER@', begin=begin, step=1)
    return global_step


def get_lr(base_lr = 0.001,
           T_periods = [250000, 250000, 250000, 250000],
           restarts = [250000, 500000, 750000],
           weights=[1, 1, 1],
           eta_min=0):
    with fluid.default_main_program()._lr_schedule_guard():
        global_step = decay_step_counter()
        lr = fluid.layers.create_global_var(shape=[1], value=base_lr, dtype=Dtype, persistable=True, name="learning_rate")
        num_segs = len(restarts)
        restart_point = 0
        with fluid.layers.Switch() as switch:
            with switch.case(global_step == 0):
                pass
            for i in range(num_segs):
                T_max = T_periods[i]
                weight = weights[i]
                with switch.case(global_step < restarts[i]):
                    with fluid.layers.Switch() as switch_second:
                        value_2Tmax = fluid.layers.fill_constant(shape=[1], dtype='int64', value=2*T_max)
                        step_checker = global_step-restart_point-1-T_max
                        with switch_second.case(fluid.layers.elementwise_mod(step_checker, value_2Tmax)==0):
                            var_value = lr + (base_lr - eta_min) * (1 - math.cos(math.pi / float(T_max))) / 2
                            fluid.layers.assign(var_value, lr)
                        with switch_second.default():
                            double_step = fluid.layers.cast(global_step, dtype='float64') - float(restart_point)
                            double_scale = (1 + fluid.layers.cos(math.pi * double_step / float(T_max))) / \
                                           (1 + fluid.layers.cos(math.pi * (double_step - 1) / float(T_max)))
                            float_scale = fluid.layers.cast(double_scale, dtype=Dtype)
                            var_value = float_scale * (lr - eta_min) + eta_min
                            fluid.layers.assign(var_value, lr)
                with switch.case(global_step == restarts[i]):
                    var_value = fluid.layers.fill_constant(
                        shape=[1], dtype=Dtype, value=float(base_lr*weight))
                    fluid.layers.assign(var_value, lr)
                restart_point = restarts[i]
            T_max = T_periods[num_segs]
            with switch.default():
                with fluid.layers.Switch() as switch_second:
                    value_2Tmax = fluid.layers.fill_constant(shape=[1], dtype='int64', value=2*T_max)
                    step_checker = global_step-restart_point-1-T_max
                    with switch_second.case(fluid.layers.elementwise_mod(step_checker, value_2Tmax)==0):
                        var_value = lr + (base_lr - eta_min) * (1 - math.cos(math.pi / float(T_max))) / 2
                        fluid.layers.assign(var_value, lr)
                    with switch_second.default():
                        double_step = fluid.layers.cast(global_step, dtype='float64') - float(restart_point)
                        double_scale = (1 + fluid.layers.cos(math.pi * double_step / float(T_max))) / \
                                       (1 + fluid.layers.cos(math.pi * (double_step - 1) / float(T_max)))
                        float_scale = fluid.layers.cast(double_scale, dtype=Dtype)
                        var_value = float_scale * (lr - eta_min) + eta_min
                        fluid.layers.assign(var_value, lr)
        return lr

