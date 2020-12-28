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

# code was heavily based on https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
import os
import paddle
import numpy as np
from collections import OrderedDict
from abc import ABC, abstractmethod

from .criterions.builder import build_criterion
from ..solver import build_lr_scheduler, build_optimizer
from ..metrics import build_metric
from ..utils.visual import tensor2img


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:          initialize the class.
        -- <setup_input>:       unpack data from dataset and apply preprocessing.
        -- <forward>:           produce intermediate results.
        -- <train_iter>:        calculate losses, gradients, and update network weights.

    # trainer training logic:
    #
    #                build_model                               ||    model(BaseModel)
    #                     |                                    ||
    #               build_dataloader                           ||    dataloader
    #                     |                                    ||
    #               model.setup_lr_schedulers                  ||    lr_scheduler
    #                     |                                    ||
    #               model.setup_optimizers                     ||    optimizers
    #                     |                                    ||
    #     train loop (model.setup_input + model.train_iter)    ||    train loop
    #                     |                                    ||
    #         print log (model.get_current_losses)             ||
    #                     |                                    ||
    #         save checkpoint (model.nets)                     \/

    """
    def __init__(self, params=None):
        """Initialize the BaseModel class.

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <super(YourClass, self).__init__(self, cfg)>
        Then, you need to define four lists:
            -- self.losses (dict):          specify the training losses that you want to plot and save.
            -- self.nets (dict):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (dict):    define and initialize optimizers. You can define one optimizer for each network.
                                          If two networks are updated at the same time, you can use itertools.chain to group them.
                                          See cycle_gan_model.py for an example.

        Args:
            params (dict): Hyper params for train or test. Default: None.
        """
        self.params = params
        self.is_train = True if self.params is None else self.params.get(
            'is_train', True)

        self.nets = OrderedDict()
        self.optimizers = OrderedDict()
        self.metrics = OrderedDict()
        self.losses = OrderedDict()
        self.visual_items = OrderedDict()

    @abstractmethod
    def setup_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Args:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <train_iter> and <test_iter>."""
        pass

    @abstractmethod
    def train_iter(self, optims=None):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def test_iter(self, metrics=None):
        """Calculate metrics; called in every test iteration"""
        self.eval()
        with paddle.no_grad():
            self.forward()
        self.train()

    def setup_train_mode(self, is_train):
        self.is_train = is_train

    def setup_lr_schedulers(self, cfg):
        self.lr_scheduler = build_lr_scheduler(cfg)
        return self.lr_scheduler

    def setup_optimizers(self, lr, cfg):
        if cfg.get('name', None):
            cfg_ = cfg.copy()
            net_names = cfg_.pop('net_names')
            parameters = []
            for net_name in net_names:
                parameters += self.nets[net_name].parameters()
            self.optimizers['optim'] = build_optimizer(cfg_, lr, parameters)
        else:
            for opt_name, opt_cfg in cfg.items():
                cfg_ = opt_cfg.copy()
                net_names = cfg_.pop('net_names')
                parameters = []
                for net_name in net_names:
                    parameters += self.nets[net_name].parameters()
                self.optimizers[opt_name] = build_optimizer(
                    cfg_, lr, parameters)

        return self.optimizers

    def setup_metrics(self, cfg):
        if isinstance(list(cfg.values())[0], dict):
            for metric_name, cfg_ in cfg.items():
                self.metrics[metric_name] = build_metric(cfg_)
        else:
            metric = build_metric(cfg)
            self.metrics[metric.__class__.__name__] = metric

        return self.metrics

    def eval(self):
        """Make nets eval mode during test time"""
        for net in self.nets.values():
            net.eval()

    def train(self):
        """Make nets train mode during train time"""
        for net in self.nets.values():
            net.train()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        if hasattr(self, 'image_paths'):
            return self.image_paths
        return []

    def get_current_visuals(self):
        """Return visualization images."""
        return self.visual_items

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        return self.losses

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Args:
            nets (network list): a list of networks
            requires_grad (bool): whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.trainable = requires_grad
