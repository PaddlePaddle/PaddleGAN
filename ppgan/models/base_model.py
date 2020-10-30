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

from ..solver.lr_scheduler import build_lr_scheduler


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """
    def __init__(self, cfg):
        """Initialize the BaseModel class.

        Args:
            cfg (Dict)-- configs of Model.

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <super(YourClass, self).__init__(self, cfg)>
        Then, you need to define four lists:
            -- self.losses (dict):          specify the training losses that you want to plot and save.
            -- self.nets (dict):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (dict):    define and initialize optimizers. You can define one optimizer for each network.
                                          If two networks are updated at the same time, you can use itertools.chain to group them.
                                          See cycle_gan_model.py for an example.
        """
        self.cfg = cfg
        self.is_train = cfg.is_train
        self.save_dir = os.path.join(
            cfg.output_dir,
            cfg.model.name)  # save all the checkpoints to save_dir

        self.losses = OrderedDict()
        self.nets = OrderedDict()
        self.visual_items = OrderedDict()
        self.optimizers = OrderedDict()
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def build_lr_scheduler(self):
        self.lr_scheduler = build_lr_scheduler(self.cfg.lr_scheduler)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with paddle.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def get_current_visuals(self):
        """Return visualization images."""
        return self.visual_items

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        return self.losses

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Args:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.trainable = requires_grad
