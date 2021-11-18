#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import re
import copy
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.utils import spectral_norm

from ppgan.utils.photopen import build_norm_layer, simam, Dict
from .builder import DISCRIMINATORS



class NLayersDiscriminator(nn.Layer):
    def __init__(self, opt):
        super(NLayersDiscriminator, self).__init__()
        
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        input_nc = self.compute_D_input_nc(opt)
        layer_count = 0

        layer = nn.Sequential(
            nn.Conv2D(input_nc, nf, kw, 2, padw),
            nn.GELU()
        )
        self.add_sublayer('block_'+str(layer_count), layer)
        layer_count += 1

        feat_size_prev = np.floor((opt.crop_size + padw * 2 - (kw - 2)) / 2).astype('int64')
        InstanceNorm = build_norm_layer('instance')
        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            feat_size = np.floor((feat_size_prev + padw * 2 - (kw - stride)) / stride).astype('int64')
            feat_size_prev = feat_size
            layer = nn.Sequential(
                spectral_norm(nn.Conv2D(nf_prev, nf, kw, stride, padw, 
                    weight_attr=None,
                    bias_attr=None)),
                InstanceNorm(nf),
                nn.GELU()
            )
            self.add_sublayer('block_'+str(layer_count), layer)
            layer_count += 1

        layer = nn.Conv2D(nf, 1, kw, 1, padw)
        self.add_sublayer('block_'+str(layer_count), layer)
        layer_count += 1

    def forward(self, input):
        output = []
        for layer in self._sub_layers.values():
            output.append(simam(layer(input)))
            input = output[-1]

        return output

    def compute_D_input_nc(self, opt):
        input_nc = opt.label_nc + opt.output_nc
        if opt.contain_dontcare_label:
            input_nc += 1
        if not opt.no_instance:
            input_nc += 1
        return input_nc
    
@DISCRIMINATORS.register()
class MultiscaleDiscriminator(nn.Layer):
    def __init__(self,
                 ndf,
                 num_D,
                 crop_size,
                 label_nc,
                 output_nc,
                 contain_dontcare_label,
                 no_instance,
                 n_layers_D,
                 
                ):
        super(MultiscaleDiscriminator, self).__init__()
        
        opt = {
            'ndf': ndf,
            'num_D': num_D,
            'crop_size': crop_size,
            'label_nc': label_nc,
            'output_nc': output_nc,
            'contain_dontcare_label': contain_dontcare_label,
            'no_instance': no_instance,
            'n_layers_D': n_layers_D,

        }
        opt = Dict(opt)

        for i in range(opt.num_D):
            sequence = []
            crop_size_bkp = opt.crop_size
            feat_size = opt.crop_size
            for j in range(i):
                sequence += [nn.AvgPool2D(3, 2, 1)]
                feat_size = np.floor((feat_size + 1 * 2 - (3 - 2)) / 2).astype('int64')
            opt.crop_size = feat_size
            sequence += [NLayersDiscriminator(opt)]
            opt.crop_size = crop_size_bkp
            sequence = nn.Sequential(*sequence)
            self.add_sublayer('nld_'+str(i), sequence)

    def forward(self, input):
        output = []
        for layer in self._sub_layers.values():
            output.append(layer(input))
        return output

