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

import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.utils import spectral_norm

from ppgan.utils.photopen import build_norm_layer, simam, Dict
from .builder import GENERATORS

class SPADE(nn.Layer):
    def __init__(self, config_text, norm_nc, label_nc):
        super(SPADE, self).__init__()

        parsed = re.search(r'spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        self.param_free_norm = build_norm_layer(param_free_norm_type)(norm_nc)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(*[
            nn.Conv2D(label_nc, nhidden, ks, 1, pw),
            nn.GELU(),
        ])
        self.mlp_gamma = nn.Conv2D(nhidden, norm_nc, ks, 1, pw)
        self.mlp_beta = nn.Conv2D(nhidden, norm_nc, ks, 1, pw)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, x.shape[2:])
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class SPADEResnetBlock(nn.Layer):
    def __init__(self, fin, fout, opt):
        super(SPADEResnetBlock, self).__init__()

        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        
        # define spade layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.spade_0 = SPADE(spade_config_str, fin, opt.semantic_nc)
        self.spade_1 = SPADE(spade_config_str, fmiddle, opt.semantic_nc)
        if self.learned_shortcut:
            self.spade_s = SPADE(spade_config_str, fin, opt.semantic_nc)

        # define act_conv layers
        self.act_conv_0 = nn.Sequential(*[
            nn.GELU(),
            spectral_norm(nn.Conv2D(fin, fmiddle, 3, 1, 1, 
                weight_attr=None,
                bias_attr=None)),
            ])
        self.act_conv_1 = nn.Sequential(*[
            nn.GELU(),
            spectral_norm(nn.Conv2D(fmiddle, fout, 3, 1, 1, 
                weight_attr=None,
                bias_attr=None)),
            ])
        if self.learned_shortcut:
            self.act_conv_s = nn.Sequential(*[
                spectral_norm(nn.Conv2D(fin, fout, 1, 1, 0, bias_attr=False,
                    weight_attr=None)),
                ])


    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.act_conv_0(self.spade_0(x, seg))
        dx = self.act_conv_1(self.spade_1(dx, seg))

        return simam(dx + x_s)

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.act_conv_s(self.spade_s(x, seg))
        else:
            x_s = x
        return x_s

@GENERATORS.register()
class SPADEGenerator(nn.Layer):
    def __init__(self, 
                 ngf,
                 num_upsampling_layers,
                 crop_size,
                 aspect_ratio,
                 norm_G,
                 semantic_nc,
                 use_vae,
                 nef,
                 ):
        super(SPADEGenerator, self).__init__()
        
        opt = {
             'ngf': ngf,
             'num_upsampling_layers': num_upsampling_layers,
             'crop_size': crop_size,
             'aspect_ratio': aspect_ratio,
             'norm_G': norm_G,
             'semantic_nc': semantic_nc,
             'use_vae': use_vae,
             'nef': nef,
            }
        self.opt = Dict(opt)
        
        nf = self.opt.ngf
        self.sw, self.sh = self.compute_latent_vector_size(self.opt)

        if self.opt.use_vae:
            self.fc = nn.Linear(opt.z_dim, 16 * opt.nef * self.sw * self.sh)
            self.head_0 = SPADEResnetBlock(16 * opt.nef, 16 * nf, self.opt)
        else:
            self.fc = nn.Conv2D(self.opt.semantic_nc, 16 * nf, 3, 1, 1)
            self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, self.opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, self.opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, self.opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, self.opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, self.opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, self.opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, self.opt)

        final_nc = nf

        if self.opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, self.opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2D(final_nc, 3, 3, 1, 1)

        self.up = nn.Upsample(scale_factor=2)

    def forward(self, input, z=None):
        seg = input
        if self.opt.use_vae:
            x = self.fc(z)
            x = paddle.reshape(x, [-1, 16 * self.opt.nef, self.sh, self.sw])
        else:
            x = F.interpolate(seg, (self.sh, self.sw))
            x = self.fc(x)
        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)

        x = self.conv_img(F.gelu(x))
        x = F.tanh(x)

        return x

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh
    
class VAE_Encoder(nn.Layer):
    def __init__(self, opt):
        super(VAE_Encoder, self).__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.nef

        InstanceNorm = build_norm_layer('instance')
        model = [
            spectral_norm(nn.Conv2D(3, ndf, kw, 2, pw,
                    weight_attr=None,
                    bias_attr=None)),
            InstanceNorm(ndf),

            nn.GELU(),
            spectral_norm(nn.Conv2D(ndf * 1, ndf * 2, kw, 2, pw,
                    weight_attr=None,
                    bias_attr=None)),
            InstanceNorm(ndf * 2),

            nn.GELU(),
            spectral_norm(nn.Conv2D(ndf * 2, ndf * 4, kw, 2, pw,
                    weight_attr=None,
                    bias_attr=None)),
            InstanceNorm(ndf * 4),

            nn.GELU(),
            spectral_norm(nn.Conv2D(ndf * 4, ndf * 8, kw, 2, pw,
                    weight_attr=None,
                    bias_attr=None)),
            InstanceNorm(ndf * 8),

            nn.GELU(),
            spectral_norm(nn.Conv2D(ndf * 8, ndf * 8, kw, 2, pw,
                    weight_attr=None,
                    bias_attr=None)),
            InstanceNorm(ndf * 8),
        ]
        if opt.crop_size >= 256:
            model += [
                nn.GELU(),
                spectral_norm(nn.Conv2D(ndf * 8, ndf * 8, kw, 2, pw,
                        weight_attr=None,
                        bias_attr=None)),
                InstanceNorm(ndf * 8),
            ]
        model += [nn.GELU(),]

        self.flatten = nn.Flatten(1, -1)
        self.so = 4
        self.fc_mu = nn.Linear(ndf * 8 * self.so * self.so, opt.z_dim)
        self.fc_var = nn.Linear(ndf * 8 * self.so * self.so, opt.z_dim)

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        
        x = self.flatten(x)

        return self.fc_mu(x), self.fc_var(x)

