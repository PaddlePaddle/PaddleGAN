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
import paddle.nn.functional as F
from .base_model import BaseModel

from .builder import MODELS
from .generators.builder import build_generator
from .criterions import build_criterion
from .discriminators.builder import build_discriminator

from ..modules.init import init_weights
from ..solver import build_optimizer
from ppgan.utils.photopen import data_onehot_pro, Dict


@MODELS.register()
class PhotoPenModel(BaseModel):
    def __init__(self,
                 generator,
                 discriminator,
                 criterion,
                 label_nc,
                 contain_dontcare_label,
                 batchSize,
                 crop_size,
                 lambda_feat,
                ):

        super(PhotoPenModel, self).__init__()
        
        opt = {
             'label_nc': label_nc,
             'contain_dontcare_label': contain_dontcare_label,
             'batchSize': batchSize,
             'crop_size': crop_size,
             'lambda_feat': lambda_feat,
#              'semantic_nc': semantic_nc,
#              'use_vae': use_vae,
#              'nef': nef,
            }
        self.opt = Dict(opt)
        
        
        # define nets
        self.nets['net_gen'] = build_generator(generator)
#         init_weights(self.nets['net_gen'])
        self.nets['net_des'] = build_discriminator(discriminator)
#         init_weights(self.nets['net_des'])
        self.net_vgg = build_criterion(criterion)
    
    def setup_input(self, input):
        if 'img' in input.keys():
            self.img = paddle.to_tensor(input['img'])
        self.ins = paddle.to_tensor(input['ins'])
        self.img_paths = input['img_path']

    def forward(self):
        self.one_hot = data_onehot_pro(self.ins, self.opt)
        self.img_f = self.nets['net_gen'](self.one_hot)
        self.visual_items['img_f'] = self.img_f

    def backward_G(self):
        fake_data = paddle.concat((self.one_hot, self.img_f), 1)
        real_data = paddle.concat((self.one_hot, self.img), 1)
        fake_and_real_data = paddle.concat((fake_data, real_data), 0)
        pred = self.nets['net_des'](fake_and_real_data)
        
        """content loss"""
        g_ganloss = 0.
        for i in range(len(pred)):
            pred_i = pred[i][-1][:self.opt.batchSize]
            new_loss = -pred_i.mean() # hinge loss
            g_ganloss += new_loss
        g_ganloss /= len(pred)

        g_featloss = 0.
        for i in range(len(pred)):
            for j in range(len(pred[i]) - 1): # 除去最后一层的中间层featuremap
                unweighted_loss = (pred[i][j][:self.opt.batchSize] - pred[i][j][self.opt.batchSize:]).abs().mean() # L1 loss
                g_featloss += unweighted_loss * self.opt.lambda_feat / len(pred)
                
        g_vggloss = self.net_vgg(self.img, self.img_f)
        self.g_loss = g_ganloss + g_featloss + g_vggloss
        
        self.g_loss.backward()
        self.losses['g_ganloss'] = g_ganloss
        self.losses['g_featloss'] = g_featloss
        self.losses['g_vggloss'] = g_vggloss
        

    def backward_D(self):
        fake_data = paddle.concat((self.one_hot, self.img_f), 1)
        real_data = paddle.concat((self.one_hot, self.img), 1)
        fake_and_real_data = paddle.concat((fake_data, real_data), 0)
        pred = self.nets['net_des'](fake_and_real_data)
        
        """content loss"""
        df_ganloss = 0.
        for i in range(len(pred)):
            pred_i = pred[i][-1][:self.opt.batchSize]
            new_loss = -paddle.minimum(-pred_i - 1, paddle.zeros_like(pred_i)).mean() # hingle loss
            df_ganloss += new_loss
        df_ganloss /= len(pred)

        dr_ganloss = 0.
        for i in range(len(pred)):
            pred_i = pred[i][-1][self.opt.batchSize:]
            new_loss = -paddle.minimum(pred_i - 1, paddle.zeros_like(pred_i)).mean() # hingle loss
            dr_ganloss += new_loss
        dr_ganloss /= len(pred)

        self.d_loss = df_ganloss + dr_ganloss
        self.d_loss.backward()
        self.losses['df_ganloss'] = df_ganloss
        self.losses['dr_ganloss'] = dr_ganloss
        
        
    def train_iter(self, optimizers=None):
        self.forward()
        self.optimizers['optimG'].clear_grad()
        self.backward_G()
        self.optimizers['optimG'].step()
        
        self.forward()
        self.optimizers['optimD'].clear_grad()
        self.backward_D()
        self.optimizers['optimD'].step()
 
    def test_iter(self, metrics=None):
        self.eval()
        with paddle.no_grad():
            self.forward()
        self.train()

    def setup_optimizers(self, lr, cfg):
        for opt_name, opt_cfg in cfg.items():
            if opt_name == 'lr':
                learning_rate = opt_cfg
                continue
            cfg_ = opt_cfg.copy()
            net_names = cfg_.pop('net_names')
            parameters = []
            for net_name in net_names:
                parameters += self.nets[net_name].parameters()
            if opt_name == 'optimG':
                lr = learning_rate * 4
            else:
                lr = learning_rate
            self.optimizers[opt_name] = build_optimizer(
                cfg_, lr, parameters)

        return self.optimizers
