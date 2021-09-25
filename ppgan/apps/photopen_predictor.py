# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

from PIL import Image, ImageOps
import cv2
import numpy as np

import paddle

from .base_predictor import BasePredictor
from ppgan.models.generators import SPADEGenerator
from ppgan.utils.photopen import data_onehot_pro

class OPT():
    def __init__(self):
        super(OPT, self).__init__()
        self.D_steps_per_G=1
        self.aspect_ratio=1.0
        self.batchSize=1
        self.beta1=0.0
        self.beta2=0.9
        self.cache_filelist_read=True
        self.cache_filelist_write=True
        self.change_min=0.1
        self.checkpoints_dir='./checkpoints'
        self.coco_no_portraits=False
        self.contain_dontcare_label=True
        self.continue_train=False
        self.crop_size=256
        self.dataroot='/home/aistudio/data/scene/'
        self.dataset_mode='coco'
        self.debug=False
        self.display_freq=100
        self.display_winsize=256
        self.gan_mode='hinge'
        self.gpu_ids=[]
        self.init_type='xavier'
        self.init_variance=0.02
        self.isTrain=True
        self.label_nc=12
        self.lambda_feat=10.0
        self.lambda_kld=0.05
        self.lambda_mask=100.0
        self.lambda_vgg=0.4
        self.load_from_opt_file=False
        self.load_size=286
        self.lr=0.0002
        self.max_dataset_size=9223372036854775807
        self.model='pix2pix'
        self.nThreads=0
        self.n_layers_D=6
        self.name='label2coco'
        self.ndf=128
        self.nef=16
        self.netD='multiscale'
        self.netD_subarch='n_layer'
        self.netG='spade'
        self.ngf=64
        self.ngf=24
        self.niter=50
        self.niter_decay=0
        self.no_TTUR=False
        self.no_flip=False
        self.no_ganFeat_loss=False
        self.no_html=False
        self.no_instance=False
        self.no_pairing_check=False
        self.no_vgg_loss=False
        self.norm_D='spectralinstance'
        self.norm_E='spectralinstance'
#         self.norm_G='spectralspadesyncbatch3x3'
        self.norm_G='spectralspadebatch3x3'
        self.num_D=4
        self.num_upsampling_layers='normal'
        self.optimizer='adam'
        self.output_nc=3
        self.phase='train'
        self.preprocess_mode='resize_and_crop'
        self.print_freq=100
        self.save_epoch_freq=10
        self.save_latest_freq=5000
        self.semantic_nc=14
        self.serial_batches=False
        self.tf_log=False
        self.use_vae=False
        self.which_epoch='latest'
        self.z_dim=256

opt = OPT()

class PhotoPenPredictor(BasePredictor):
    def __init__(self,
                 output_path,
                 weight_path):

        # 初始化模型
        gen = SPADEGenerator(opt)
        gen.eval()
        gpara = paddle.load(weight_path)
        gen.set_state_dict(gpara)
        
        self.gen = gen
        self.output_path = output_path
        

    def run(self, semantic_label_path):
        sem = Image.open(semantic_label_path)
        sem = sem.resize((opt.crop_size, opt.crop_size), Image.NEAREST)
        sem = np.array(sem).astype('float32')
        sem = paddle.to_tensor(sem)
        sem = sem.reshape([1, 1, opt.crop_size, opt.crop_size])
        
        one_hot = data_onehot_pro(sem, opt)
        predicted = self.gen(one_hot)
        pic = predicted.numpy()[0].reshape((3, 256, 256)).transpose((1,2,0))
        pic = ((pic + 1.) / 2. * 255).astype('uint8')
        
        pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
        cv2.imwrite(self.output_path, pic)
        
        
        