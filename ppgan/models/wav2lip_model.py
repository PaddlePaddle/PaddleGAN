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

import paddle
from paddle.utils.download import get_weights_path_from_url
from .base_model import BaseModel

from .builder import MODELS
from .generators.builder import build_generator
from .discriminators.builder import build_discriminator

from ..solver import build_optimizer
from ..modules.init import init_weights

SYNCNET_WEIGHT_URL = 'https://paddlegan.bj.bcebos.com/models/syncnet.pdparams'
syncnet_T = 5
syncnet_mel_step_size = 16


def cosine_loss(a, v, y):
    logloss = paddle.nn.BCELoss()
    d = paddle.nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)
    return loss


def get_sync_loss(mel, g, netD):
    g = g[:, :, :, g.shape[3] // 2:]
    g = paddle.concat([g[:, :, i] for i in range(syncnet_T)], axis=1)
    a, v = netD(mel, g)
    y = paddle.ones((g.shape[0], 1)).astype('float32')
    return cosine_loss(a, v, y)


lipsync_weight_path = '/workspace/PaddleGAN/lipsync_expert.pdparams'


@MODELS.register()
class Wav2LipModel(BaseModel):
    """ This class implements the Wav2lip model, Wav2lip paper: https://arxiv.org/abs/2008.10010.

    The model training requires dataset.
    By default, it uses a '--netG Wav2lip' generator,
    a '--netD SyncNetColor' discriminator.
    """
    def __init__(self,
                 generator,
                 discriminator=None,
                 syncnet_wt=1.0,
                 max_eval_steps=700,
                 is_train=True):
        """Initialize the Wav2lip class.

        Parameters:
            opt (config dict)-- stores all the experiment flags; needs to be a subclass of Dict
        """
        super(Wav2LipModel, self).__init__()
        self.syncnet_wt = syncnet_wt
        self.is_train = is_train
        self.eval_step = 0
        self.max_eval_steps = max_eval_steps
        self.eval_sync_losses, self.eval_recon_losses = [], []
        # define networks (both generator and discriminator)
        self.nets['netG'] = build_generator(generator)
        init_weights(self.nets['netG'], distribution='uniform')
        if self.is_train:
            self.nets['netD'] = build_discriminator(discriminator)
            weights_path = get_weights_path_from_url(SYNCNET_WEIGHT_URL)
            params = paddle.load(weights_path)
            self.nets['netD'].load_dict(params)

        if self.is_train:
            self.recon_loss = paddle.nn.L1Loss()

    def setup_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.x = paddle.to_tensor(input['x'])
        self.indiv_mels = paddle.to_tensor(input['indiv_mels'])
        self.mel = paddle.to_tensor(input['mel'])
        self.y = paddle.to_tensor(input['y'])

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        self.g = self.nets['netG'](self.indiv_mels, self.x)

    def backward_G(self):
        """Calculate GAN loss for the generator"""
        if self.syncnet_wt > 0.:
            self.sync_loss = get_sync_loss(self.mel, self.g, self.nets['netD'])
        else:
            self.sync_loss = 0.
        self.l1_loss = self.recon_loss(self.g, self.y)

        self.losses['sync_loss'] = self.sync_loss
        self.losses['l1_loss'] = self.l1_loss

        self.loss_G = self.syncnet_wt * self.sync_loss + (
            1 - self.syncnet_wt) * self.l1_loss
        self.loss_G.backward()

    def train_iter(self, optimizers=None):
        # forward
        self.forward()

        # update G
        self.set_requires_grad(self.nets['netD'], False)
        self.set_requires_grad(self.nets['netG'], True)
        self.optimizers['optimizer_G'].clear_grad()
        self.backward_G()
        self.optimizers['optimizer_G'].step()

    def test_iter(self, metrics=None):
        self.eval_step += 1
        self.nets['netG'].eval()
        with paddle.no_grad():
            self.forward()

            sync_loss = get_sync_loss(self.mel, self.g, self.nets['netD'])
            l1loss = self.recon_loss(self.g, self.y)

            self.eval_sync_losses.append(sync_loss.numpy().item())
        self.eval_recon_losses.append(l1loss.numpy().item())
        if self.eval_step == self.max_eval_steps:
            averaged_sync_loss = sum(self.eval_sync_losses) / len(
                self.eval_sync_losses)
            averaged_recon_loss = sum(self.eval_recon_losses) / len(
                self.eval_recon_losses)
            if averaged_sync_loss < .75:
                self.syncnet_wt = 0.01

            print('L1: {}, Sync loss: {}'.format(averaged_recon_loss,
                                                 averaged_sync_loss))
            self.eval_step = 0
            self.eval_sync_losses, self.eval_recon_losses = [], []
        self.nets['netG'].train()
