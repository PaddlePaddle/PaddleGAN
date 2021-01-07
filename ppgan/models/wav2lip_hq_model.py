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
import paddle.nn.functional as F
from .base_model import BaseModel

from .builder import MODELS
from .generators.builder import build_generator
from .discriminators.builder import build_discriminator
from .criterions import build_criterion

from ..solver import build_optimizer
from ..modules.init import init_weights

syncnet_T = 5
syncnet_mel_step_size = 16


class IterLoader:
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._epoch = 1

    @property
    def epoch(self):
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)


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
wav2lip_weight_path = '/workspace/PaddleGAN/wav2lip.pdparams'


@MODELS.register()
class Wav2LipModelHq(BaseModel):
    """ This class implements the Wav2lip model, Wav2lip paper: https://arxiv.org/abs/2008.10010.

    The model training requires dataset.
    By default, it uses a '--netG Wav2lip' generator,
    a '--netD SyncNetColor' discriminator.
    """
    def __init__(self,
                 generator,
                 discriminator_sync=None,
                 discriminator_hq=None,
                 syncnet_wt=1.0,
                 disc_wt=0.07,
                 is_train=True):
        """Initialize the Wav2lip class.

        Parameters:
            opt (config dict)-- stores all the experiment flags; needs to be a subclass of Dict
        """
        super(Wav2LipModelHq, self).__init__()
        self.syncnet_wt = syncnet_wt
        self.disc_wt = disc_wt
        self.is_train = is_train
        # define networks (both generator and discriminator)
        self.nets['netG'] = build_generator(generator)
        init_weights(self.nets['netG'], 'kaiming')
        #params_G = paddle.load(wav2lip_weight_path)
        #self.nets['netG'].load_dict(params_G)
        if self.is_train:
            self.nets['netDS'] = build_discriminator(discriminator_sync)
            params = paddle.load(lipsync_weight_path)
            self.nets['netDS'].load_dict(params)

            self.nets['netDH'] = build_discriminator(discriminator_hq)
            init_weights(self.nets['netDH'], 'kaiming')

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
            self.sync_loss = get_sync_loss(self.mel, self.g, self.nets['netDS'])
        else:
            self.sync_loss = 0.
        self.l1_loss = self.recon_loss(self.g, self.y)

        if self.disc_wt > 0.:
            if isinstance(self.nets['netDH'], paddle.DataParallel
                          ):  #paddle.fluid.dygraph.parallel.DataParallel)
                self.perceptual_loss = self.nets[
                    'netDH']._layers.perceptual_forward(self.g)
            else:
                self.perceptual_loss = self.nets['netDH'].perceptual_forward(
                    self.g)
        else:
            self.perceptual_loss = 0.

        self.losses['sync_loss'] = self.sync_loss
        self.losses['l1_loss'] = self.l1_loss
        self.losses['perceptual_loss'] = self.perceptual_loss

        self.loss_G = self.syncnet_wt * self.sync_loss + self.disc_wt * self.perceptual_loss + (
            1 - self.syncnet_wt - self.disc_wt) * self.l1_loss
        self.loss_G.backward()

    def backward_D(self):
        self.pred_real = self.nets['netDH'](self.y)
        self.disc_real_loss = F.binary_cross_entropy(
            self.pred_real, paddle.ones((len(self.pred_real), 1)))
        self.losses['disc_real_loss'] = self.disc_real_loss
        self.disc_real_loss.backward()

        self.pred_fake = self.nets['netDH'](self.g.detach())
        self.disc_fake_loss = F.binary_cross_entropy(
            self.pred_fake, paddle.zeros((len(self.pred_fake), 1)))
        self.losses['disc_fake_loss'] = self.disc_fake_loss
        self.disc_fake_loss.backward()

    def train_iter(self, optimizers=None):
        # forward
        self.forward()

        # update G
        self.set_requires_grad(self.nets['netDS'], False)
        self.set_requires_grad(self.nets['netG'], True)
        self.set_requires_grad(self.nets['netDH'], True)

        self.optimizers['optimizer_G'].clear_grad()
        self.optimizers['optimizer_DH'].clear_grad()
        self.backward_G()
        self.optimizers['optimizer_G'].step()

        self.optimizers['optimizer_DH'].clear_grad()
        self.backward_D()
        self.optimizers['optimizer_DH'].step()

    def test(self, test_dataloader):
        eval_steps = 700
        sync_losses, recon_losses, disc_real_losses, disc_fake_losses, perceptual_losses = [], [], [], [], []
        step = 0
        iter_loader = IterLoader(test_dataloader)
        data = next(iter_loader)
        while 1:
            step += 1
            self.setup_input(data)
            self.nets['netG'].eval()
            self.nets['netDH'].eval()
            with paddle.no_grad():
                self.forward()

                sync_loss = get_sync_loss(self.mel, self.g, self.nets['netDS'])
                l1loss = self.recon_loss(self.g, self.y)

                pred_real = self.nets['netDH'](self.y)
                pred_fake = self.nets['netDH'](self.g)
                disc_real_loss = F.binary_cross_entropy(
                    pred_real, paddle.ones((len(pred_real), 1)))
                disc_fake_loss = F.binary_cross_entropy(
                    pred_fake, paddle.zeros((len(pred_fake), 1)))

                disc_fake_losses.append(disc_fake_loss.numpy().item())
                disc_real_losses.append(disc_real_loss.numpy().item())

                sync_losses.append(sync_loss.numpy().item())
                recon_losses.append(l1loss.numpy().item())

                if self.disc_wt > 0.:
                    if isinstance(
                            self.nets['netDH'], paddle.DataParallel
                    ):  #paddle.fluid.dygraph.parallel.DataParallel)
                        perceptual_loss = self.nets[
                            'netDH']._layers.perceptual_forward(
                                self.g).numpy().item()
                    else:
                        perceptual_loss = self.nets['netDH'].perceptual_forward(
                            self.g).numpy().item()
                else:
                    perceptual_loss = 0.
                perceptual_losses.append(perceptual_loss)

                if step > eval_steps:
                    averaged_sync_loss = sum(sync_losses) / len(sync_losses)
                    averaged_recon_loss = sum(recon_losses) / len(recon_losses)
                    averaged_perceptual_loss = sum(perceptual_losses) / len(
                        perceptual_losses)
                    averaged_disc_fake_loss = sum(disc_fake_losses) / len(
                        disc_fake_losses)
                    averaged_disc_real_loss = sum(disc_real_losses) / len(
                        disc_real_losses)
                    if averaged_sync_loss < .75:
                        self.syncnet_wt = 0.01

                    print(
                        'L1: {}, Sync loss: {}, Percep: {}, Fake: {}, Real: {}'.
                        format(averaged_recon_loss, averaged_sync_loss,
                               averaged_perceptual_loss,
                               averaged_disc_fake_loss,
                               averaged_disc_real_loss))
                    break
        self.nets['netG'].train()
        self.nets['netDH'].train()
