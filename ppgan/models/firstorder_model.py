# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

# code was heavily based on https://github.com/AliaksandrSiarohin/first-order-model

import paddle

from .base_model import BaseModel
from .builder import MODELS
from .discriminators.builder import build_discriminator
from .generators.builder import build_generator
from ..modules.init import init_weights
from ..solver import build_optimizer
from paddle.optimizer.lr import MultiStepDecay
from ..modules.init import reset_parameters, uniform_
import paddle.nn as nn
import numpy as np
from paddle.utils import try_import
import paddle.nn.functional as F
import cv2


def init_weight(net):
    def reset_func(m):
        if isinstance(m, (nn.BatchNorm, nn.BatchNorm2D, nn.SyncBatchNorm)):
            m.weight = uniform_(m.weight, 0, 1)
        elif hasattr(m, 'weight') and hasattr(m, 'bias'):
            reset_parameters(m)

    net.apply(reset_func)


@MODELS.register()
class FirstOrderModel(BaseModel):
    """ This class implements the FirstOrderMotion model, FirstOrderMotion paper:
    https://proceedings.neurips.cc/paper/2019/file/31c0b36aef265d9221af80872ceb62f9-Paper.pdf.
    """
    def __init__(self,
                 common_params,
                 train_params,
                 generator,
                 discriminator=None):
        super(FirstOrderModel, self).__init__()

        # def local var
        self.input_data = None
        self.generated = None
        self.losses_generator = None
        self.train_params = train_params
        # define networks
        generator_cfg = generator
        generator_cfg.update({'common_params': common_params})
        generator_cfg.update({'train_params': train_params})
        generator_cfg.update(
            {'dis_scales': discriminator.discriminator_cfg.scales})
        self.nets['Gen_Full'] = build_generator(generator_cfg)
        discriminator_cfg = discriminator
        discriminator_cfg.update({'common_params': common_params})
        discriminator_cfg.update({'train_params': train_params})
        self.nets['Dis'] = build_discriminator(discriminator_cfg)
        self.visualizer = Visualizer()

    def setup_lr_schedulers(self, lr_cfg):
        self.kp_lr = MultiStepDecay(learning_rate=lr_cfg['lr_kp_detector'],
                                    milestones=lr_cfg['epoch_milestones'],
                                    gamma=0.1)
        self.gen_lr = MultiStepDecay(learning_rate=lr_cfg['lr_generator'],
                                     milestones=lr_cfg['epoch_milestones'],
                                     gamma=0.1)
        self.dis_lr = MultiStepDecay(learning_rate=lr_cfg['lr_discriminator'],
                                     milestones=lr_cfg['epoch_milestones'],
                                     gamma=0.1)
        self.lr_scheduler = {
            "kp_lr": self.kp_lr,
            "gen_lr": self.gen_lr,
            "dis_lr": self.dis_lr
        }
    
    def setup_net_parallel(self):
        if isinstance(self.nets['Gen_Full'], paddle.DataParallel):
            self.nets['kp_detector'] = self.nets[
                'Gen_Full']._layers.kp_extractor
            self.nets['generator'] = self.nets['Gen_Full']._layers.generator
            self.nets['discriminator'] = self.nets['Dis']._layers.discriminator
        else:
            self.nets['kp_detector'] = self.nets['Gen_Full'].kp_extractor
            self.nets['generator'] = self.nets['Gen_Full'].generator
            self.nets['discriminator'] = self.nets['Dis'].discriminator

    def setup_optimizers(self, lr_cfg, optimizer):
        self.setup_net_parallel()
        # init params
        init_weight(self.nets['kp_detector'])
        init_weight(self.nets['generator'])
        init_weight(self.nets['discriminator'])

        # define loss functions
        self.losses = {}

        self.optimizers['optimizer_KP'] = build_optimizer(
            optimizer,
            self.kp_lr,
            parameters=self.nets['kp_detector'].parameters())
        self.optimizers['optimizer_Gen'] = build_optimizer(
            optimizer,
            self.gen_lr,
            parameters=self.nets['generator'].parameters())
        self.optimizers['optimizer_Dis'] = build_optimizer(
            optimizer,
            self.dis_lr,
            parameters=self.nets['discriminator'].parameters())

    def setup_input(self, input):
        self.input_data = input

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.losses_generator, self.generated = \
            self.nets['Gen_Full'](self.input_data.copy(), self.nets['discriminator'])
        self.visual_items['driving_source_gen'] = self.visualizer.visualize(
            self.input_data['driving'].detach(),
            self.input_data['source'].detach(), self.generated)

    def backward_G(self):
        loss_values = [val.mean() for val in self.losses_generator.values()]
        loss = paddle.add_n(loss_values)
        self.losses = dict(zip(self.losses_generator.keys(), loss_values))
        loss.backward()

    def backward_D(self):
        losses_discriminator = self.nets['Dis'](self.input_data.copy(),
                                                self.generated)
        loss_values = [val.mean() for val in losses_discriminator.values()]
        loss = paddle.add_n(loss_values)
        loss.backward()
        self.losses.update(dict(zip(losses_discriminator.keys(), loss_values)))

    def train_iter(self, optimizers=None):
        self.forward()
        # update G
        self.set_requires_grad(self.nets['discriminator'], False)
        self.optimizers['optimizer_KP'].clear_grad()
        self.optimizers['optimizer_Gen'].clear_grad()
        self.backward_G()
        outs = {}
        self.optimizers['optimizer_KP'].step()
        self.optimizers['optimizer_Gen'].step()

        # update D
        if self.train_params['loss_weights']['generator_gan'] != 0:
            self.set_requires_grad(self.nets['discriminator'], True)
            self.optimizers['optimizer_Dis'].clear_grad()
            self.backward_D()
            self.optimizers['optimizer_Dis'].step()

    def test_iter(self, metrics=None):
        self.setup_net_parallel()
        self.nets['kp_detector'].eval()
        self.nets['generator'].eval()
        loss_list = []
        with paddle.no_grad():
            kp_source = self.nets['kp_detector'](self.input_data['video'][:, :,
                                                                          0])
            for frame_idx in range(self.input_data['video'].shape[2]):
                source = self.input_data['video'][:, :, 0]
                driving = self.input_data['video'][:, :, frame_idx]
                kp_driving = self.nets['kp_detector'](driving)
                out = self.nets['generator'](source,
                                             kp_source=kp_source,
                                             kp_driving=kp_driving)
                loss = paddle.abs(out['prediction'] -
                                  driving).mean().cpu().numpy()
                loss_list.append(loss)
        print("Reconstruction loss: %s" % np.mean(loss_list))
        self.nets['kp_detector'].train()
        self.nets['generator'].train()


class Visualizer:
    def __init__(self, kp_size=3, draw_border=False, colormap='gist_rainbow'):
        plt = try_import('matplotlib.pyplot')
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)

    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image)
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2
        num_kp = kp_array.shape[0]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image * 255).astype(np.uint8)
        for kp_ind, kp in enumerate(kp_array):
            color = cv2.applyColorMap(
                np.array(kp_ind / num_kp * 255).astype(np.uint8),
                cv2.COLORMAP_JET)[0][0]
            color = (int(color[0]), int(color[1]), int(color[2]))
            image = cv2.circle(image, (int(kp[1]), int(kp[0])), self.kp_size,
                               color, 3)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype('float32') / 255.0
        return image

    def create_image_column_with_kp(self, images, kp):
        image_array = np.array(
            [self.draw_image_with_kp(v, k) for v, k in zip(images, kp)])
        return self.create_image_column(image_array)

    def create_image_column(self, images, draw_border=False):
        if draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            if type(arg) == tuple:
                out.append(self.create_image_column_with_kp(arg[0], arg[1]))
            else:
                out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)

    def visualize(self, driving, source, out):
        images = []
        # Source image with keypoints
        source = source.cpu().numpy()
        kp_source = out['kp_source']['value'].cpu().numpy()
        source = np.transpose(source, [0, 2, 3, 1])
        images.append((source, kp_source))

        # Equivariance visualization
        if 'transformed_frame' in out:
            transformed = out['transformed_frame'].cpu().numpy()
            transformed = np.transpose(transformed, [0, 2, 3, 1])
            transformed_kp = out['transformed_kp']['value'].cpu().numpy()
            images.append((transformed, transformed_kp))

        # Driving image with keypoints
        kp_driving = out['kp_driving']['value'].cpu().numpy()
        driving = driving.cpu().numpy()
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append((driving, kp_driving))

        # Deformed image
        if 'deformed' in out:
            deformed = out['deformed'].cpu().numpy()
            deformed = np.transpose(deformed, [0, 2, 3, 1])
            images.append(deformed)

        # Result with and without keypoints
        prediction = out['prediction'].cpu().numpy()
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        if 'kp_norm' in out:
            kp_norm = out['kp_norm']['value'].cpu().numpy()
            images.append((prediction, kp_norm))
        images.append(prediction)

        ## Occlusion map
        if 'occlusion_map' in out:
            occlusion_map = out['occlusion_map'].cpu().tile([1, 3, 1, 1])
            occlusion_map = F.interpolate(occlusion_map,
                                          size=source.shape[1:3]).numpy()
            occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
            images.append(occlusion_map)

        # Deformed images according to each individual transform
        if 'sparse_deformed' in out:
            full_mask = []
            for i in range(out['sparse_deformed'].shape[1]):
                image = out['sparse_deformed'][:, i].cpu()
                image = F.interpolate(image, size=source.shape[1:3])
                mask = out['mask'][:, i:(i + 1)].cpu().tile([1, 3, 1, 1])
                mask = F.interpolate(mask, size=source.shape[1:3])
                image = np.transpose(image.numpy(), (0, 2, 3, 1))
                mask = np.transpose(mask.numpy(), (0, 2, 3, 1))

                if i != 0:
                    color = np.array(
                        self.colormap(
                            (i - 1) /
                            (out['sparse_deformed'].shape[1] - 1)))[:3]
                else:
                    color = np.array((0, 0, 0))

                color = color.reshape((1, 1, 1, 3))

                images.append(image)
                if i != 0:
                    images.append(mask * color)
                else:
                    images.append(mask)

                full_mask.append(mask * color)

            images.append(sum(full_mask))

        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image
