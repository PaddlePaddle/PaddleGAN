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
from skimage.draw import circle
import matplotlib.pyplot as plt
import paddle.nn.functional as F


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
        self.Gen_Full = build_generator(generator_cfg)
        discriminator_cfg = discriminator
        discriminator_cfg.update({'common_params': common_params})
        discriminator_cfg.update({'train_params': train_params})
        self.Dis = build_discriminator(discriminator_cfg)
        self.visualizer = Visualizer()

        if isinstance(self.Gen_Full, paddle.DataParallel):
            self.nets['kp_detector'] = self.Gen_Full._layers.kp_extractor
            self.nets['generator'] = self.Gen_Full._layers.generator
            self.nets['discriminator'] = self.Dis._layers.discriminator
        else:
            self.nets['kp_detector'] = self.Gen_Full.kp_extractor
            self.nets['generator'] = self.Gen_Full.generator
            self.nets['discriminator'] = self.Dis.discriminator

        # init params
        init_weight(self.nets['kp_detector'])
        init_weight(self.nets['generator'])
        init_weight(self.nets['discriminator'])

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

    def setup_optimizers(self, lr_cfg, optimizer):
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
            self.Gen_Full(self.input_data.copy(), self.nets['discriminator'])
        self.visual_items['driving_source_gen'] = self.visualizer.visualize(
            self.input_data['driving'].detach(),
            self.input_data['source'].detach(), self.generated)

    def backward_G(self):
        loss_values = [val.mean() for val in self.losses_generator.values()]
        loss = paddle.add_n(loss_values)
        self.losses = dict(zip(self.losses_generator.keys(), loss_values))
        loss.backward()

    def backward_D(self):
        losses_discriminator = self.Dis(self.input_data.copy(), self.generated)
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


class Visualizer:
    def __init__(self, kp_size=5, draw_border=False, colormap='gist_rainbow'):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)

    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image)
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2
        num_kp = kp_array.shape[0]
        for kp_ind, kp in enumerate(kp_array):
            rr, cc = circle(kp[1], kp[0], self.kp_size, shape=image.shape[:2])
            image[rr, cc] = np.array(self.colormap(kp_ind / num_kp))[:3]
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
