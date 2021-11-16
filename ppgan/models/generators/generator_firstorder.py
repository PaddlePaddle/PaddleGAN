# code was heavily based on https://github.com/AliaksandrSiarohin/first-order-model
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/AliaksandrSiarohin/first-order-model/blob/master/LICENSE.md

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn

from ppgan.models.generators.builder import GENERATORS
from .occlusion_aware import OcclusionAwareGenerator
from ...modules.first_order import make_coordinate_grid, ImagePyramide, detach_kp
from ...modules.keypoint_detector import KPDetector

import paddle.vision.models.vgg as vgg
from ppgan.utils.download import get_path_from_url


@GENERATORS.register()
class FirstOrderGenerator(nn.Layer):
    """
    Args:
      kp_detector_cfg:
        temperature (flost): parameter of softmax
        block_expansion (int): block_expansion * (2**i) output features for each block i
        max_features (int): input features cannot larger than max_features for encoding images
        num_blocks (int): number of blocks for encoding images
      generator_cfg:
        block_expansion (int): block_expansion * (2**i) output features for each block i
        max_features (int): input features cannot larger than max_features for encoding images
        num_down_blocks (int): Downsampling block number for use in encoder.
        num_bottleneck_blocks (int): block number for use in decoder.
        estimate_occlusion_map (bool): whether to extimate occlusion_map
      common_params:
        num_kp (int): number of keypoints
        num_channels (int): image channels
        estimate_jacobian (bool): whether to estimate jacobian values of keypoints
      train_params:
        transform_params: transform keypoints and its jacobians
        scale: extract the features of image pyramids
        loss_weights: weight of [generator, discriminator, feature_matching, perceptual,
                                 equivariance_value, equivariance_jacobian]

    """
    def __init__(self, generator_cfg, kp_detector_cfg, common_params,
                 train_params, dis_scales):
        super(FirstOrderGenerator, self).__init__()
        self.kp_extractor = KPDetector(**kp_detector_cfg, **common_params)
        self.generator = OcclusionAwareGenerator(**generator_cfg,
                                                 **common_params)
        self.train_params = train_params
        self.scales = train_params['scales']
        self.disc_scales = dis_scales
        self.pyramid = ImagePyramide(self.scales, self.generator.num_channels)
        self.loss_weights = train_params['loss_weights']
        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = VGG19()

    def forward(self, x, discriminator, kp_extractor_ori=None):
        kp_source = self.kp_extractor(x['source'])
        kp_driving = self.kp_extractor(x['driving'])
        generated = self.generator(x['source'],
                                   kp_source=kp_source,
                                   kp_driving=kp_driving)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})

        loss_values = {}

        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'])
        # VGG19 perceptual Loss
        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = paddle.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
            loss_values['perceptual'] = value_total

        # Generator Loss
        if self.loss_weights['generator_gan'] != 0:
            discriminator_maps_generated = discriminator(
                pyramide_generated, kp=detach_kp(kp_driving))
            discriminator_maps_real = discriminator(pyramide_real,
                                                    kp=detach_kp(kp_driving))
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                value = ((1 - discriminator_maps_generated[key])**2).mean()
                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total
            # Feature matching Loss
            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(
                            zip(discriminator_maps_real[key],
                                discriminator_maps_generated[key])):

                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = paddle.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][
                            i] * value
                loss_values['feature_matching'] = value_total
        if (self.loss_weights['equivariance_value'] +
                self.loss_weights['equivariance_jacobian']) != 0:
            transform = Transform(x['driving'].shape[0],
                                  **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['driving'])
            transformed_kp = self.kp_extractor(transformed_frame)
            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp

            # Value loss part
            if self.loss_weights['equivariance_value'] != 0:
                value = paddle.abs(
                    kp_driving['value'] -
                    transform.warp_coordinates(transformed_kp['value'])).mean()
                loss_values['equivariance_value'] = self.loss_weights[
                    'equivariance_value'] * value

            # jacobian loss part
            if self.loss_weights['equivariance_jacobian'] != 0:
                jacobian_transformed = paddle.matmul(
                    *broadcast(transform.jacobian(transformed_kp['value']),
                               transformed_kp['jacobian']))
                normed_driving = paddle.inverse(kp_driving['jacobian'])
                normed_transformed = jacobian_transformed
                value = paddle.matmul(
                    *broadcast(normed_driving, normed_transformed))
                eye = paddle.tensor.eye(2, dtype='float32').reshape(
                    (1, 1, 2, 2))
                eye = paddle.tile(eye, [1, value.shape[1], 1, 1])
                value = paddle.abs(eye - value).mean()
                loss_values['equivariance_jacobian'] = self.loss_weights[
                    'equivariance_jacobian'] * value

        if kp_extractor_ori is not None:
            recon_loss = paddle.nn.loss.L1Loss()

            kp_distillation_loss_source = recon_loss(
                kp_extractor_ori(x['source'])['value'],
                self.kp_extractor(x['source'])['value'])
            kp_distillation_loss_driving = recon_loss(
                kp_extractor_ori(x['driving'])['value'],
                self.kp_extractor(x['driving'])['value'])
            loss_values[
                "kp_distillation_loss"] = kp_distillation_loss_source + kp_distillation_loss_driving

        return loss_values, generated


class VGG19(nn.Layer):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()
        pretrained_url = 'https://paddlegan.bj.bcebos.com/models/vgg19.pdparams'
        weight_path = get_path_from_url(pretrained_url)
        state_dict = paddle.load(weight_path)
        _vgg = getattr(vgg, 'vgg19')()
        _vgg.load_dict(state_dict)
        vgg_pretrained_features = _vgg.features
        self.slice1 = paddle.nn.Sequential()
        self.slice2 = paddle.nn.Sequential()
        self.slice3 = paddle.nn.Sequential()
        self.slice4 = paddle.nn.Sequential()
        self.slice5 = paddle.nn.Sequential()
        for x in range(2):
            self.slice1.add_sublayer(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_sublayer(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_sublayer(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_sublayer(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_sublayer(str(x), vgg_pretrained_features[x])

        self.register_buffer(
            'mean',
            paddle.to_tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]))
        # the std is for image with range [-1, 1]
        self.register_buffer(
            'std',
            paddle.to_tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1]))
        if not requires_grad:
            for param in self.parameters():
                param.stop_gradient = True

    def forward(self, x):
        x = (x - self.mean) / self.std
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class Transform:
    """
    Random tps transformation for equivariance constraints. See Sec 3.3
    """
    def __init__(self, bs, **kwargs):
        noise = paddle.distribution.Normal(loc=[0],
                                           scale=[kwargs['sigma_affine']
                                                  ]).sample([bs, 2, 3])
        noise = noise.reshape((bs, 2, 3))
        self.theta = noise + paddle.tensor.eye(2, 3, dtype='float32').reshape(
            (1, 2, 3))
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid(
                (kwargs['points_tps'], kwargs['points_tps'])).unsqueeze(0)
            buf = paddle.distribution.Normal(
                loc=[0], scale=[kwargs['sigma_tps']
                                ]).sample([bs, 1, kwargs['points_tps']**2])
            self.control_params = buf.reshape((bs, 1, kwargs['points_tps']**2))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], 'float32').unsqueeze(0)
        grid = grid.reshape((1, frame.shape[2] * frame.shape[3], 2))
        grid = self.warp_coordinates(grid).reshape(
            (self.bs, frame.shape[2], frame.shape[3], 2))
        return F.grid_sample(frame,
                             grid,
                             mode='bilinear',
                             padding_mode='reflection',
                             align_corners=True)

    def warp_coordinates(self, coordinates):
        theta = self.theta.astype('float32')
        theta = theta.unsqueeze(1)
        coordinates = coordinates.unsqueeze(-1)

        # If x1:(1, 5, 2, 2), x2:(10, 100, 2, 1)
        # torch.matmul can broadcast x1, x2 to (10, 100, ...)
        # In PDPD, it should be done manually
        theta_part_a = theta[:, :, :, :2]
        theta_part_b = theta[:, :, :, 2:]

        transformed = paddle.fluid.layers.matmul(
            *broadcast(theta_part_a, coordinates)) + theta_part_b  #M*p + m0
        transformed = transformed.squeeze(-1)
        if self.tps:
            control_points = self.control_points.astype('float32')
            control_params = self.control_params.astype('float32')
            distances = coordinates.reshape(
                (coordinates.shape[0], -1, 1, 2)) - control_points.reshape(
                    (1, 1, -1, 2))
            distances = distances.abs().sum(-1)

            result = distances * distances
            result = result * paddle.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(2).reshape((self.bs, coordinates.shape[1], 1))
            transformed = transformed + result
        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        assert len(new_coordinates.shape) == 3
        grad_x = paddle.grad(new_coordinates[:, :, 0].sum(),
                             coordinates,
                             create_graph=True)
        grad_y = paddle.grad(new_coordinates[:, :, 1].sum(),
                             coordinates,
                             create_graph=True)
        jacobian = paddle.concat(
            [grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], axis=-2)
        return jacobian


def broadcast(x, y):
    """
    Broadcast before matmul
    """
    if len(x.shape) != len(y.shape):
        raise ValueError(x.shape, '!=', y.shape)
    *dim_x, _, _ = x.shape
    *dim_y, _, _ = y.shape
    max_shape = np.max(np.stack([dim_x, dim_y], axis=0), axis=0)
    x_bc = paddle.broadcast_to(x, (*max_shape, x.shape[-2], x.shape[-1]))
    y_bc = paddle.broadcast_to(y, (*max_shape, y.shape[-2], y.shape[-1]))
    return x_bc, y_bc
