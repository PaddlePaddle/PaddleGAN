import os
import sys
import argparse
import numpy as np

import paddle
from paddle.jit import TracedLayer

import ppgan
from ppgan.utils.config import get_config
from ppgan.utils.setup import setup
from ppgan.engine.trainer import Trainer
from ppgan.utils.animate import normalize_kp

import numpy as np
from scipy.spatial import ConvexHull

import paddle
from paddle.static import InputSpec


def normalize_kp(kp_source,
                 kp_driving,
                 kp_driving_initial,
                 adapt_movement_scale=False,
                 use_relative_movement=False,
                 use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = paddle.matmul(
                kp_driving['jacobian'],
                paddle.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = paddle.matmul(jacobian_diff,
                                               kp_source['jacobian'])

    return kp_new
    

class KpDetector(paddle.nn.Layer):
    def __init__(self, cfg, weight_path):
        super(KpDetector, self).__init__()
        model = ppgan.models.builder.build_model(cfg.model)
        model.setup_train_mode(is_train=False)
        weight = paddle.load(weight_path)

        model.nets['Gen_Full'].kp_extractor.set_state_dict(weight['kp_detector'])
        
        self.kp_detector = model.nets['Gen_Full'].kp_extractor
    
    def forward(self, image):

        return self.kp_detector(image)

class Generator(paddle.nn.Layer):
    def __init__(self, cfg, weight_path):
        super(Generator, self).__init__()
        model = ppgan.models.builder.build_model(cfg.model)
        model.setup_train_mode(is_train=False)
        weight = paddle.load(weight_path)

        model.nets['Gen_Full'].generator.set_state_dict(weight['generator'])

        self.generator = model.nets['Gen_Full'].generator

    def forward(self, source, kp_source, kp_driving, kp_driving_initial):
        relative = True
        adapt_movement_scale = False

        kp_norm = normalize_kp(
            kp_source=kp_source,
            kp_driving=kp_driving,
            kp_driving_initial=kp_driving_initial,
            use_relative_movement=relative,
            use_relative_jacobian=relative,
            adapt_movement_scale=adapt_movement_scale)
        # print('generator shape:', source.shape, kp_source.shape, kp_norm.shape)
        out = self.generator(source, kp_source=kp_source, kp_driving=kp_norm)
        return out['prediction']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="config of model.")
    parser.add_argument(
        "--weight_path",
        type=str,
        default=None,
        help="weight path")
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="export weight path")
    args = parser.parse_args()
    
    cfg = get_config(args.config)
    kp_detector = KpDetector(cfg, args.weight_path)
    generator = Generator(cfg, args.weight_path)

    source = np.random.rand(1, 3, 256, 256).astype('float32')
    driving = np.random.rand(1, 3, 256, 256).astype('float32')
    value = np.random.rand(1, 10, 2).astype('float32')
    j = np.random.rand(1, 10, 2, 2).astype('float32')

    source = paddle.to_tensor(source)
    driving1 = {'value': paddle.to_tensor(value), 'jacobian': paddle.to_tensor(j)}
    driving2 = {'value': paddle.to_tensor(value), 'jacobian': paddle.to_tensor(j)}
    driving3 = {'value': paddle.to_tensor(value), 'jacobian': paddle.to_tensor(j)}
    driving = paddle.to_tensor(driving)
    outpath = os.path.join(args.output_path, "fom_dy2st")
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    paddle.jit.save(kp_detector, os.path.join(outpath, "kp_detector"), input_spec=[source])
    paddle.jit.save(generator, os.path.join(outpath, "generator"), input_spec=[source, driving1, driving2, driving3])
