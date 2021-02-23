#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import os
import sys

import yaml
import pickle
import imageio
import numpy as np
from tqdm import tqdm
from skimage import img_as_ubyte
from skimage.transform import resize
from scipy.spatial import ConvexHull

import paddle
from ppgan.utils.download import get_path_from_url
from ppgan.utils.animate import normalize_kp
from ppgan.modules.keypoint_detector import KPDetector
from ppgan.models.generators.occlusion_aware import OcclusionAwareGenerator

from .base_predictor import BasePredictor


class FirstOrderPredictor(BasePredictor):
    def __init__(self,
                 output='output',
                 weight_path=None,
                 config=None,
                 relative=False,
                 adapt_scale=False,
                 find_best_frame=False,
                 best_frame=None):
        if config is not None and isinstance(config, str):
            self.cfg = yaml.load(config, Loader=yaml.SafeLoader)
        elif isinstance(config, dict):
            self.cfg = config
        elif config is None:
            self.cfg = {
                'model_params': {
                    'common_params': {
                        'num_kp': 10,
                        'num_channels': 3,
                        'estimate_jacobian': True
                    },
                    'kp_detector_params': {
                        'temperature': 0.1,
                        'block_expansion': 32,
                        'max_features': 1024,
                        'scale_factor': 0.25,
                        'num_blocks': 5
                    },
                    'generator_params': {
                        'block_expansion': 64,
                        'max_features': 512,
                        'num_down_blocks': 2,
                        'num_bottleneck_blocks': 6,
                        'estimate_occlusion_map': True,
                        'dense_motion_params': {
                            'block_expansion': 64,
                            'max_features': 1024,
                            'num_blocks': 5,
                            'scale_factor': 0.25
                        }
                    }
                }
            }
            if weight_path is None:
                vox_cpk_weight_url = 'https://paddlegan.bj.bcebos.com/applications/first_order_model/vox-cpk.pdparams'
                weight_path = get_path_from_url(vox_cpk_weight_url)

        self.weight_path = weight_path
        if not os.path.exists(output):
            os.makedirs(output)
        self.output = output
        self.relative = relative
        self.adapt_scale = adapt_scale
        self.find_best_frame = find_best_frame
        self.best_frame = best_frame
        self.generator, self.kp_detector = self.load_checkpoints(
            self.cfg, self.weight_path)

    def run(self, source_image, driving_video):
        source_image = imageio.imread(source_image)
        reader = imageio.get_reader(driving_video)
        fps = reader.get_meta_data()['fps']
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()

        source_image = resize(source_image, (256, 256))[..., :3]
        driving_video = [
            resize(frame, (256, 256))[..., :3] for frame in driving_video
        ]

        if self.find_best_frame or self.best_frame is not None:
            i = self.best_frame if self.best_frame is not None else self.find_best_frame_func(
                source_image, driving_video)

            print("Best frame: " + str(i))
            driving_forward = driving_video[i:]
            driving_backward = driving_video[:(i + 1)][::-1]
            predictions_forward = self.make_animation(
                source_image,
                driving_forward,
                self.generator,
                self.kp_detector,
                relative=self.relative,
                adapt_movement_scale=self.adapt_scale)
            predictions_backward = self.make_animation(
                source_image,
                driving_backward,
                self.generator,
                self.kp_detector,
                relative=self.relative,
                adapt_movement_scale=self.adapt_scale)
            predictions = predictions_backward[::-1] + predictions_forward[1:]
        else:
            predictions = self.make_animation(
                source_image,
                driving_video,
                self.generator,
                self.kp_detector,
                relative=self.relative,
                adapt_movement_scale=self.adapt_scale)
            imageio.mimsave(os.path.join(self.output, 'result.mp4'),
                            [img_as_ubyte(frame) for frame in predictions],
                            fps=fps)

    def load_checkpoints(self, config, checkpoint_path):

        generator = OcclusionAwareGenerator(
            **config['model_params']['generator_params'],
            **config['model_params']['common_params'])

        kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                                 **config['model_params']['common_params'])

        checkpoint = paddle.load(self.weight_path)
        generator.set_state_dict(checkpoint['generator'])

        kp_detector.set_state_dict(checkpoint['kp_detector'])

        generator.eval()
        kp_detector.eval()

        return generator, kp_detector

    def make_animation(self,
                       source_image,
                       driving_video,
                       generator,
                       kp_detector,
                       relative=True,
                       adapt_movement_scale=True):
        with paddle.no_grad():
            predictions = []
            source = paddle.to_tensor(source_image[np.newaxis].astype(
                np.float32)).transpose([0, 3, 1, 2])

            driving = paddle.to_tensor(
                np.array(driving_video)[np.newaxis].astype(
                    np.float32)).transpose([0, 4, 1, 2, 3])
            kp_source = kp_detector(source)
            kp_driving_initial = kp_detector(driving[:, :, 0])

            for frame_idx in tqdm(range(driving.shape[2])):
                driving_frame = driving[:, :, frame_idx]
                kp_driving = kp_detector(driving_frame)
                kp_norm = normalize_kp(
                    kp_source=kp_source,
                    kp_driving=kp_driving,
                    kp_driving_initial=kp_driving_initial,
                    use_relative_movement=relative,
                    use_relative_jacobian=relative,
                    adapt_movement_scale=adapt_movement_scale)
                out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

                predictions.append(
                    np.transpose(out['prediction'].numpy(), [0, 2, 3, 1])[0])
        return predictions

    def find_best_frame_func(self, source, driving):
        import face_alignment

        def normalize_kp(kp):
            kp = kp - kp.mean(axis=0, keepdims=True)
            area = ConvexHull(kp[:, :2]).volume
            area = np.sqrt(area)
            kp[:, :2] = kp[:, :2] / area
            return kp

        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                          flip_input=True)

        kp_source = fa.get_landmarks(255 * source)[0]
        kp_source = normalize_kp(kp_source)
        norm = float('inf')
        frame_num = 0
        for i, image in tqdm(enumerate(driving)):
            kp_driving = fa.get_landmarks(255 * image)[0]
            kp_driving = normalize_kp(kp_driving)
            new_norm = (np.abs(kp_source - kp_driving)**2).sum()
            if new_norm < norm:
                norm = new_norm
                frame_num = i
        return frame_num
