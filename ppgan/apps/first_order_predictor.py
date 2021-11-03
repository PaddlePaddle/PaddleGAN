#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import cv2
import math

import yaml
import pickle
import imageio
import time
import numpy as np
from tqdm import tqdm, trange
from scipy.spatial import ConvexHull

import sys
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.insert(0, os.path.dirname(root_path))

import paddle
from ppgan.utils.download import get_path_from_url
from ppgan.utils.animate import normalize_kp
from ppgan.modules.keypoint_detector import KPDetector
from ppgan.models.generators.occlusion_aware import OcclusionAwareGenerator
from ppgan.faceutils import face_detection
from ppgan.faceutils.mask.face_parser import FaceParser
from ppgan.faceutils.face_detection.detection_utils import union_results, polygon2mask
from ppgan.faceutils.face_alignment.align_face import get_eyes, align_face, crop
import face_alignment
# sys.path.insert(0, '/home/user/paddle/PaddleGAN/GFPGAN/')
from gfpgan import GFPGANer
import dlib
import skimage
import moviepy.editor as mp

from .base_predictor import BasePredictor


class FirstOrderPredictor(BasePredictor):
    def __init__(self,
                 output='output',
                 weight_path=None,
                 config=None,
                 relative=False,
                 adapt_scale=False,
                 find_best_frame=False,
                 best_frame=None,
                 ratio=1.0,
                 filename='result.mp4',
                 face_detector='sfd',
                 multi_person=False,
                 image_size=256,
                 face_enhancement=False,
                 gfpgan_model_path=None, 
                 batch_size=1,
                 mobile_net=False, 
                 preprocessing=True,
                 face_align=False):
        if config is not None and isinstance(config, str):
            with open(config) as f:
                self.cfg = yaml.load(f, Loader=yaml.SafeLoader)
        elif isinstance(config, dict):
            self.cfg = config
        elif config is None:
            self.cfg = {
                'model': {
                    'common_params': {
                        'num_kp': 10,
                        'num_channels': 3,
                        'estimate_jacobian': True
                    },
                    'generator': {
                        'kp_detector_cfg': {
                            'temperature': 0.1,
                            'block_expansion': 32,
                            'max_features': 1024,
                            'scale_factor': 0.25,
                            'num_blocks': 5
                        },
                        'generator_cfg': {
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
            }
        self.image_size = image_size
        if weight_path is None:
            if mobile_net:
                vox_cpk_weight_url = 'https://paddlegan.bj.bcebos.com/applications/first_order_model/vox_mobile.pdparams'

            else:
                if self.image_size == 512:
                    vox_cpk_weight_url = 'https://paddlegan.bj.bcebos.com/applications/first_order_model/vox-cpk-512.pdparams'
                else:
                    vox_cpk_weight_url = 'https://paddlegan.bj.bcebos.com/applications/first_order_model/vox-cpk.pdparams'
            weight_path = get_path_from_url(vox_cpk_weight_url)

        self.weight_path = weight_path
        if not os.path.exists(output):
            os.makedirs(output)
        self.output = output
        self.filename = filename
        self.relative = relative
        self.adapt_scale = adapt_scale
        self.find_best_frame = find_best_frame
        self.best_frame = best_frame
        self.ratio = ratio
        self.face_detector = face_detector
        start = time.time()
        self.generator, self.kp_detector = self.load_checkpoints(
            self.cfg, self.weight_path)
        print("model loading" , time.time() - start)
        self.multi_person = multi_person
        self.face_enhancement = face_enhancement
        self.batch_size = batch_size
        # from realesrgan import RealESRGANer
        # bg_upsampler = RealESRGANer(
        #         scale=2,
        #         model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        #         tile=400,
        #         tile_pad=10,
        #         pre_pad=0,
        #         half=True) 
        self.gfpganer = GFPGANer(model_path=gfpgan_model_path, 
                                         upscale = 2, 
                                         arch = 'clean',
                                         channel_multiplier = 2,
                                         bg_upsampler = None)
        if face_enhancement:
            from ppgan.faceutils.face_enhancement import FaceEnhancement
            self.faceenhancer = FaceEnhancement(batch_size=batch_size)
            # self.faceenhancer =  GFPGANer(model_path=gfpgan_model_path, 
            #                              upscale = 2, 
            #                              arch = 'clean',
            #                              channel_multiplier = 2,
            #                              bg_upsampler = None)
        self.detection_func = union_results
        self.preprocessing = preprocessing
        self.face_alignment = face_align
     

    def read_img(self, path):
        img = imageio.imread(path)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        # som images have 4 channels
        if img.shape[2] > 3:
            img = img[:, :, :3]

        h, w, _ = img.shape
        if h >= 1024 or w >= 1024:
            if h > w:
                r = 1024.0 / h
                dim = (int(r * w), 1024)
            else:
                r = 1024.0 / w
                dim = (1024, int(r*h))
            img = cv2.resize(img, dim)
        return img

    def write_with_audio(self, audio, out_frame, fps):
        if audio is None:
            imageio.mimsave(os.path.join(self.output, self.filename),
                            [frame for frame in out_frame],
                            fps=fps)
        else:
            temp = 'tmp.mp4'
            imageio.mimsave(temp,
                            [frame for frame in out_frame],
                            fps=fps)
            videoclip_2 = mp.VideoFileClip(temp)
            videoclip_2.set_audio(audio).write_videofile(os.path.join(self.output, self.filename),
                                                            audio_codec="aac")
            os.remove(temp)


    def run(self, source_image, driving_video, filename):
        
        self.filename = filename
        videoclip_1 = mp.VideoFileClip(driving_video)
        audio = videoclip_1.audio
        def get_prediction(face_image):
            if self.find_best_frame or self.best_frame is not None:
                i = self.best_frame if self.best_frame is not None else self.find_best_frame_func(
                    source_image, driving_video)

                print("Best frame: " + str(i))
                driving_forward = driving_video[i:]
                driving_backward = driving_video[:(i + 1)][::-1]
                predictions_forward = self.make_animation(
                    face_image,
                    driving_forward,
                    self.generator,
                    self.kp_detector,
                    relative=self.relative,
                    adapt_movement_scale=self.adapt_scale)
                predictions_backward = self.make_animation(
                    face_image,
                    driving_backward,
                    self.generator,
                    self.kp_detector,
                    relative=self.relative,
                    adapt_movement_scale=self.adapt_scale)
                predictions = predictions_backward[::-1] + predictions_forward[
                                                           1:]
            else:
                predictions = self.make_animation(
                    face_image,
                    driving_video,
                    self.generator,
                    self.kp_detector,
                    relative=self.relative,
                    adapt_movement_scale=self.adapt_scale)
            return predictions

        source_image = self.read_img(source_image)
        _, _, source_image = self.gfpganer.enhance(cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR))
        source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)

        reader = imageio.get_reader(driving_video)
        fps = reader.get_meta_data()['fps']
        
        try:
            raw_driving_video = [im for im in reader]
        except RuntimeError:
            print("Read driving video error!")
            pass
        reader.close()
    
        driving_video = []
        if self.preprocessing:
            if self.face_alignment:
                self.face_alignment_preprocessing(raw_driving_video)
            else:
                self.preprocessing_video(raw_driving_video)

        driving_video = [
            cv2.resize(frame, (self.image_size, self.image_size)) / 255.0 for frame in raw_driving_video
        ]
        results = []
        start = time.time()
        bboxes, coords = self.extract_bbox(source_image.copy())
        print("extract bboxes", time.time() - start)
        print(str(len(bboxes)) + " persons have been detected")
        areas = [x[4] for x in bboxes]
        indices = np.argsort(areas)
        bboxes = bboxes[indices]
        coords = coords[indices]
        # for multi person
        # 
        for rec in bboxes:
            face_image = source_image.copy()[rec[1]:rec[3], rec[0]:rec[2]]
            face_image = cv2.resize(face_image, (self.image_size, self.image_size)) / 255.0
            predictions = get_prediction(face_image)
            results.append({'rec': rec, 'predict': [predictions[i] for i in range(predictions.shape[0])]})
            if len(bboxes) == 1 or not self.multi_person:
                break
        out_frame = []
        start = time.time()
        box_masks = self.extract_masks(results, coords, source_image)
        print("masks extraction: ", time.time()-start)
        start = time.time()

        patch = np.zeros(source_image.shape).astype('uint8')
        mask = np.zeros(source_image.shape[:2]).astype('uint8')
        for i in trange(len(driving_video)):
            frame = source_image.copy()
            # patch = np.zeros(frame.shape).astype('uint8')
            # mask = np.zeros(frame.shape[:2]).astype('uint8')
            for j, result  in enumerate(results):
                x1, y1, x2, y2, _ = result['rec']

                out = result['predict'][i]
                out = cv2.resize(out.astype(np.uint8), (x2-x1, y2-y1))
        
                if len(results) == 1:
                    frame[y1:y2, x1:x2] = out
                    break
                else:
                    #patch = np.zeros(frame.shape).astype('uint8')
                    patch[y1:y2, x1:x2] = out * np.dstack([(box_masks[j] > 0)]*3)
                    #mask = np.zeros(frame.shape[:2]).astype('uint8')
                    mask[y1:y2, x1:x2] = box_masks[j]
            frame = cv2.copyTo(patch, mask, frame)
            out_frame.append(frame)
            patch[:, :, :] = 0
            mask[:, :] = 0          

        print("video stitching", time.time() - start)
        start = time.time()
        self.write_with_audio(audio, out_frame, fps)
        print("video writing", time.time() - start)


    def load_checkpoints(self, config, checkpoint_path):

        generator = OcclusionAwareGenerator(
            **config['model']['generator']['generator_cfg'],
            **config['model']['common_params'], inference=True)

        kp_detector = KPDetector(
            **config['model']['generator']['kp_detector_cfg'],
            **config['model']['common_params'])

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
                np.array(driving_video).astype(
                    np.float32)).transpose([0, 3, 1, 2])
            
            kp_source = kp_detector(source)
            kp_driving_initial = kp_detector(driving[0:1])
            kp_source_batch = {}
            kp_source_batch["value"] = paddle.tile(kp_source["value"], repeat_times=[self.batch_size, 1, 1])
            kp_source_batch["jacobian"] = paddle.tile(kp_source["jacobian"], repeat_times=[self.batch_size, 1, 1, 1])
            source = paddle.tile(source, repeat_times=[self.batch_size, 1, 1, 1])
            begin_idx = 0
            for frame_idx in tqdm(range(int(np.ceil(float(driving.shape[0]) / self.batch_size)))):
                frame_num = min(self.batch_size, driving.shape[0] - begin_idx)
                driving_frame = driving[begin_idx: begin_idx + frame_num]
                kp_driving = kp_detector(driving_frame)
                kp_source_img = {}
                kp_source_img["value"] = kp_source_batch["value"][0:frame_num]
                kp_source_img["jacobian"] = kp_source_batch["jacobian"][0:frame_num]

                kp_norm = normalize_kp(
                    kp_source=kp_source,
                    kp_driving=kp_driving,
                    kp_driving_initial=kp_driving_initial,
                    use_relative_movement=relative,
                    use_relative_jacobian=relative,
                    adapt_movement_scale=adapt_movement_scale)

                out = generator(source[0:frame_num], kp_source=kp_source_img, kp_driving=kp_norm)
                img = np.transpose(out['prediction'].numpy(), [0, 2, 3, 1]) * 255.0
                if self.face_enhancement:
                #     _, _, img = self.faceenhancer.enhance(img[0])
                    img = self.faceenhancer.enhance_from_batch(img)
                # print(img.shape)
                predictions.append(img)
                begin_idx += frame_num
        return np.concatenate(predictions)

    def find_best_frame_func(self, source, driving):


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
            new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
            if new_norm < norm:
                norm = new_norm
                frame_num = i
        return frame_num

    def extract_bbox(self, image):
        detector = face_detection.FaceAlignment(
            face_detection.LandmarksType._2D,
            flip_input=False,
            face_detector=self.face_detector)

        # frame = [image]
        predictions = detector.get_detections_for_image(np.array(image))
        result, coords = self.detection_func(image, predictions)
        return np.array(result), np.array(coords)

    def extract_masks(self, results, coords, source_image):
        face_parcer = FaceParser()
        box_masks = []
        if len(results) != 1:
            frame = source_image.copy()         
            for i in range(len(results)):
                x1, y1, x2, y2, _ = results[i]['rec']
                polygon_mask = polygon2mask(coords[i], frame.shape[:2])[y1:y2, x1:x2]
                frame_box = cv2.resize(frame[y1:y2, x1:x2], (512, 512)) 
                box_mask = face_parcer.parse(frame_box.astype(np.float32))
                box_mask = np.array(box_mask).astype('uint8')        
                box_mask = cv2.resize(box_mask,  (x2 - x1, y2 - y1))
                box_mask[box_mask != 0] = 1
                box_masks.append(cv2.bitwise_and(box_mask, polygon_mask))
        return box_masks

    def preprocessing_video(self, raw_driving_video):
        detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
            flip_input=False,
            face_detector=self.face_detector)
     
        all_bboxes = []
        for frame in raw_driving_video:
            bboxes = detector.get_detections_for_image(np.array(frame))[0]
            bboxes = [(bboxes[2] + bboxes[0])//2, (bboxes[3]+bboxes[1])//2, bboxes[2] - bboxes[0], bboxes[3] - bboxes[1]]
            all_bboxes.append(bboxes)
        all_bboxes = np.array(all_bboxes)
        bh = max(all_bboxes[:, 2])
        bw = max(all_bboxes[:, 3])
  
        for i  in trange(len(all_bboxes)):
            all_bboxes[i] = [max(0, int(all_bboxes[i][0] - bw * 0.8)), 
                            max(0, int(all_bboxes[i][1] - bh * 0.9)), 
                            int(all_bboxes[i][0] + bw * 0.8), 
                            int(all_bboxes[i][1] + bh * 0.9)]
        for i in trange(len(all_bboxes)):
            raw_driving_video[i] = raw_driving_video[i][all_bboxes[i][1]:all_bboxes[i][3], all_bboxes[i][0]:all_bboxes[i][2]]
       
          
    def face_alignment_preprocessing(self, raw_driving_video):
        self.preprocessing_video(raw_driving_video)
        fa_predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, 
        flip_input=False, 
        face_detector='sfd')
        for i in trange(len(raw_driving_video)):
            eyes = get_eyes(fa_predictor, raw_driving_video[i])
            raw_driving_video[i] = align_face(raw_driving_video[i], eyes[0], eyes[1])
        self.preprocessing_video(raw_driving_video)
            
