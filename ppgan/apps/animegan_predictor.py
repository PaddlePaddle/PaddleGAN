import os
import numpy as np
import cv2

import paddle
from .base_predictor import BasePredictor
from ppgan.datasets.transforms import ResizeToScale
import paddle.vision.transforms as T
from ppgan.models.generators import AnimeGenerator
from ppgan.utils.download import get_path_from_url


class AnimeGANPredictor(BasePredictor):
    def __init__(self,
                 output_path='output_dir',
                 weight_path=None,
                 use_adjust_brightness=True):
        self.output_path = output_path
        self.input_size = (256, 256)
        self.use_adjust_brightness = use_adjust_brightness
        if weight_path is None:
            vox_cpk_weight_url = 'https://paddlegan.bj.bcebos.com/models/animeganv2_hayao.pdparams'
            weight_path = get_path_from_url(vox_cpk_weight_url)
        self.weight_path = weight_path
        self.generator = self.load_checkpoints()
        self.transform = T.Compose([
            ResizeToScale((256, 256), 32),
            T.Transpose(),
            T.Normalize([127.5, 127.5, 127.5], [127.5, 127.5, 127.5])
        ])

    def load_checkpoints(self):
        generator = AnimeGenerator()
        checkpoint = paddle.load(self.weight_path)
        generator.set_state_dict(checkpoint['netG'])
        generator.eval()
        return generator

    @staticmethod
    def calc_avg_brightness(img):
        R = img[..., 0].mean()
        G = img[..., 1].mean()
        B = img[..., 2].mean()

        brightness = 0.299 * R + 0.587 * G + 0.114 * B
        return brightness, B, G, R

    @staticmethod
    def adjust_brightness(dst, src):
        brightness1, B1, G1, R1 = AnimeGANPredictor.calc_avg_brightness(src)
        brightness2, B2, G2, R2 = AnimeGANPredictor.calc_avg_brightness(dst)
        brightness_difference = brightness1 / brightness2
        dstf = dst * brightness_difference
        dstf = np.clip(dstf, 0, 255)
        dstf = np.uint8(dstf)
        return dstf

    def run(self, image):
        image = cv2.cvtColor(cv2.imread(image, flags=cv2.IMREAD_COLOR),
                             cv2.COLOR_BGR2RGB)
        transformed_image = self.transform(image)
        anime = (self.generator(paddle.to_tensor(transformed_image[None, ...]))
                 * 0.5 + 0.5)[0].numpy() * 255
        anime = anime.transpose((1, 2, 0))
        if self.use_adjust_brightness:
            anime = self.adjust_brightness(anime, image)
        else:
            anime = anime.astype('uint8')
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        save_path = os.path.join(self.output_path, 'anime.png')
        cv2.imwrite(save_path, cv2.cvtColor(anime, cv2.COLOR_RGB2BGR))
        return image
