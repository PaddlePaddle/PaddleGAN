# code was heavily based on https://github.com/wtjiang98/PSGAN
# MIT License 
# Copyright (c) 2020 Wentao Jiang


import os.path as osp

import numpy as np
import cv2
from PIL import Image
import paddle
import paddle.vision.transforms as T
from paddle.utils.download import get_path_from_url
import pickle
from .model import BiSeNet

BISENET_WEIGHT_URL = 'https://paddlegan.bj.bcebos.com/models/bisenet.pdparams'


class FaceParser:
    def __init__(self, device="cpu"):
        self.mapper = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 0,
            7: 11,
            8: 12,
            9: 0,
            10: 6,
            11: 8,
            12: 7,
            13: 9,
            14: 13,
            15: 0,
            16: 0,
            17: 10,
            18: 0
        }
        #self.dict = paddle.to_tensor(mapper)
        self.save_pth = get_path_from_url(BISENET_WEIGHT_URL,
                                          osp.split(osp.realpath(__file__))[0])

        self.net = BiSeNet(n_classes=19)

        self.transforms = T.Compose([
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def parse(self, image):
        assert image.shape[:2] == (512, 512)
        image = image / 255.0
        image = image.transpose((2, 0, 1))
        image = self.transforms(image)

        state_dict = paddle.load(self.save_pth)
        self.net.set_dict(state_dict)
        self.net.eval()

        with paddle.no_grad():
            image = paddle.to_tensor(image)
            image = image.unsqueeze(0)
            out = self.net(image)[0]
            parsing = out.squeeze(0).argmax(0)  #argmax(0).astype('float32')

        parse_np = parsing.numpy()
        h, w = parse_np.shape
        result = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                result[i][j] = self.mapper[parse_np[i][j]]

        result = paddle.to_tensor(result).astype('float32')
        return result
