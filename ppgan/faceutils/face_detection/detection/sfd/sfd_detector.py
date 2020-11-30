import os
import cv2
from paddle.utils.download import get_weights_path_from_url

from ..core import FaceDetector

from .net_s3fd import s3fd
from .bbox import *
from .detect import *

models_urls = {
    's3fd': 'https://paddlegan.bj.bcebos.com/models/s3fd_paddle.pdparams',
}


class SFDDetector(FaceDetector):
    def __init__(self, path_to_detector=None, verbose=False):
        super(SFDDetector, self).__init__(verbose)

        # Initialise the face detector
        if path_to_detector is None:
            model_weights_path = get_weights_path_from_url(
                models_urls['s3fd'], cur_path)
            model_weights = paddle.load(model_weights_path)
        else:
            model_weights = paddle.load(path_to_detector)

        self.face_detector = s3fd()
        self.face_detector.load_dict(model_weights)
        self.face_detector.eval()

    def detect_from_image(self, tensor_or_path):
        image = self.tensor_or_path_to_ndarray(tensor_or_path)

        bboxlist = detect(self.face_detector, image)
        keep = nms(bboxlist, 0.3)
        bboxlist = bboxlist[keep, :]
        bboxlist = [x for x in bboxlist if x[-1] > 0.5]

        return bboxlist

    def detect_from_batch(self, images):
        bboxlists = batch_detect(self.face_detector, images)
        keeps = [
            nms(bboxlists[:, i, :], 0.3) for i in range(bboxlists.shape[1])
        ]
        bboxlists = [bboxlists[keep, i, :] for i, keep in enumerate(keeps)]
        bboxlists = [[x for x in bboxlist if x[-1] > 0.5]
                     for bboxlist in bboxlists]

        return bboxlists

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0
