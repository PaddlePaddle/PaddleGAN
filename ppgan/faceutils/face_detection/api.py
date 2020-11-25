from __future__ import print_function
import os
import paddle
from enum import Enum
import numpy as np
import cv2

from .utils import *
import sys


class LandmarksType(Enum):
    """Enum class defining the type of landmarks to detect.

    ``_2D`` - the detected points ``(x,y)`` are detected in a 2D space and follow the visible contour of the face
    ``_2halfD`` - this points represent the projection of the 3D points into 3D
    ``_3D`` - detect the points ``(x,y,z)``` in a 3D space

    """
    _2D = 1
    _2halfD = 2
    _3D = 3


class NetworkSize(Enum):
    # TINY = 1
    # SMALL = 2
    # MEDIUM = 3
    LARGE = 4

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __int__(self):
        return self.value


ROOT = os.path.dirname(os.path.abspath(__file__))


class FaceAlignment:
    def __init__(self,
                 landmarks_type,
                 network_size=NetworkSize.LARGE,
                 device='cuda',
                 flip_input=False,
                 face_detector='sfd',
                 verbose=False):
        self.device = device
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        self.verbose = verbose

        network_size = int(network_size)

        # Get the face detector
        face_detector_module = __import__(
            'face_detection.detection.' + face_detector, globals(), locals(),
            [face_detector], 0)
        self.face_detector = face_detector_module.FaceDetector(device=device,
                                                               verbose=verbose)

    def get_detections_for_batch(self, images):
        images = images[..., ::-1]
        detected_faces = self.face_detector.detect_from_batch(images.copy())
        results = []

        for i, d in enumerate(detected_faces):
            if len(d) == 0:
                results.append(None)
                continue
            d = d[0]
            d = np.clip(d, 0, None)

            x1, y1, x2, y2 = map(int, d[:-1])
            results.append((x1, y1, x2, y2))

        return results


if __name__ == '__main__':
    detector = FaceAlignment(LandmarksType._2D)
    img = cv2.imread('./vFG638.png')
    img_e = np.expand_dims(img, 0)
    result = detector.get_detections_for_batch(img_e)
    print(result)
    x1, y1, x2, y2 = result[0]
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
    cv2.imwrite('./rec.png', img)
