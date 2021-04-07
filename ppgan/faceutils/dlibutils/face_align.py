#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import cv2
import math
import numpy as np
from PIL import Image

from .dlib_utils import detect, landmarks


def align_crop(image: Image):
    faces = detect(image)

    assert len(faces) > 0, 'Cannot detect face!!!'

    face = get_max_face(faces)
    lms = landmarks(image, face)
    lms = lms[:, ::-1]

    image = np.array(image)
    image_align, landmarks_align = align(image, lms)
    image_crop = crop(image_align, landmarks_align)
    return image_crop


def get_max_face(faces):
    if len(faces) == 1:
        return faces[0]

    else:
        # find max face
        areas = []
        for face in faces:
            left = face.left()
            top = face.top()
            right = face.right()
            bottom = face.bottom()
            areas.append((bottom - top) * (right - left))
        max_face_index = np.argmax(areas)
        return faces[max_face_index]


def align(image, lms):
    # rotation angle
    left_eye_corner = lms[36]
    right_eye_corner = lms[45]
    radian = np.arctan((left_eye_corner[1] - right_eye_corner[1]) / (left_eye_corner[0] - right_eye_corner[0]))

    # image size after rotating
    height, width, _ = image.shape
    cos = math.cos(radian)
    sin = math.sin(radian)
    new_w = int(width * abs(cos) + height * abs(sin))
    new_h = int(width * abs(sin) + height * abs(cos))

    # translation
    Tx = new_w // 2 - width // 2
    Ty = new_h // 2 - height // 2

    # affine matrix
    M = np.array([[cos, sin, (1 - cos) * width / 2. - sin * height / 2. + Tx],
                  [-sin, cos, sin * width / 2. + (1 - cos) * height / 2. + Ty]])

    image_rotate = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(255, 255, 255))

    landmarks = np.concatenate([lms, np.ones((lms.shape[0], 1))], axis=1)
    landmarks_rotate = np.dot(M, landmarks.T).T
    return image_rotate, landmarks_rotate


def crop(image, lms):
    lms_top = np.min(lms[:, 1])
    lms_bottom = np.max(lms[:, 1])
    lms_left = np.min(lms[:, 0])
    lms_right = np.max(lms[:, 0])

    # expand bbox
    top = int(lms_top - 0.8 * (lms_bottom - lms_top))
    bottom = int(lms_bottom + 0.3 * (lms_bottom - lms_top))
    left = int(lms_left - 0.3 * (lms_right - lms_left))
    right = int(lms_right + 0.3 * (lms_right - lms_left))

    if bottom - top > right - left:
        left -= ((bottom - top) - (right - left)) // 2
        right = left + (bottom - top)
    else:
        top -= ((right - left) - (bottom - top)) // 2
        bottom = top + (right - left)

    image_crop = np.ones((bottom - top + 1, right - left + 1, 3), np.uint8) * 255

    h, w = image.shape[:2]
    left_white = max(0, -left)
    left = max(0, left)
    right = min(right, w - 1)
    right_white = left_white + (right - left)
    top_white = max(0, -top)
    top = max(0, top)
    bottom = min(bottom, h - 1)
    bottom_white = top_white + (bottom - top)

    image_crop[top_white:bottom_white+1, left_white:right_white+1] = image[top:bottom+1, left:right+1].copy()
    return image_crop
