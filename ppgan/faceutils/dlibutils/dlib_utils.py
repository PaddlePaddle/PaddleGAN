# code was heavily based on https://github.com/wtjiang98/PSGAN
# MIT License
# Copyright (c) 2020 Wentao Jiang

import os
import os.path as osp

import numpy as np
from PIL import Image
from paddle.utils import try_import
import cv2
from ..image import resize_by_max
from paddle.utils.download import get_weights_path_from_url

LANDMARKS_WEIGHT_URL = 'https://paddlegan.bj.bcebos.com/models/lms.dat'


def detect(image: Image):
    dlib = try_import('dlib')
    image = np.asarray(image)
    h, w = image.shape[:2]
    image = resize_by_max(image, 361)
    actual_h, actual_w = image.shape[:2]
    detector = dlib.get_frontal_face_detector()
    faces_on_small = detector(image, 1)
    faces = dlib.rectangles()
    for face in faces_on_small:
        faces.append(
            dlib.rectangle(int(face.left() / actual_w * w + 0.5),
                           int(face.top() / actual_h * h + 0.5),
                           int(face.right() / actual_w * w + 0.5),
                           int(face.bottom() / actual_h * h + 0.5)))
    return faces


def crop(image: Image, face, up_ratio, down_ratio, width_ratio):
    dlib = try_import('dlib')
    width, height = image.size
    face_height = face.height()
    face_width = face.width()
    delta_up = up_ratio * face_height
    delta_down = down_ratio * face_height
    delta_width = width_ratio * width

    img_left = int(max(0, face.left() - delta_width))
    img_top = int(max(0, face.top() - delta_up))
    img_right = int(min(width, face.right() + delta_width))
    img_bottom = int(min(height, face.bottom() + delta_down))
    image = image.crop((img_left, img_top, img_right, img_bottom))
    face = dlib.rectangle(face.left() - img_left,
                          face.top() - img_top,
                          face.right() - img_left,
                          face.bottom() - img_top)
    face_expand = dlib.rectangle(img_left, img_top, img_right, img_bottom)
    center = face_expand.center()
    width, height = image.size
    crop_left = img_left
    crop_top = img_top
    crop_right = img_right
    crop_bottom = img_bottom
    if width > height:
        left = int(center.x - height / 2)
        right = int(center.x + height / 2)
        if left < 0:
            left, right = 0, height
        elif right > width:
            left, right = width - height, width
        image = image.crop((left, 0, right, height))
        face = dlib.rectangle(face.left() - left, face.top(),
                              face.right() - left, face.bottom())
        crop_left += left
        crop_right = crop_left + height
    elif width < height:
        top = int(center.y - width / 2)
        bottom = int(center.y + width / 2)
        if top < 0:
            top, bottom = 0, width
        elif bottom > height:
            top, bottom = height - width, height
        image = image.crop((0, top, width, bottom))
        face = dlib.rectangle(face.left(),
                              face.top() - top, face.right(),
                              face.bottom() - top)
        crop_top += top
        crop_bottom = crop_top + width
    crop_face = dlib.rectangle(crop_left, crop_top, crop_right, crop_bottom)
    return image, face, crop_face


def crop_by_image_size(image: Image, face):
    dlib = try_import('dlib')
    center = face.center()
    width, height = image.size
    if width > height:
        left = int(center.x - height / 2)
        right = int(center.x + height / 2)
        if left < 0:
            left, right = 0, height
        elif right > width:
            left, right = width - height, width
        image = image.crop((left, 0, right, height))
        face = dlib.rectangle(face.left() - left, face.top(),
                              face.right() - left, face.bottom())
    elif width < height:
        top = int(center.y - width / 2)
        bottom = int(center.y + width / 2)
        if top < 0:
            top, bottom = 0, width
        elif bottom > height:
            top, bottom = height - width, height
        image = image.crop((0, top, width, bottom))
        face = dlib.rectangle(face.left(),
                              face.top() - top, face.right(),
                              face.bottom() - top)
    return image, face


def landmarks(image: Image, face):
    dlib = try_import('dlib')
    weight_path = get_weights_path_from_url(LANDMARKS_WEIGHT_URL)
    predictor = dlib.shape_predictor(weight_path)
    shape = predictor(np.asarray(image), face).parts()
    return np.array([[p.y, p.x] for p in shape])


def crop_from_array(image: np.array, face):
    dlib = try_import('dlib')
    ratio = 0.20 / 0.85  # delta_size / face_size
    height, width = image.shape[:2]
    face_height = face.height()
    face_width = face.width()
    delta_height = ratio * face_height
    delta_width = ratio * width

    img_left = int(max(0, face.left() - delta_width))
    img_top = int(max(0, face.top() - delta_height))
    img_right = int(min(width, face.right() + delta_width))
    img_bottom = int(min(height, face.bottom() + delta_height))
    image = image[img_top:img_bottom, img_left:img_right]
    face = dlib.rectangle(face.left() - img_left,
                          face.top() - img_top,
                          face.right() - img_left,
                          face.bottom() - img_top)
    center = face.center()
    height, width = image.shape[:2]
    if width > height:
        left = int(center.x - height / 2)
        right = int(center.x + height / 2)
        if left < 0:
            left, right = 0, height
        elif right > width:
            left, right = width - height, width
        image = image[0:height, left:right]
        face = dlib.rectangle(face.left() - left, face.top(),
                              face.right() - left, face.bottom())
    elif width < height:
        top = int(center.y - width / 2)
        bottom = int(center.y + width / 2)
        if top < 0:
            top, bottom = 0, width
        elif bottom > height:
            top, bottom = height - width, height
        image = image[top:bottom, 0:width]
        face = dlib.rectangle(face.left(),
                              face.top() - top, face.right(),
                              face.bottom() - top)
    return image, face
