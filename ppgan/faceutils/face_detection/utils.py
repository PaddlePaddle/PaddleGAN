import time
import paddle
import math
import numpy as np
import cv2


def _gaussian(size=3,
              sigma=0.25,
              amplitude=1,
              normalize=False,
              width=None,
              height=None,
              sigma_horz=None,
              sigma_vert=None,
              mean_horz=0.5,
              mean_vert=0.5):
    # handle some defaults
    if width is None:
        width = size
    if height is None:
        height = size
    if sigma_horz is None:
        sigma_horz = sigma
    if sigma_vert is None:
        sigma_vert = sigma
    center_x = mean_horz * width + 0.5
    center_y = mean_vert * height + 0.5
    gauss = np.empty((height, width), dtype=np.float32)
    # generate kernel
    for i in range(height):
        for j in range(width):
            gauss[i][j] = amplitude * math.exp(-(math.pow(
                (j + 1 - center_x) / (sigma_horz * width), 2) / 2.0 + math.pow(
                    (i + 1 - center_y) / (sigma_vert * height), 2) / 2.0))
    if normalize:
        gauss = gauss / np.sum(gauss)
    return gauss


def transform(point, center, scale, resolution, invert=False):
    """Generate and affine transformation matrix.

    Given a set of points, a center, a scale and a targer resolution, the
    function generates and affine transformation matrix. If invert is ``True``
    it will produce the inverse transformation.

    Args:
        point {paddle.tensor} -- the input 2D point
        center {paddle.tensor or numpy.array} -- the center around which to perform the transformations
        scale {float} -- the scale of the face/object
        resolution {float} -- the output resolution
        invert {bool} -- define wherever the function should produce the direct or the
        inverse transformation matrix (default: {False})
    """
    _pt = paddle.ones([3])
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = paddle.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = paddle.inverse(t)

    new_point = (paddle.matmul(t, _pt))[0:2]

    return new_point.astype('int32')


def crop(image, center, scale, resolution=256.0):
    """Center crops an image or set of heatmaps

    Args:
        image {numpy.array} -- an rgb image
        center {numpy.array} -- the center of the object, usually the same as of the bounding box
        scale {float} -- scale of the face
        resolution {float} -- the size of the output cropped image (default: {256.0})

    """
    """ Crops the image around the center. Input is expected to be an np.ndarray """
    ul = transform([1, 1], center, scale, resolution, True)
    ul = ul.numpy()
    br = transform([resolution, resolution], center, scale, resolution, True)
    br = br.numpy()
    if image.ndim > 2:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0], image.shape[2]],
                          dtype=np.int32)
        newImg = np.zeros(newDim, dtype=np.uint8)
    else:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
        newImg = np.zeros(newDim, dtype=np.uint8)
    ht = image.shape[0]
    wd = image.shape[1]
    newX = np.array(
        [max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
    newY = np.array(
        [max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
    oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
    oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)
    newImg[newY[0] - 1:newY[1],
           newX[0] - 1:newX[1]] = image[oldY[0] - 1:oldY[1],
                                        oldX[0] - 1:oldX[1], :]
    newImg = cv2.resize(newImg,
                        dsize=(int(resolution), int(resolution)),
                        interpolation=cv2.INTER_LINEAR)
    return newImg


def shuffle_lr(parts, pairs=None):
    """Shuffle the points left-right according to the axis of symmetry
    of the object.

    Args:
        parts {paddle.tensor} -- a 3D or 4D object containing the
        heatmaps.
        pairs {list of integers} -- [order of the flipped points] (default: {None})
    """
    if pairs is None:
        pairs = [
            16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 26, 25,
            24, 23, 22, 21, 20, 19, 18, 17, 27, 28, 29, 30, 35, 34, 33, 32, 31,
            45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41, 40, 54, 53, 52, 51, 50,
            49, 48, 59, 58, 57, 56, 55, 64, 63, 62, 61, 60, 67, 66, 65
        ]
    if parts.dim == 3:
        parts = parts.gather(paddle.to_tensor(pairs))
    else:
        parts = paddle.to_tensor(parts.numpy()[:, pairs, ...])

    return parts
