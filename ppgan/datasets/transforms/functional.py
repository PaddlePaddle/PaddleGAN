from __future__ import division

import sys
import math
import numbers
import warnings
import collections

import numpy as np
from PIL import Image
from numpy import sin, cos, tan
import paddle

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

from . import functional_cv2 as F_cv2
from paddle.vision.transforms.functional import _is_numpy_image, _is_pil_image

__all__ = ['add']


def add(pic, value):
    if not (_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(
            type(pic)))

    if _is_pil_image(pic):
        raise NotImplementedError('add not support pil image')
    else:
        return F_cv2.add(pic, value)
