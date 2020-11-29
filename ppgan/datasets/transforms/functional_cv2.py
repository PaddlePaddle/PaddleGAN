from __future__ import division

import sys
import numbers
import warnings
import collections

import numpy as np
from numpy import sin, cos, tan

import paddle
from paddle.utils import try_import

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


def add(image, value):
    return np.clip(image + value, 0, 255).astype('uint8')
