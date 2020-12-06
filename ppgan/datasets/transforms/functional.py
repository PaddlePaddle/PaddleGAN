#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
from __future__ import division

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
