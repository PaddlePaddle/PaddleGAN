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

import math
import paddle
import numpy as np
from PIL import Image

irange = range


def make_grid(tensor, nrow=8, normalize=False, range=None, scale_each=False):
    """Make a grid of images.
    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
    """
    if not (isinstance(tensor, paddle.Tensor) or
            (isinstance(tensor, list)
             and all(isinstance(t, paddle.Tensor) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(
            type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = paddle.stack(tensor, 0)

    # single image H x W
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    # single image
    if tensor.dim() == 3:
        # if single-channel, convert to 3-channel
        if tensor.shape[0] == 1:
            tensor = paddle.concat([tensor, tensor, tensor], 0)
        tensor = tensor.unsqueeze(0)

    # single-channel images
    if tensor.dim() == 4 and tensor.shape[1] == 1:
        tensor = paddle.concat([tensor, tensor, tensor], 1)

    if normalize is True:
        # avoid modifying tensor in-place
        tensor = tensor.astype(tensor.dtype)
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img[:] = img.clip(min=min, max=max)
            img[:] = (img - min) / (max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            # loop over mini-batch dimension
            for t in tensor:
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.shape[0] == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2]), int(tensor.shape[3])
    num_channels = tensor.shape[1]
    canvas = paddle.zeros((num_channels, height * ymaps, width * xmaps),
                          dtype=tensor.dtype)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            canvas[:, y * height:(y + 1) * height,
                   x * width:(x + 1) * width] = tensor[k]
            k = k + 1
    return canvas


def tensor2img(input_image, min_max=(-1., 1.), image_num=1, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor): the input image tensor array
        image_num (int): the convert iamge numbers
        imtype (type): the desired type of the converted numpy array
    """
    def processing(img, transpose=True):
        """"processing one numpy image.

        Parameters:
            im (tensor): the input image numpy array
        """
        # grayscale to RGB
        if img.shape[0] == 1:
            img = np.tile(img, (3, 1, 1))
        img = img.clip(min_max[0], min_max[1])
        img = (img - min_max[0]) / (min_max[1] - min_max[0])
        if imtype == np.uint8:
            # scaling
            img = img * 255.0
        # tranpose
        img = np.transpose(img, (1, 2, 0)) if transpose else img
        return img

    if not isinstance(input_image, np.ndarray):
        # convert it into a numpy array
        image_numpy = input_image.numpy()
        ndim = image_numpy.ndim
        if ndim == 4:
            image_numpy = image_numpy[0:image_num]
        elif ndim == 3:
            # NOTE for eval mode, need add dim
            image_numpy = np.expand_dims(image_numpy, 0)
            image_num = 1
        else:
            raise ValueError(
                "Image numpy ndim is {} not 3 or 4, Please check data".format(
                    ndim))

        if image_num == 1:
            # for one image, log HWC image
            image_numpy = processing(image_numpy[0])
        else:
            # for more image, log NCHW image
            image_numpy = np.stack(
                [processing(im, transpose=False) for im in image_numpy])

    else:
        # if it is a numpy array, do nothing
        image_numpy = input_image
    image_numpy = image_numpy.round()
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def mask2image(mask: np.array, format="HWC"):
    H, W = mask.shape

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(int(mask.max())):
        color = np.random.rand(1, 1, 3) * 255
        canvas += (mask == i)[:, :, None] * color.astype(np.uint8)
    return canvas
