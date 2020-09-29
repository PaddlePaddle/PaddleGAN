import sys
import types
import random
import numbers
import warnings
import traceback
import collections
import numpy as np

from paddle.utils import try_import
import paddle.vision.transforms.functional as F
import paddle.vision.transforms.transforms as T

from .builder import TRANSFORMS

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


class Transform():
    def _set_attributes(self, args):
        """
        Set attributes from the input list of parameters.

        Args:
            args (list): list of parameters.
        """
        if args:
            for k, v in args.items():
                # print(k, v)
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    def apply_image(self, input):
        raise NotImplementedError

    def __call__(self, inputs):
        # print('debug:', type(inputs), type(inputs[0]))
        if isinstance(inputs, tuple):
            inputs = list(inputs)
        if self.keys is not None:
            for i, key in enumerate(self.keys):
                if isinstance(inputs, dict):
                    inputs[key] = getattr(self, 'apply_' + key)(inputs[key])
                elif isinstance(inputs, (list, tuple)):
                    inputs[i] = getattr(self, 'apply_' + key)(inputs[i])
        else:
            inputs = self.apply_image(inputs)

        if isinstance(inputs, list):
            inputs = tuple(inputs)

        return inputs


@TRANSFORMS.register()
class Resize(Transform):
    """Resize the input Image to the given size.

    Args:
        size (int|list|tuple): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Interpolation mode of resize. Default: 1.
            0 : cv2.INTER_NEAREST
            1 : cv2.INTER_LINEAR
            2 : cv2.INTER_CUBIC
            3 : cv2.INTER_AREA
            4 : cv2.INTER_LANCZOS4
            5 : cv2.INTER_LINEAR_EXACT
            7 : cv2.INTER_MAX
            8 : cv2.WARP_FILL_OUTLIERS
            16: cv2.WARP_INVERSE_MAP

    """
    def __init__(self, size, interpolation=1, keys=None):
        super().__init__()
        assert isinstance(size, int) or (isinstance(size, Iterable)
                                         and len(size) == 2)
        self._set_attributes(locals())
        if isinstance(self.size, Iterable):
            self.size = tuple(size)

    def apply_image(self, img):
        return F.resize(img, self.size, self.interpolation)


@TRANSFORMS.register()
class RandomCrop(Transform):
    def __init__(self, output_size, keys=None):
        super().__init__()
        self._set_attributes(locals())
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def _get_params(self, img):
        h, w, _ = img.shape
        th, tw = self.output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def apply_image(self, img):
        i, j, h, w = self._get_params(img)
        cropped_img = img[i:i + h, j:j + w]
        return cropped_img


@TRANSFORMS.register()
class PairedRandomCrop(RandomCrop):
    def __init__(self, output_size, keys=None):
        super().__init__(output_size, keys)

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def apply_image(self, img, crop_prams=None):
        if crop_prams is not None:
            i, j, h, w = crop_prams
        else:
            i, j, h, w = self._get_params(img)
        cropped_img = img[i:i + h, j:j + w]
        return cropped_img

    def __call__(self, inputs):
        if isinstance(inputs, tuple):
            inputs = list(inputs)
        if self.keys is not None:
            if isinstance(inputs, dict):
                crop_params = self._get_params(inputs[self.keys[0]])
            elif isinstance(inputs, (list, tuple)):
                crop_params = self._get_params(inputs[0])

            for i, key in enumerate(self.keys):
                if isinstance(inputs, dict):
                    inputs[key] = getattr(self, 'apply_' + key)(inputs[key],
                                                                crop_params)
                elif isinstance(inputs, (list, tuple)):
                    inputs[i] = getattr(self, 'apply_' + key)(inputs[i],
                                                              crop_params)
        else:
            crop_params = self._get_params(inputs)
            inputs = self.apply_image(inputs, crop_params)

        if isinstance(inputs, list):
            inputs = tuple(inputs)
        return inputs


@TRANSFORMS.register()
class RandomHorizontalFlip(Transform):
    """Horizontally flip the input data randomly with a given probability.

    Args:
        prob (float): Probability of the input data being flipped. Default: 0.5
    """
    def __init__(self, prob=0.5, keys=None):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img):
        if np.random.random() < self.prob:
            return F.flip(img, code=1)
        return img


# import paddle
# paddle.vision.transforms.RandomHorizontalFlip


@TRANSFORMS.register()
class PairedRandomHorizontalFlip(RandomHorizontalFlip):
    def __init__(self, prob=0.5, keys=None):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img, flip):
        if flip:
            return F.flip(img, code=1)
        return img

    def __call__(self, inputs):
        if isinstance(inputs, tuple):
            inputs = list(inputs)
        flip = np.random.random() < self.prob
        if self.keys is not None:

            for i, key in enumerate(self.keys):
                if isinstance(inputs, dict):
                    inputs[key] = getattr(self, 'apply_' + key)(inputs[key],
                                                                flip)
                elif isinstance(inputs, (list, tuple)):
                    inputs[i] = getattr(self, 'apply_' + key)(inputs[i], flip)
        else:
            inputs = self.apply_image(inputs, flip)

        if isinstance(inputs, list):
            inputs = tuple(inputs)

        return inputs


@TRANSFORMS.register()
class Normalize(Transform):
    """Normalize the input data with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels,
    this transform will normalize each channel of the input data.
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (int|float|list): Sequence of means for each channel.
        std (int|float|list): Sequence of standard deviations for each channel.

    """
    def __init__(self, mean=0.0, std=1.0, keys=None):
        super().__init__()
        self._set_attributes(locals())

        if isinstance(mean, numbers.Number):
            mean = [mean, mean, mean]

        if isinstance(std, numbers.Number):
            std = [std, std, std]

        self.mean = np.array(mean, dtype=np.float32).reshape(len(mean), 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(len(std), 1, 1)

    def apply_image(self, img):
        return (img - self.mean) / self.std


@TRANSFORMS.register()
class Permute(Transform):
    """Change input data to a target mode.
    For example, most transforms use HWC mode image,
    while the Neural Network might use CHW mode input tensor.
    Input image should be HWC mode and an instance of numpy.ndarray.

    Args:
        mode (str): Output mode of input. Default: "CHW".
        to_rgb (bool): Convert 'bgr' image to 'rgb'. Default: True.

    """
    def __init__(self, mode="CHW", to_rgb=True, keys=None):
        super().__init__()
        self._set_attributes(locals())
        assert mode in [
            "CHW"
        ], "Only support 'CHW' mode, but received mode: {}".format(mode)
        self.mode = mode
        self.to_rgb = to_rgb

    def apply_image(self, img):
        if self.to_rgb:
            img = img[..., ::-1]
        if self.mode == "CHW":
            return img.transpose((2, 0, 1))
        return img


# import paddle
# paddle.vision.transforms.Normalize
# TRANSFORMS.register(T.Normalize)


class Crop():
    def __init__(self, pos, size):
        self.pos = pos
        self.size = size

    def __call__(self, img):
        oh, ow, _ = img.shape
        x, y = self.pos
        th = tw = self.size
        if (ow > tw or oh > th):
            return img[y:y + th, x:x + tw]

        return img
