import math
import numpy as np
from PIL import Image

import paddle

# set random seed for reproducibility
np.random.seed(0)


def is_image_file(filename):
    return any(
        filename.endswith(extension)
        for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def gaussian_noise(image, std_dev):
    noise = np.rint(
        np.random.normal(loc=0.0, scale=std_dev, size=np.shape(image)))
    return Image.fromarray(np.clip(image + noise, 0, 255).astype(np.uint8))


#################################################################################
# MATLAB imresize taken from ESRGAN (https://github.com/xinntao/BasicSR)
#################################################################################


def cubic(x):
    absx = paddle.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    temp1 = paddle.cast((absx <= 1), absx.dtype)
    temp2 = paddle.cast((absx > 1), absx.dtype) * paddle.cast(
        (absx <= 2), absx.dtype)
    return (1.5 * absx3 - 2.5 * absx2 +
            1) * temp1 + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * temp2


def calculate_weights_indices(in_length, out_length, scale, kernel,
                              kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = paddle.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = paddle.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.reshape([out_length, 1]).expand([
        out_length, P
    ]) + paddle.linspace(0, P - 1, P).reshape([1, P]).expand([out_length, P])

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.reshape([out_length, 1]).expand([out_length, P
                                                            ]) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = paddle.sum(weights, 1).reshape([out_length, 1])
    weights = weights / weights_sum.expand([out_length, P])

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = np.sum((weights.numpy() == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices[:, 1:1 + P - 2]
        weights = weights[:, 1:1 + P - 2]

    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices[:, 0:P - 2]
        weights = weights[:, 0:P - 2]

    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


def imresize(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: CHW RGB [0,1]
    # output: CHW RGB [0,1] w/o round

    in_C, in_H, in_W = img.shape
    _, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = paddle.zeros([in_C, in_H + sym_len_Hs + sym_len_He, in_W])
    img_aug[:, sym_len_Hs:sym_len_Hs + in_H, :] = img

    sym_patch = img[:, :sym_len_Hs, :]
    inv_idx = paddle.arange(sym_patch.shape[1] - 1, -1, -1)
    sym_patch_inv = paddle.index_select(sym_patch, inv_idx, 1)

    img_aug[:, :sym_len_Hs, :] = sym_patch_inv

    sym_patch = img[:, -sym_len_He:, :]
    inv_idx = paddle.arange(sym_patch.shape[1] - 1, -1, -1)
    sym_patch_inv = paddle.index_select(sym_patch, inv_idx, 1)

    img_aug[:,
            sym_len_Hs + in_H:sym_len_Hs + in_H + sym_len_He, :] = sym_patch_inv

    out_1 = paddle.zeros([in_C, out_H, in_W])
    kernel_width = weights_H.shape[1]
    for i in range(out_H):
        idx = int(indices_H[i][0])

        out_1[0, i, :] = paddle.mv(
            img_aug[0, idx:idx + kernel_width, :].transpose([1, 0]),
            (weights_H[i]))
        out_1[1, i, :] = paddle.mv(
            img_aug[1, idx:idx + kernel_width, :].transpose([1, 0]),
            (weights_H[i]))
        out_1[2, i, :] = paddle.mv(
            img_aug[2, idx:idx + kernel_width, :].transpose([1, 0]),
            (weights_H[i]))

    # process W dimension
    # symmetric copying
    out_1_aug = paddle.zeros([in_C, out_H, in_W + sym_len_Ws + sym_len_We])
    out_1_aug[:, :, sym_len_Ws:sym_len_Ws + in_W] = out_1

    sym_patch = out_1[:, :, :sym_len_Ws]
    inv_idx = paddle.arange(sym_patch.shape[2] - 1, -1, -1)
    sym_patch_inv = paddle.index_select(sym_patch, inv_idx, 2)
    out_1_aug[:, :, 0:sym_len_Ws] = sym_patch_inv

    sym_patch = out_1[:, :, -sym_len_We:]
    inv_idx = paddle.arange(sym_patch.shape[2] - 1, -1, -1)
    sym_patch_inv = paddle.index_select(sym_patch, inv_idx, 2)
    out_1_aug[:, :,
              sym_len_Ws + in_W:sym_len_Ws + in_W + sym_len_We] = sym_patch_inv

    out_2 = paddle.zeros([in_C, out_H, out_W])
    kernel_width = weights_W.shape[1]
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[0, :, i] = out_1_aug[0, :,
                                   idx:idx + kernel_width].mv(weights_W[i])
        out_2[1, :, i] = out_1_aug[1, :,
                                   idx:idx + kernel_width].mv(weights_W[i])
        out_2[2, :, i] = out_1_aug[2, :,
                                   idx:idx + kernel_width].mv(weights_W[i])

    return paddle.clip(out_2, 0, 1)


def to_pil_image(pic, mode=None):
    """Convert a tensor or an ndarray to PIL Image.

    Args:
        pic (paddle.Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).

    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes

    Returns:
        PIL Image: Image converted to PIL Image.
    """
    if not (isinstance(pic, paddle.Tensor) or isinstance(pic, np.ndarray)):
        raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(
            type(pic)))

    elif isinstance(pic, paddle.Tensor):
        if len(pic.shape) not in {2, 3}:
            raise ValueError(
                'pic should be 2/3 dimensional. Got {} dimensions.'.format(
                    pic.ndimension()))

        elif len(pic.shape) == 2:
            # if 2D image, add channel dimension (CHW)
            pic = pic.unsqueeze(0)

    elif isinstance(pic, np.ndarray):
        if pic.ndim not in {2, 3}:
            raise ValueError(
                'pic should be 2/3 dimensional. Got {} dimensions.'.format(
                    pic.ndim))

        elif pic.ndim == 2:
            # if 2D image, add channel dimension (HWC)
            pic = np.expand_dims(pic, 2)

    npimg = pic
    if isinstance(pic, paddle.Tensor) and mode != 'F':
        pic = pic.numpy()

    if pic.dtype == 'float32':
        npimg = np.transpose((pic * 255.).astype('uint8'), (1, 2, 0))

    if not isinstance(npimg, np.ndarray):
        raise TypeError('Input pic must be a torch.Tensor or NumPy ndarray, ' +
                        'not {}'.format(type(npimg)))

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = 'L'
        elif npimg.dtype == np.int16:
            expected_mode = 'I;16'
        elif npimg.dtype == np.int32:
            expected_mode = 'I'
        elif npimg.dtype == np.float32:
            expected_mode = 'F'
        if mode is not None and mode != expected_mode:
            raise ValueError(
                "Incorrect mode ({}) supplied for input type {}. Should be {}".
                format(mode, np.dtype, expected_mode))
        mode = expected_mode

    elif npimg.shape[2] == 2:
        permitted_2_channel_modes = ['LA']
        if mode is not None and mode not in permitted_2_channel_modes:
            raise ValueError("Only modes {} are supported for 2D inputs".format(
                permitted_2_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'LA'

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ['RGBA', 'CMYK', 'RGBX']
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError("Only modes {} are supported for 4D inputs".format(
                permitted_4_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGBA'
    else:
        permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError("Only modes {} are supported for 3D inputs".format(
                permitted_3_channel_modes))
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGB'

    if mode is None:
        raise TypeError('Input type {} is not supported'.format(npimg.dtype))

    return Image.fromarray(npimg, mode=mode)
