import cv2
import numpy as np


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def resize_and_crop_image(image, dim):
    if image.shape[0] > image.shape[1]:
        img = image_resize(image, width=dim)
        yshift, xshift = (image.shape[0] - image.shape[1]) // 2, 0
        y_start = (img.shape[0] - img.shape[1]) // 2
        y_end = y_start + dim
        return img[y_start:y_end, :, :], (xshift, yshift)
    else:
        img = image_resize(image, height=dim)
        yshift, xshift = 0, (image.shape[1] - image.shape[0]) // 2
        x_start = (img.shape[1] - img.shape[0]) // 2
        x_end = x_start + dim
        return img[:, x_start:x_end, :], (xshift, yshift)


def resize_and_crop_batch(frames, dim):
    """
    Center crop + resize to (dim x dim)
    inputs:
        - frames: list of images (numpy arrays)
        - dim: output dimension size
    """
    smframes = []
    xshift, yshift = 0, 0
    for i in range(len(frames)):
        smframe, (xshift, yshift) = resize_and_crop_image(frames[i], dim)
        smframes.append(smframe)
    smframes = np.stack(smframes)
    return smframes, (xshift, yshift)
