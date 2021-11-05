# code was heavily based on https://github.com/wtjiang98/PSGAN
# MIT License 
# Copyright (c) 2020 Wentao Jiang

import cv2
import numpy as np


def generate_P_from_lmks(lmks, resize, w, h):
    """generate P from lmks"""
    diff_size = (64, 64)
    xs, ys = np.meshgrid(np.linspace(0, resize - 1, resize),
                         np.linspace(0, resize - 1, resize))
    xs = xs[None].repeat(68, axis=0)
    ys = ys[None].repeat(68, axis=0)
    fix = np.concatenate([ys, xs], axis=0)

    lmks = lmks.transpose(1, 0).reshape(-1, 1, 1)

    diff = fix - lmks
    diff = diff.transpose(1, 2, 0)
    diff = cv2.resize(diff, diff_size, interpolation=cv2.INTER_NEAREST)
    diff = diff.transpose(2, 0, 1)

    return diff


def copy_area(tar, src, lms):
    rect = [
        int(min(lms[:, 1])) - 16,
        int(min(lms[:, 0])) - 16,
        int(max(lms[:, 1])) + 16 + 1,
        int(max(lms[:, 0])) + 16 + 1
    ]
    tar[rect[1]:rect[3], rect[0]:rect[2]] = \
        src[rect[1]:rect[3], rect[0]:rect[2]]
    src[rect[1]:rect[3], rect[0]:rect[2]] = 0


def rebound_box(mask, mask_B, mask_face):
    """solver ps"""
    index_tmp = mask.nonzero()
    x_index = index_tmp[0]
    y_index = index_tmp[1]
    index_tmp = mask_B.nonzero()
    x_B_index = index_tmp[0]
    y_B_index = index_tmp[1]
    mask_temp = np.copy(mask)
    mask_B_temp = np.copy(mask_B)
    mask_temp[min(x_index) - 16:max(x_index) + 17, min(y_index) - 16:max(y_index) + 17] =\
        mask_face[min(x_index) -
                    16:max(x_index) +
                    17, min(y_index) -
                    16:max(y_index) +
                    17]
    mask_B_temp[min(x_B_index) - 16:max(x_B_index) + 17, min(y_B_index) - 16:max(y_B_index) + 17] =\
        mask_face[min(x_B_index) -
                    16:max(x_B_index) +
                    17, min(y_B_index) -
                    16:max(y_B_index) +
                    17]
    return mask_temp, mask_B_temp


def calculate_consis_mask(mask, mask_B):
    h_a, w_a = mask.shape[1:]
    h_b, w_b = mask_B.shape[1:]
    mask_transpose = np.transpose(mask, (1, 2, 0))
    mask_B_transpose = np.transpose(mask_B, (1, 2, 0))
    mask = cv2.resize(mask_transpose,
                      dsize=(w_a // 4, h_a // 4),
                      interpolation=cv2.INTER_NEAREST)
    mask = np.transpose(mask, (2, 0, 1))
    mask_B = cv2.resize(mask_B_transpose,
                        dsize=(w_b // 4, h_b // 4),
                        interpolation=cv2.INTER_NEAREST)
    mask_B = np.transpose(mask_B, (2, 0, 1))
    """calculate consistency mask between images"""
    h_a, w_a = mask.shape[1:]
    h_b, w_b = mask_B.shape[1:]

    mask_lip = mask[0]
    mask_skin = mask[1]
    mask_eye = mask[2]

    mask_B_lip = mask_B[0]
    mask_B_skin = mask_B[1]
    mask_B_eye = mask_B[2]

    maskA_one_hot = np.zeros((h_a * w_a, 3))
    maskA_one_hot[:, 0] = mask_skin.flatten()
    maskA_one_hot[:, 1] = mask_eye.flatten()
    maskA_one_hot[:, 2] = mask_lip.flatten()

    maskB_one_hot = np.zeros((h_b * w_b, 3))
    maskB_one_hot[:, 0] = mask_B_skin.flatten()
    maskB_one_hot[:, 1] = mask_B_eye.flatten()
    maskB_one_hot[:, 2] = mask_B_lip.flatten()

    con_mask = np.matmul(maskA_one_hot.reshape((h_a * w_a, 3)),
                         np.transpose(maskB_one_hot.reshape((h_b * w_b, 3))))
    con_mask = np.clip(con_mask, 0, 1)
    return con_mask


def cal_hist(image):
    """
        cal cumulative hist for channel list
    """
    hists = []
    for i in range(0, 3):
        channel = image[i]
        hist, _ = np.histogram(channel, bins=256, range=(0, 255))
        sum = hist.sum()
        pdf = [v / sum for v in hist]
        for i in range(1, 256):
            pdf[i] = pdf[i - 1] + pdf[i]
        hists.append(pdf)
    return hists


def cal_trans(ref, adj):
    """
        calculate transfer function
        algorithm refering to wiki item: Histogram matching
    """
    table = list(range(0, 256))
    for i in list(range(1, 256)):
        for j in list(range(1, 256)):
            if ref[i] >= adj[j - 1] and ref[i] <= adj[j]:
                table[i] = j
                break
    table[255] = 255
    return table


def histogram_matching(dstImg, refImg, index):
    """
        perform histogram matching
        dstImg is transformed to have the same the histogram with refImg's
        index[0], index[1]: the index of pixels that need to be transformed in dstImg
        index[2], index[3]: the index of pixels that to compute histogram in refImg
    """
    dst_align = [dstImg[i, index[0], index[1]] for i in range(0, 3)]
    ref_align = [refImg[i, index[2], index[3]] for i in range(0, 3)]
    hist_ref = cal_hist(ref_align)
    hist_dst = cal_hist(dst_align)
    tables = [cal_trans(hist_dst[i], hist_ref[i]) for i in range(0, 3)]

    mid = dst_align.copy()
    for i in range(0, 3):
        for k in range(0, len(index[0])):
            dst_align[i][k] = tables[i][int(mid[i][k])]

    for i in range(0, 3):
        dstImg[i, index[0], index[1]] = dst_align[i]

    return dstImg


def hisMatch(input_data, target_data, mask_src, mask_tar, index):
    """solver ps"""
    mask_src = np.float32(np.clip(mask_src, 0, 1))
    mask_tar = np.float32(np.clip(mask_tar, 0, 1))
    input_masked = np.float32(input_data) * mask_src
    target_masked = np.float32(target_data) * mask_tar
    input_match = histogram_matching(input_masked, target_masked, index)
    return input_match


def mask_preprocess(mask, mask_B):
    """solver ps"""
    index_tmp = mask.nonzero()
    x_index = index_tmp[0]
    y_index = index_tmp[1]
    index_tmp = mask_B.nonzero()
    x_B_index = index_tmp[0]
    y_B_index = index_tmp[1]
    index = [x_index, y_index, x_B_index, y_B_index]
    index_2 = [x_B_index, y_B_index, x_index, y_index]
    return [mask, mask_B, index, index_2]


def generate_mask_aug(mask, lmks):

    lms_eye_left = lmks[42:48]
    lms_eye_right = lmks[36:42]

    mask_eye_left = np.zeros_like(mask)
    mask_eye_right = np.zeros_like(mask)

    mask_face = np.float32(mask == 1) + np.float32(mask == 6)

    copy_area(mask_eye_left, mask_face, lms_eye_left)
    copy_area(mask_eye_right, mask_face, lms_eye_right)

    mask_skin = mask_face

    mask_lip = np.float32(mask == 7) + np.float32(mask == 9)

    mask_eye = mask_eye_left + mask_eye_right
    mask_aug = np.concatenate(
        (np.expand_dims(mask_lip, 0), np.expand_dims(
            mask_skin, 0), np.expand_dims(mask_eye, 0)), 0)

    return mask_aug
