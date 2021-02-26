#Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import os
import fnmatch
import numpy as np
import cv2
import paddle
from PIL import Image
from cv2 import imread
from scipy import linalg
from inception import InceptionV3

try:
    from tqdm import tqdm
except:

    def tqdm(x):
        return x


""" based on https://github.com/mit-han-lab/gan-compression/blob/master/metric/fid_score.py
"""
"""
inceptionV3 pretrain model is convert from pytorch, pretrain_model url is https://paddle-gan-models.bj.bcebos.com/params_inceptionV3.tar.gz
"""


def _calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    m1 = np.atleast_1d(mu1)
    m2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    t = sigma1.dot(sigma2)
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) -
            2 * tr_covmean)


def _get_activations_from_ims(img, model, batch_size, dims, use_gpu,
                              premodel_path):
    n_batches = (len(img) + batch_size - 1) // batch_size
    n_used_img = len(img)

    pred_arr = np.empty((n_used_img, dims))

    for i in tqdm(range(n_batches)):
        start = i * batch_size
        end = start + batch_size
        if end > len(img):
            end = len(img)
        images = img[start:end]
        if images.shape[1] != 3:
            images = images.transpose((0, 3, 1, 2))
        images /= 255

        images = paddle.to_tensor(images)
        param_dict, _ = paddle.load(premodel_path)
        model.set_dict(param_dict)
        model.eval()
        pred = model(images)[0][0]
        pred_arr[start:end] = pred.reshape(end - start, -1)

    return pred_arr


def _compute_statistic_of_img(img, model, batch_size, dims, use_gpu,
                              premodel_path):
    act = _get_activations_from_ims(img, model, batch_size, dims, use_gpu,
                                    premodel_path)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_fid_given_img(img_fake,
                            img_real,
                            batch_size,
                            use_gpu,
                            dims,
                            premodel_path,
                            model=None):
    assert os.path.exists(
        premodel_path
    ), 'pretrain_model path {} is not exists! Please download it first'.format(
        premodel_path)
    if model is None:
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx])

    m1, s1 = _compute_statistic_of_img(img_fake, model, batch_size, dims,
                                       use_gpu, premodel_path)
    m2, s2 = _compute_statistic_of_img(img_real, model, batch_size, dims,
                                       use_gpu, premodel_path)

    fid_value = _calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value


def _get_activations(files,
                     model,
                     batch_size,
                     dims,
                     use_gpu,
                     premodel_path,
                     style=None):
    if len(files) % batch_size != 0:
        print(('Warning: number of images is not a multiple of the '
               'batch size. Some samples are going to be ignored.'))
    if batch_size > len(files):
        print(('Warning: batch size is bigger than the datasets size. '
               'Setting batch size to datasets size'))
        batch_size = len(files)

    n_batches = len(files) // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))
    for i in tqdm(range(n_batches)):
        start = i * batch_size
        end = start + batch_size

        # same as stargan-v2 official implementation: resize to 256 first, then resize to 299
        if style == 'stargan':
            img_list = []
            for f in files[start:end]:
                im = Image.open(str(f)).convert('RGB')
                if im.size[0] != 299:
                    im = im.resize((256, 256), 2)
                    im = im.resize((299, 299), 2)

                img_list.append(np.array(im).astype('float32'))

            images = np.array(img_list)
        else:
            images = np.array(
                [imread(str(f)).astype(np.float32) for f in files[start:end]])

        if len(images.shape) != 4:
            images = imread(str(files[start]))
            images = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
            images = np.array([images.astype(np.float32)])

        images = images.transpose((0, 3, 1, 2))
        images /= 255

        # imagenet normalization
        if style == 'stargan':
            mean = np.array([0.485, 0.456, 0.406]).astype('float32')
            std = np.array([0.229, 0.224, 0.225]).astype('float32')
            images[:] = (images[:] - mean[:, None, None]) / std[:, None, None]

        if style == 'stargan':
            pred_arr[start:end] = inception_infer(images, premodel_path)
        else:
            with paddle.guard():
                images = paddle.to_tensor(images)
                param_dict, _ = paddle.load(premodel_path)
                model.set_dict(param_dict)
                model.eval()

                pred = model(images)[0][0].numpy()

                pred_arr[start:end] = pred.reshape(end - start, -1)

    return pred_arr


def inception_infer(x, model_path):
    exe = paddle.static.Executor()
    [inference_program, feed_target_names,
     fetch_targets] = paddle.static.load_inference_model(model_path, exe)
    results = exe.run(inference_program,
                      feed={feed_target_names[0]: x},
                      fetch_list=fetch_targets)
    return results[0]


def _calculate_activation_statistics(files,
                                     model,
                                     premodel_path,
                                     batch_size=50,
                                     dims=2048,
                                     use_gpu=False,
                                     style=None):
    act = _get_activations(files, model, batch_size, dims, use_gpu,
                           premodel_path, style)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _compute_statistics_of_path(path,
                                model,
                                batch_size,
                                dims,
                                use_gpu,
                                premodel_path,
                                style=None):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        files = []
        for root, dirnames, filenames in os.walk(path):
            for filename in fnmatch.filter(
                    filenames, '*.jpg') or fnmatch.filter(filenames, '*.png'):
                files.append(os.path.join(root, filename))
        m, s = _calculate_activation_statistics(files, model, premodel_path,
                                                batch_size, dims, use_gpu,
                                                style)
    return m, s


def calculate_fid_given_paths(paths,
                              premodel_path,
                              batch_size,
                              use_gpu,
                              dims,
                              model=None,
                              style=None):
    assert os.path.exists(
        premodel_path
    ), 'pretrain_model path {} is not exists! Please download it first'.format(
        premodel_path)
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    if model is None and style != 'stargan':
        with paddle.guard():
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
            model = InceptionV3([block_idx], class_dim=1008)

    m1, s1 = _compute_statistics_of_path(paths[0], model, batch_size, dims,
                                         use_gpu, premodel_path, style)
    m2, s2 = _compute_statistics_of_path(paths[1], model, batch_size, dims,
                                         use_gpu, premodel_path, style)

    fid_value = _calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value
