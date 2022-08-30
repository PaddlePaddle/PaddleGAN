#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

import copy
import os
import cv2
import numpy as np
import paddle
from ppgan.apps.styleganv2_predictor import StyleGANv2Predictor
from ppgan.utils.download import get_path_from_url
from clip import tokenize, load_model

model_cfgs = {
    'ffhq-config-f': {
        'direction_urls':
        'https://paddlegan.bj.bcebos.com/models/stylegan2-ffhq-config-f-styleclip-global-directions.pdparams',
        'stat_urls':
        'https://paddlegan.bj.bcebos.com/models/stylegan2-ffhq-config-f-styleclip-stats.pdparams'
    }
}


def make_image(tensor):
    return (((tensor.detach() + 1) / 2 * 255).clip(min=0, max=255).transpose(
        (0, 2, 3, 1)).numpy().astype('uint8'))


# prompt engineering
prompt_templates = [
    'a bad photo of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'a low resolution photo of a {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a photo of a nice {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a good photo of a {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a dark photo of a {}.',
    'graffiti of the {}.',
]


@paddle.no_grad()
def get_delta_t(neutral, target, model, templates=prompt_templates):
    text_features = []
    for classname in [neutral, target]:
        texts = [template.format(classname)
                 for template in templates]  #format with class
        texts = tokenize(texts)  #tokenize
        class_embeddings = model.encode_text(texts)  #embed with text encoder
        class_embeddings /= class_embeddings.norm(axis=-1, keepdim=True)
        class_embedding = class_embeddings.mean(axis=0)
        class_embedding /= class_embedding.norm()
        text_features.append(class_embedding)
    text_features = paddle.stack(text_features, axis=1).t()

    delta_t = (text_features[1] - text_features[0])
    delta_t = delta_t / delta_t.norm()
    return delta_t


@paddle.no_grad()
def get_ds_from_dt(global_style_direction,
                   delta_t,
                   generator,
                   beta_threshold,
                   relative=False,
                   soft_threshold=False):
    delta_s = global_style_direction @ delta_t
    delta_s_max = delta_s.abs().max()
    print(f'max delta_s is {delta_s_max.item()}')
    if relative: beta_threshold *= delta_s_max
    # apply beta threshold (disentangle)
    select = delta_s.abs() < beta_threshold
    num_channel = paddle.sum(~select).item()
    # threshold in style direction
    delta_s[select] = delta_s[select] * soft_threshold
    delta_s /= delta_s_max  # normalize

    # delta_s -> style dict
    dic = []
    ind = 0
    for layer in range(len(generator.w_idx_lst)):  # 26
        dim = generator.channels_lst[layer]
        if layer in generator.style_layers:
            dic.append(paddle.to_tensor(delta_s[ind:ind + dim]))
            ind += dim
        else:
            dic.append(paddle.zeros([dim]))
    return dic, num_channel


class StyleGANv2ClipPredictor(StyleGANv2Predictor):
    def __init__(self, model_type=None, direction_path=None, stat_path=None, **kwargs):
        super().__init__(model_type=model_type, **kwargs)

        if direction_path is None and model_type is not None:
            assert model_type in model_cfgs, f'There is not any pretrained direction file for {model_type} model.'
            direction_path = get_path_from_url(
                model_cfgs[model_type]['direction_urls'])
        self.fs3 = paddle.load(direction_path)

        self.clip_model, _ = load_model('ViT_B_32', pretrained=True)
        self.manipulator = Manipulator(self.generator, model_type=model_type, stat_path=stat_path)

    def get_delta_s(self,
                    neutral,
                    target,
                    beta_threshold,
                    relative=False,
                    soft_threshold=0):
        # get delta_t in CLIP text space (text directions)
        delta_t = get_delta_t(neutral, target, self.clip_model)
        # get delta_s in global image directions
        delta_s, num_channel = get_ds_from_dt(self.fs3, delta_t, self.generator,
                                              beta_threshold, relative,
                                              soft_threshold)
        print(
            f'{num_channel} channels will be manipulated under the {"relative" if relative else ""} beta threshold {beta_threshold}'
        )
        return delta_s

    @paddle.no_grad()
    def gengrate(self, latent: paddle.Tensor, delta_s, lst_alpha):
        styles = self.generator.style_affine(latent)
        styles = self.manipulator.manipulate(styles, delta_s, lst_alpha)
        # synthesis images from manipulated styles
        img_gen = self.manipulator.synthesis_from_styles(styles)
        return img_gen, styles

    @paddle.no_grad()
    def run(self, latent, neutral, target, offset, beta_threshold=0.8):
        latent = paddle.to_tensor(
            np.load(latent)).unsqueeze(0).astype('float32')
        delta_s = self.get_delta_s(neutral, target, beta_threshold)
        img_gen, styles = self.gengrate(latent, delta_s, [0, offset])
        imgs = make_image(paddle.concat(img_gen))
        src_img = imgs[0]
        dst_img = imgs[1]

        dst_latent = styles[1]
        os.makedirs(self.output_path, exist_ok=True)
        save_src_path = os.path.join(self.output_path, 'src.editing.png')
        cv2.imwrite(save_src_path, cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR))
        save_dst_path = os.path.join(self.output_path, 'dst.editing.png')
        cv2.imwrite(save_dst_path, cv2.cvtColor(dst_img, cv2.COLOR_RGB2BGR))
        save_path = os.path.join(self.output_path, 'dst.editing.pd')
        paddle.save(dst_latent, save_path)
        return src_img, dst_img, dst_latent


@paddle.no_grad()
def extract_global_direction(G,
                             lst_alpha,
                             batchsize=5,
                             num=100,
                             dataset_name='',
                             seed=None):
    from tqdm import tqdm
    import PIL
    """Extract global style direction in 100 images
    """
    assert len(lst_alpha) == 2  #[-5, 5]
    assert num < 200
    #np.random.seed(0)
    # get intermediate latent of n samples
    try:
        S = paddle.load(f'S-{dataset_name}.pdparams')
        S = [S[i][:num] for i in range(len(G.w_idx_lst))]
    except:
        print('No pre-computed S, run tools/styleclip_getf.py first!')
        exit()
    # total channel used: 1024 -> 6048 channels, 256 -> 4928 channels
    print(
        f"total channels to manipulate: {sum([G.channels_lst[i] for i in G.style_layers])}"
    )

    manipulator = Manipulator(G, model_type=dataset_name,
                              stat_path=f'stylegan2-{dataset_name}-styleclip-stats.pdparams')
    model, preprocess = load_model('ViT_B_32', pretrained=True)

    nbatch = int(num / batchsize)
    all_feats = list()
    for layer in G.style_layers:
        print(f'\nStyle manipulation in layer "{layer}"')
        for channel_ind in tqdm(range(G.channels_lst[layer])):
            styles = manipulator.manipulate_one_channel(copy.deepcopy(S), layer,
                                                        channel_ind, lst_alpha,
                                                        num)
            # 2 * num images
            feats = list()
            for img_ind in range(nbatch):  # batch size * nbatch * 2
                start = img_ind * batchsize
                end = img_ind * batchsize + batchsize
                synth_imgs = manipulator.synthesis_from_styles(
                    styles, [start, end])
                synth_imgs = [(synth_img.transpose((0, 2, 3, 1)) * 127.5 +
                               128).clip(0, 255).astype('uint8').numpy()
                              for synth_img in synth_imgs]
                imgs = list()
                for i in range(batchsize):
                    img0 = PIL.Image.fromarray(synth_imgs[0][i])
                    img1 = PIL.Image.fromarray(synth_imgs[1][i])
                    imgs.append(preprocess(img0).unsqueeze(0))
                    imgs.append(preprocess(img1).unsqueeze(0))
                feat = model.encode_image(paddle.concat(imgs))
                feats.append(feat.numpy())
            all_feats.append(np.concatenate(feats).reshape([-1, 2, 512]))
    all_feats = np.stack(all_feats)
    np.save(f'fs-{dataset_name}.npy', all_feats)

    fs = all_feats  #L B 2 512
    fs1 = fs / np.linalg.norm(fs, axis=-1)[:, :, :, None]
    fs2 = fs1[:, :, 1, :] - fs1[:, :, 0, :]  # 5*sigma - (-5)* sigma
    fs3 = fs2 / np.linalg.norm(fs2, axis=-1)[:, :, None]
    fs3 = fs3.mean(axis=1)
    fs3 = fs3 / np.linalg.norm(fs3, axis=-1)[:, None]

    paddle.save(paddle.to_tensor(fs3),
                f'stylegan2-{dataset_name}-styleclip-global-directions.pdparams'
                )  # global style direction


class Manipulator():
    """Manipulator for style editing
    The paper uses 100 image pairs to estimate the mean for alpha(magnitude of the perturbation) [-5, 5]
    """
    def __init__(self, generator, model_type='ffhq-config-f', stat_path=None):
        self.generator = generator

        if stat_path is None and model_type is not None:
            assert model_type in model_cfgs, f'There is not any pretrained stat file for {model_type} model.'
            stat_path = get_path_from_url(
                model_cfgs[model_type]['direction_urls'])
        data = paddle.load(stat_path)
        self.S_mean = data['mean']
        self.S_std = data['std']

    @paddle.no_grad()
    def manipulate(self, styles, delta_s, lst_alpha):
        """Edit style by given delta_style
        - use perturbation (delta s) * (alpha) as a boundary
        """
        styles = [copy.deepcopy(styles) for _ in range(len(lst_alpha))]

        for (alpha, style) in zip(lst_alpha, styles):
            for i in range(len(self.generator.w_idx_lst)):
                style[i] += delta_s[i] * alpha
        return styles

    @paddle.no_grad()
    def manipulate_one_channel(self,
                               styles,
                               layer_ind,
                               channel_ind: int,
                               lst_alpha=[0],
                               num_images=100):
        """Edit style from given layer, channel index
        - use mean value of pre-saved style
        - use perturbation (pre-saved style std) * (alpha) as a boundary
        """
        assert 0 <= channel_ind < styles[layer_ind].shape[1]
        boundary = self.S_std[layer_ind][channel_ind].item()
        # apply self.S_mean value for given layer, channel_ind
        for img_ind in range(num_images):
            styles[layer_ind][img_ind,
                              channel_ind] = self.S_mean[layer_ind][channel_ind]
        styles = [copy.deepcopy(styles) for _ in range(len(lst_alpha))]
        perturbation = (paddle.to_tensor(lst_alpha) * boundary).numpy().tolist()
        # apply one channel manipulation
        for img_ind in range(num_images):
            for edit_ind, delta in enumerate(perturbation):
                styles[edit_ind][layer_ind][img_ind, channel_ind] += delta
        return styles

    @paddle.no_grad()
    def synthesis_from_styles(self, styles, slice=None, randomize_noise=True):
        """Synthesis edited styles from styles, lst_alpha
        """
        imgs = list()
        if slice is not None:
            for style in styles:
                style_ = [list() for _ in range(len(self.generator.w_idx_lst))]
                for i in range(len(self.generator.w_idx_lst)):
                    style_[i] = style[i][slice[0]:slice[1]]
                imgs.append(
                    self.generator.synthesis(style_,
                                             randomize_noise=randomize_noise))
        else:
            for style in styles:
                imgs.append(
                    self.generator.synthesis(style,
                                             randomize_noise=randomize_noise))
        return imgs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('runtype',
                        type=str,
                        default='generate',
                        choices=['generate', 'test', 'extract'])
    parser.add_argument("--latent",
                        type=str,
                        default='output_dir/sample/dst.npy',
                        help="path to first image latent codes")
    parser.add_argument("--neutral",
                        type=str,
                        default=None,
                        help="neutral description")
    parser.add_argument("--target",
                        type=str,
                        default=None,
                        help="neutral description")
    parser.add_argument("--direction_path",
                        type=str,
                        default=None,
                        help="path to latent editing directions")
    parser.add_argument("--stat_path",
                        type=str,
                        default=None,
                        help="path to latent stat files")
    parser.add_argument("--direction_offset",
                        type=float,
                        default=5,
                        help="offset value of edited attribute")
    parser.add_argument("--beta_threshold",
                        type=float,
                        default=0.12,
                        help="beta threshold for channel editing")
    parser.add_argument('--dataset_name', type=str,
                        default='ffhq-config-f')  #'animeface-512')
    args = parser.parse_args()
    runtype = args.runtype
    if runtype in ['test', 'extract']:
        dataset_name = args.dataset_name
        G = StyleGANv2Predictor(model_type=dataset_name).generator
        if runtype == 'test':  # test manipulator
            from ppgan.utils.visual import make_grid, tensor2img, save_image
            num_images = 2
            lst_alpha = [-5, 0, 5]
            layer = 6
            channel_ind = 501
            manipulator = Manipulator(G, model_type=dataset_name, stat_path=args.stat_path)
            styles = manipulator.manipulate_one_channel(layer, channel_ind,
                                                        lst_alpha, num_images)
            imgs = manipulator.synthesis_from_styles(styles)
            print(len(imgs), imgs[0].shape)
            save_image(
                tensor2img(make_grid(paddle.concat(imgs), nrow=num_images)),
                f'sample.png')
        elif runtype == 'extract':  # train: extract global style direction
            batchsize = 10
            num_images = 100
            lst_alpha = [-5, 5]
            extract_global_direction(G,
                                     lst_alpha,
                                     batchsize,
                                     num_images,
                                     dataset_name=dataset_name)
    else:
        predictor = StyleGANv2ClipPredictor(model_type=args.dataset_name,
                                            seed=None,
                                            direction_path=args.direction_path,
                                            stat_path=args.stat_path)
        predictor.run(args.latent, args.neutral, args.target,
                      args.direction_offset, args.beta_threshold)
