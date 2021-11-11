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

import os
import cv2
import numpy as np
import paddle
from paddle import optimizer as optim
from paddle.nn import functional as F
from paddle.vision import transforms
from tqdm import tqdm
from PIL import Image
from .styleganv2_predictor import StyleGANv2Predictor
from .pixel2style2pixel_predictor import run_alignment
from ..metrics.lpips import LPIPS


def get_lr(t, ts, initial_lr, final_lr):
    alpha = pow(final_lr / initial_lr, 1 / ts)**(t * ts)

    return initial_lr * alpha


def make_image(tensor):
    return (((tensor.detach() + 1) / 2 * 255).clip(min=0, max=255).transpose(
        (0, 2, 3, 1)).numpy().astype('uint8'))


class StyleGANv2FittingPredictor(StyleGANv2Predictor):
    def run(self,
            image,
            need_align=False,
            start_lr=0.1,
            final_lr=0.025,
            latent_level=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            step=100,
            mse_weight=1,
            pre_latent=None):

        if need_align:
            src_img = run_alignment(image)
        else:
            src_img = Image.open(image).convert("RGB")

        generator = self.generator
        generator.train()

        percept = LPIPS(net='vgg')
        # on PaddlePaddle, lpips's default eval mode means no gradients.
        percept.train()

        n_mean_latent = 4096

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.Transpose(),
            transforms.Normalize([127.5, 127.5, 127.5], [127.5, 127.5, 127.5]),
        ])

        imgs = paddle.to_tensor(transform(src_img)).unsqueeze(0)

        if pre_latent is None:
            with paddle.no_grad():
                noise_sample = paddle.randn(
                    (n_mean_latent, generator.style_dim))
                latent_out = generator.style(noise_sample)

                latent_mean = latent_out.mean(0)

            latent_in = latent_mean.detach().clone().unsqueeze(0).tile(
                (imgs.shape[0], 1))
            latent_in = latent_in.unsqueeze(1).tile(
                (1, generator.n_latent, 1)).detach()

        else:
            latent_in = paddle.to_tensor(np.load(pre_latent)).unsqueeze(0)

        var_levels = list(latent_level)
        const_levels = [
            i for i in range(generator.n_latent) if i not in var_levels
        ]
        assert len(var_levels) > 0
        if len(const_levels) > 0:
            latent_fix = latent_in.index_select(paddle.to_tensor(const_levels),
                                                1).detach().clone()
            latent_in = latent_in.index_select(paddle.to_tensor(var_levels),
                                               1).detach().clone()

        latent_in.stop_gradient = False

        optimizer = optim.Adam(parameters=[latent_in], learning_rate=start_lr)

        pbar = tqdm(range(step))

        for i in pbar:
            t = i / step
            lr = get_lr(t, step, start_lr, final_lr)
            optimizer.set_lr(lr)

            if len(const_levels) > 0:
                latent_dict = {}
                for idx, idx2 in enumerate(var_levels):
                    latent_dict[idx2] = latent_in[:, idx:idx + 1]
                for idx, idx2 in enumerate(const_levels):
                    latent_dict[idx2] = (latent_fix[:, idx:idx + 1]).detach()
                latent_list = []
                for idx in range(generator.n_latent):
                    latent_list.append(latent_dict[idx])
                latent_n = paddle.concat(latent_list, 1)
            else:
                latent_n = latent_in

            img_gen, _ = generator([latent_n],
                                   input_is_latent=True,
                                   randomize_noise=False)

            batch, channel, height, width = img_gen.shape

            if height > 256:
                factor = height // 256

                img_gen = img_gen.reshape((batch, channel, height // factor,
                                           factor, width // factor, factor))
                img_gen = img_gen.mean([3, 5])

            p_loss = percept(img_gen, imgs).sum()
            mse_loss = F.mse_loss(img_gen, imgs)
            loss = p_loss + mse_weight * mse_loss

            optimizer.clear_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(
                (f"perceptual: {p_loss.numpy()[0]:.4f}; "
                 f"mse: {mse_loss.numpy()[0]:.4f}; lr: {lr:.4f}"))

        img_gen, _ = generator([latent_n],
                               input_is_latent=True,
                               randomize_noise=False)
        dst_img = make_image(img_gen)[0]
        dst_latent = latent_n.numpy()[0]

        os.makedirs(self.output_path, exist_ok=True)
        save_src_path = os.path.join(self.output_path, 'src.fitting.png')
        cv2.imwrite(save_src_path,
                    cv2.cvtColor(np.asarray(src_img), cv2.COLOR_RGB2BGR))
        save_dst_path = os.path.join(self.output_path, 'dst.fitting.png')
        cv2.imwrite(save_dst_path, cv2.cvtColor(dst_img, cv2.COLOR_RGB2BGR))
        save_npy_path = os.path.join(self.output_path, 'dst.fitting.npy')
        np.save(save_npy_path, dst_latent)

        return np.asarray(src_img), dst_img, dst_latent
