import argparse
from tqdm import tqdm
import paddle
import numpy as np

from ppgan.apps.styleganv2_predictor import StyleGANv2Predictor


def concat_style_paddle(s_lst, n_layers):
    result = [list() for _ in range(n_layers)]
    assert n_layers == len(s_lst[0])
    for i in range(n_layers):
        for s_ in s_lst:
            result[i].append(s_[i])
    for i in range(n_layers):
        result[i] = paddle.concat(result[i])
    return result


def to_np(s_lst):
    for i in range(len(s_lst)):
        s_lst[i] = s_lst[i].numpy()
    return s_lst


def concat_style_np(s_lst, n_layers):
    result = [list() for _ in range(n_layers)]
    assert n_layers == len(s_lst[0])
    for i in range(n_layers):
        for s_ in s_lst:
            result[i].append(s_[i])
    for i in range(n_layers):
        result[i] = np.concatenate(result[i])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='ffhq-config-f')
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    G = StyleGANv2Predictor(model_type=dataset_name).generator
    w_idx_lst = G.w_idx_lst

    with paddle.no_grad():
        # get intermediate latent of 100000 samples
        w_lst = list()
        z = paddle.to_tensor(
            np.random.RandomState(args.seed).randn(
                1000, 100, G.style_dim).astype('float32'))
        #z = paddle.randn([1000, 100, G.style_dim])
        for i in tqdm(range(1000)):  # 100 * 1000 = 100000 # 1000
            # apply truncation_psi=.7 truncation_cutoff=8
            w_lst.append(
                G.get_latents(z[i], truncation=0.7, truncation_cutoff=8))
        #paddle.save(paddle.concat(w_lst[:20]), f'W-{dataset_name}.pdparams')

        s_lst = []
        # get style of first 2000 sample in W
        for i in tqdm(range(20)):  # 2*1000
            s_ = G.style_affine(w_lst[i])
            s_lst.append(s_)
        paddle.save(concat_style_paddle(s_lst, len(w_idx_lst)),
                    f'S-{dataset_name}.pdparams')

        for i in tqdm(range(20)):  # 2*1000
            s_lst[i] = to_np(s_lst[i])

        # get  std, mean of 100000 style samples
        for i in tqdm(range(20, 1000)):  # 100 * 1000
            s_ = G.style_affine(w_lst[i])
            s_lst.append(to_np(s_))
        del w_lst, z, s_, G
        s_lst = concat_style_np(s_lst, len(w_idx_lst))
        s_mean = [
            paddle.mean(paddle.to_tensor(s_lst[i]), axis=0)
            for i in range(len(w_idx_lst))
        ]
        s_std = [
            paddle.std(paddle.to_tensor(s_lst[i]), axis=0)
            for i in range(len(w_idx_lst))
        ]
        paddle.save({
            'mean': s_mean,
            'std': s_std
        }, f'stylegan2-{dataset_name}-styleclip-stats.pdparams')
    print("Done.")
