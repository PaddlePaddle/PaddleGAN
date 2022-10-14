English | [Chinese](../../zh_CN/tutorials/invdn.md)

# Invertible Denoising Network: A Light Solution for Real Noise Removal

**Invertible Denoising Network: A Light Solution for Real Noise Removal** (CVPR 2021)

Official code：[https://github.com/Yang-Liu1082/InvDN](https://github.com/Yang-Liu1082/InvDN)

Paper：[https://arxiv.org/abs/2104.10546](https://arxiv.org/abs/2104.10546)

## 1、Introduction

InvDN uses invertible network to divide noise image into low resolution clean image and high frequency latent representation, which contains noise information and content information. Since the invertible network is information lossless, if we can separate the noise information in the high-frequency representation, then we can reconstruct the clean picture with the original resolution together with the clean picture with the low resolution. However, it is difficult to remove the noise in the high-frequency information. In this paper, the high-frequency latent representation with noise is directly replaced by another representation sampled from the prior distribution in the process of reduction, and then the low-resolution clean image is reconstructed back to the original resolution clean image. The network implemented in this paper is lightweight.

![invdn](https://user-images.githubusercontent.com/51016595/195344773-9ea17ef5-9edd-4310-bfff-36049bbcefde.png)


## 2 How to use

### 2.1 Quick start

After installing `PaddleGAN`, you can run a command as follows to generate the restorated image `./output_dir/Denoising/image_name.png`.

```sh
python applications/tools/invdn_denoising.py --images_path ${PATH_OF_IMAGE}
```

Where `PATH_OF_IMAGE` is the path of the image you need to denoise, or the path of the folder where the images is located.

- Note that in the author's original code, Monte Carlo self-ensemble is used for testing to improve performance, but it slows things down. Users are free to choose whether to use the `--disable_mc` parameter to turn off Monte Carlo self-ensemble for faster speeds. (Monte-carlo self-ensemble is enabled by default for $test$, and disabled by default for $train$ and $valid$.)

### 2.2 Prepare dataset

#### **Train Dataset**

In this paper, we will use SIDD, including training dataset [SIDD-Medium](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php). According to the requirements of the paper, it is necessary to process the dataset into patches of $512 \times 512$. In addition, this paper needs to produce a low-resolution version of the GT image with a size of $128 \times 128$ during training. The low-resolution image is denoted as LQ.

The processed dataset can be find in [Ai Studio](https://aistudio.baidu.com/aistudio/datasetdetail/172084).

The train dataset is placed under: `data/SIDD_Medium_Srgb_Patches_512/train/`.

#### **Test Dataset**

The test dataset is [SIDD_valid](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php). The dataset downloaded from the official website is `./ValidationNoisyBlocksSrgb.mat and ./ValidationGtBlocksSrgb.mat`. You are advised to convert it to $.png$ for convenience.

The converted dataset can be find in [Ai Studio](https://aistudio.baidu.com/aistudio/datasetdetail/172069).

The test dataset is placed under：`data/SIDD_Valid_Srgb_Patches_256/valid/`.

- The file structure under the `PaddleGAN/data` folder is
```sh
data
├─ SIDD_Medium_Srgb_Patches_512
│  └─ train
│      ├─ GT
│      │      0_0.PNG
│      │      ...
│      ├─ LQ
│      │      0_0.PNG
│      │      ...
│      └─ Noisy
│              0_0.PNG
│              ...
│
└─ SIDD_Valid_Srgb_Patches_256
    └─ valid
        ├─ GT
        │      0_0.PNG
        │      ...
        └─ Noisy
                0_0.PNG
                ...
```

### 2.3 Training

Run the following command to start training:
```sh
python -u tools/main.py --config-file configs/invdn_denoising.yaml
```
- TIPS：
In order to ensure that the total $epoch$ number is the same as in the paper configuration, we need to ensure that $total\_batchsize*iter == 1gpus*14bs*600000iters$. Also make sure that $batchsize/learning\_rate == 14/0.0002$ when $batchsize$ is changed.
For example, when using 4 GPUs, set $batchsize$ as 14, then the actual total $batchsize$ should be 14*4, and the total $iters$ needed to be set as 150,000, and the learning rate should be expanded to 8e-4.

### 2.4 Test

Run the following command to start testing:
```sh
python tools/main.py --config-file configs/invdn_denoising.yaml --evaluate-only --load ${PATH_OF_WEIGHT}
```

## 3 Results

Denoising
| model | dataset | PSNR/SSIM |
|---|---|---|
| InvDN | SIDD |  39.29 / 0.956 |


## 4 Download

| model | link |
|---|---|
| InvDN| [InvDN_Denoising](https://paddlegan.bj.bcebos.com/models/InvDN_Denoising.pdparams) |



# References

- [https://arxiv.org/abs/2104.10546](https://arxiv.org/abs/2104.10546)

```
@article{liu2021invertible,
  title={Invertible Denoising Network: A Light Solution for Real Noise Removal},
  author={Liu, Yang and Qin, Zhenyue and Anwar, Saeed and Ji, Pan and Kim, Dongwoo and Caldwell, Sabrina and Gedeon, Tom},
  journal={arXiv preprint arXiv:2104.10546},
  year={2021}
}
```
