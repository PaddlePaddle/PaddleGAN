# StarGAN V2

## 1 Introduction

  [StarGAN V2](https://arxiv.org/pdf/1912.01865.pdf)is an image-to-image translation model published on CVPR2020.
  A good image-to-image translation model should learn a mapping between different visual domains while satisfying the following properties: 1) diversity of generated images and 2) scalability over multiple domains. Existing methods address either of the issues, having limited diversity or multiple models for all domains. StarGAN v2 is a single framework that tackles both and shows significantly improved results over the baselines. Experiments on CelebA-HQ and a new animal faces dataset (AFHQ) validate superiority of StarGAN v2 in terms of visual quality, diversity, and scalability.

## 2 How to use

### 2.1 Prepare dataset

  The CelebAHQ dataset used by StarGAN V2 can be downloaded from [here](https://www.dropbox.com/s/f7pvjij2xlpff59/celeba_hq.zip?dl=0), and the AFHQ dataset can be downloaded from [here](https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=0). Then unzip dataset to the ``PaddleGAN/data`` directory.

  The structure of dataset is as following:

  ```
    ├── data
        ├── afhq
        |   ├── train
        |   |   ├── cat
        |   |   ├── dog
        |   |   └── wild
        |   └── val
        |       ├── cat
        |       ├── dog
        |       └── wild
        └── celeba_hq
            ├── train
            |   ├── female
            |   └── male
            └── val
                ├── female
                └── male

  ```

### 2.2 Train/Test

  The example uses the AFHQ dataset as an example. If you want to use the CelebAHQ dataset, you can change the config file.

  train model:
  ```
     python -u tools/main.py --config-file configs/starganv2_afhq.yaml
  ```

  test model:
  ```
     python tools/main.py --config-file configs/starganv2_afhq.yaml --evaluate-only --load ${PATH_OF_WEIGHT}
  ```

## 3 Results

![](https://user-images.githubusercontent.com/79366697/146308440-65259d70-d056-43d4-8cf5-a82530993910.jpg)

## 4 Model Download
| 模型 | 数据集 | 下载地址 |
|---|---|---|
| starganv2_afhq  | AFHQ | [starganv2_afhq](https://paddlegan.bj.bcebos.com/models/starganv2_afhq.pdparams)




# References

- 1. [StarGAN v2: Diverse Image Synthesis for Multiple Domains](https://arxiv.org/abs/1912.01865)

  ```
  @inproceedings{choi2020starganv2,
  title={StarGAN v2: Diverse Image Synthesis for Multiple Domains},
  author={Yunjey Choi and Youngjung Uh and Jaejun Yoo and Jung-Woo Ha},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
  }
  ```
