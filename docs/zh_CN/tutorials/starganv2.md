# StarGAN V2

## 1 原理介绍

  [StarGAN V2](https://arxiv.org/pdf/1912.01865.pdf)是发布在CVPR2020上的一个图像转换模型。
  一个好的图像到图像转换模型应该学习不同视觉域之间的映射，同时满足以下属性：1）生成图像的多样性和 2）多个域的可扩展性。 现有方法只解决了其中一个问题，领域的多样性有限或对所有领域用多个模型。 StarGAN V2是一个单一的框架，可以同时解决这两个问题，并在基线上显示出显着改善的结果。 CelebAHQ 和新的动物面孔数据集 (AFHQ) 上的实验验证了StarGAN V2在视觉质量、多样性和可扩展性方面的优势。

## 2 如何使用

### 2.1 数据准备

  StarGAN V2使用的CelebAHQ数据集可以从[这里](https://www.dropbox.com/s/f7pvjij2xlpff59/celeba_hq.zip?dl=0)下载，使用的AFHQ数据集可以从[这里](https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=0)下载。将数据集下载解压后放到``PaddleGAN/data``文件夹下 。

  数据的组成形式为：

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

### 2.2 训练/测试

  示例以AFHQ数据集为例。如果您想使用CelebAHQ数据集，可以在换一下配置文件。

  训练模型:
  ```
     python -u tools/main.py --config-file configs/starganv2_afhq.yaml
  ```

  测试模型:
  ```
     python tools/main.py --config-file configs/starganv2_afhq.yaml --evaluate-only --load ${PATH_OF_WEIGHT}
  ```

## 3 结果展示

![](https://user-images.githubusercontent.com/79366697/146308440-65259d70-d056-43d4-8cf5-a82530993910.jpg)

## 4 模型下载
| 模型 | 数据集 | 下载地址 |
|---|---|---|
| starganv2_afhq  | AFHQ | [starganv2_afhq](https://paddlegan.bj.bcebos.com/models/starganv2_afhq.pdparams)




# 参考文献

- 1. [StarGAN v2: Diverse Image Synthesis for Multiple Domains](https://arxiv.org/abs/1912.01865)

  ```
  @inproceedings{choi2020starganv2,
  title={StarGAN v2: Diverse Image Synthesis for Multiple Domains},
  author={Yunjey Choi and Youngjung Uh and Jaejun Yoo and Jung-Woo Ha},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
  }
  ```
