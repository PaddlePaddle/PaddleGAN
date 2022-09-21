[English](../../en_US/tutorials/swinir.md) | 中文

## SwinIR 基于Swin Transformer的用于图像恢复的强基线模型


## 1、简介

SwinIR的结构比较简单，如果看过Swin-Transformer的话就没什么难点了。作者引入Swin-T结构应用于低级视觉任务，包括图像超分辨率重建、图像去噪、图像压缩伪影去除。SwinIR网络由一个浅层特征提取模块、深层特征提取模块、重建模块构成。重建模块对不同的任务使用不同的结构。浅层特征提取就是一个3×3的卷积层。深层特征提取是k个RSTB块和一个卷积层加残差连接构成。每个RSTB（Res-Swin-Transformer-Block）由L个STL和一层卷积加残差连接构成。模型的结构如下图所示：

![](https://ai-studio-static-online.cdn.bcebos.com/b550e84915634951af756a545c643c815001be73372248b0b5179fd1652ae003)

对模型更详细的介绍，可参考论文原文[SwinIR: Image Restoration Using Swin Transformer](https://arxiv.org/pdf/2108.10257.pdf)，PaddleGAN中目前提供去噪任务的权重




## 2 如何使用

### 2.1 快速体验

安装`PaddleGAN`之后进入`PaddleGAN`文件夹下，运行如下命令即生成修复后的图像`./output_dir/Denoising/image_name.png`

```sh
python applications/tools/swinir_denoising.py --images_path ${PATH_OF_IMAGE}
```
其中`PATH_OF_IMAGE`为你需要去噪的图像路径，或图像所在文件夹的路径

### 2.2 数据准备

#### 训练数据

[DIV2K](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (800 training images) + [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) + [BSD500](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz) (400 training&testing images) + [WED](http://ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar)(4744 images)

已经整理好的数据：放在了 [Ai Studio](https://aistudio.baidu.com/aistudio/datasetdetail/149405) 里.

训练数据放在：`data/trainsets/trainH` 下

#### 测试数据

测试数据为 CBSD68：放在了 [Ai Studio](https://aistudio.baidu.com/aistudio/datasetdetail/147756) 里.

解压到：`data/triansets/CBSD68`

- 经过处理之后，`PaddleGAN/data`文件夹下的
```sh
trainsets
├── trainH
|   |-- 101085.png
|	|-- 101086.png
|	|-- ......
│   └── 201085.png
└── CBSD68
    ├── 271035.png
    |-- ......
    └── 351093.png
```



### 2.3 训练
示例以训练Denoising的数据为例。如果想训练其他任务可以更换数据集并修改配置文件

```sh
python -u tools/main.py --config-file configs/swinir_denoising.yaml
```

### 2.4 测试

测试模型：
```sh
python tools/main.py --config-file configs/swinir_denoising.yaml --evaluate-only --load ${PATH_OF_WEIGHT}
```

## 3 结果展示

去噪
| 模型 | 数据集 | PSNR/SSIM |
|---|---|---|
| SwinIR | CBSD68 |  36.0819 / 0.9464 |


## 4 模型下载

| 模型 | 下载地址 |
|---|---|
| SwinIR| [SwinIR_Denoising](https://paddlegan.bj.bcebos.com/models/SwinIR_Denoising.pdparams) |



# 参考文献

- [SwinIR: Image Restoration Using Swin Transformer](https://arxiv.org/pdf/2108.10257.pdf)

```
@article{liang2021swinir,
    title={SwinIR: Image Restoration Using Swin Transformer},
    author={Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
    journal={arXiv preprint arXiv:2108.10257},
    year={2021}
}
```
