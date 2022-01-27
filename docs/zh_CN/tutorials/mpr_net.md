# MPR_Net

## 1 原理介绍

[MPR_Net](https://arxiv.org/abs/2102.02808)是发表在CVPR2021的一篇图像修复方法。图像修复任务需要在恢复图像时在空间细节和高级上下文信息之间实现复杂的平衡。MPR_Net提出了一种新颖的协同设计，可以最佳地平衡这些相互竞争的目标。其中主要提议是一个多阶段架构，它逐步学习退化输入的恢复函数，从而将整个恢复过程分解为更易于管理的步骤。具体来说，MPR_Net首先使用编码器-解码器架构学习上下文特征，然后将它们与保留本地信息的高分辨率分支相结合。在每个阶段引入了一种新颖的每像素自适应设计，利用原位监督注意力来重新加权局部特征。这种多阶段架构的一个关键要素是不同阶段之间的信息交换。为此，MPR_Net提出了一种双向方法，其中信息不仅从早期到后期按顺序交换，而且特征处理块之间也存在横向连接以避免任何信息丢失。由此产生的紧密互连的多阶段架构，称为MPRNet，在包括图像去雨、去模糊和去噪在内的一系列任务中，在十个数据集上提供了强大的性能提升。

## 2 如何使用

### 2.1 快速体验

安装`PaddleGAN`之后运行如下代码即生成修复后的图像`output_dir/Deblurring/image_name.png`，其中`task`为你想要修复的任务，可以在`Deblurring`、`Denoising`和`Deraining`中选择，`PATH_OF_IMAGE`为你需要转换的图像路径。

```python
from ppgan.apps import MPRPredictor
predictor = MPRPredictor(task='Deblurring')
predictor.run(PATH_OF_IMAGE)
```

或者在终端中运行如下命令，也可获得相同结果：

```sh
python applications/tools/mprnet.py --input_image ${PATH_OF_IMAGE} --task Deblurring
```
其中`task`为你想要修复的任务，可以在`Deblurring`、`Denoising`和`Deraining`中选择，`PATH_OF_IMAGE`为你需要转换的图像路径。

### 2.1 数据准备

Deblurring训练数据是GoPro，用于去模糊的GoPro数据集由3214张1,280×720大小的模糊图像组成，这些图像分为2103张训练图像和1111张测试图像。可以从[这里](https://drive.google.com/file/d/1H0PIXvJH4c40pk7ou6nAwoxuR4Qh_Sa2/view?usp=sharing)下载。
下载后解压到data目录下，解压完成后数据分布如下所示：

```sh
GoPro
├── train
│   ├── input
│   └── target
└── test
    ├── input
    └── target

```

Denoising训练数据是SIDD，一个图像去噪数据集，包含来自10个不同光照条件下的3万幅噪声图像，可以从[训练数据集下载](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php)和[测试数据集下载](https://drive.google.com/drive/folders/1S44fHXaVxAYW3KLNxK41NYCnyX9S79su)下载。
下载后解压到data目录下，解压完成后数据分布如下所示：

```sh
SIDD
├── train
│   ├── input
│   └── target
└── val
    ├── input
    └── target

```

Deraining训练数据是Synthetic Rain Datasets，由13712张从多个数据集(Rain14000, Rain1800, Rain800, Rain12)收集的干净雨图像对组成，可以从[训练数据集下载](https://drive.google.com/drive/folders/1Hnnlc5kI0v9_BtfMytC2LR5VpLAFZtVe)和[测试数据集下载](https://drive.google.com/drive/folders/1PDWggNh8ylevFmrjo-JEvlmqsDlWWvZs)下载。
下载后解压到data目录下，解压完成后数据分布如下所示：

```sh
Synthetic_Rain_Datasets
├── train
│   ├── input
│   └── target
└── test
    ├── Test100
    ├── Rain100H
    ├── Rain100L
    ├── Test1200
    └── Test2800

```

### 2.2 训练
  示例以训练Deblurring的数据为例。如果想训练其他任务可以通过替换配置文件。

  ```sh
  python -u tools/main.py --config-file configs/mprnet_deblurring.yaml
  ```

### 2.3 测试

测试模型：
```sh
python tools/main.py --config-file configs/mprnet_deblurring.yaml --evaluate-only --load ${PATH_OF_WEIGHT}
```

## 3 结果展示

去模糊
| 模型 | 数据集 | PSNR/SSIM |
|---|---|---|
| MPRNet | GoPro | 33.4360/0.9410 |

去噪
| 模型 | 数据集 | PSNR/SSIM |
|---|---|---|
| MPRNet | SIDD |  43.6100 / 0.9586 |

去雨
| 模型 | 数据集 | PSNR/SSIM |
|---|---|---|
| MPRNet | Rain100L | 36.2848 / 0.9651 |


## 4 模型下载

| 模型 | 下载地址 |
|---|---|
| MPR_Deblurring | [MPR_Deblurring](https://paddlegan.bj.bcebos.com/models/MPR_Deblurring.pdparams) |
| MPR_Denoising | [MPR_Denoising](https://paddlegan.bj.bcebos.com/models/MPR_Denoising.pdparams) |
| MPR_Deraining | [MPR_Deraining](https://paddlegan.bj.bcebos.com/models/MPR_Deraining.pdparams) |


# 参考文献

- [Multi-Stage Progressive Image Restoration](https://arxiv.org/abs/2102.02808)

  ```
  @inproceedings{Kim2020U-GAT-IT:,
    title={Multi-Stage Progressive Image Restoration},
    author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat and Fahad Shahbaz Khan and Ming-Hsuan Yang and Ling Shao},
    booktitle={CVPR},
    year={2021}
  }
  ```
