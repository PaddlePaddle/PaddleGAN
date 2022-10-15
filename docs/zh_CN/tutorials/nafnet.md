[English](../../en_US/tutorials/nafnet.md) | 中文

# NAFNet：图像恢复的简单基线

## 1、简介

NAFNet提出一种超简基线方案Baseline，它不仅计算高效同时性能优于之前SOTA方案；在所得Baseline基础上进一步简化得到了NAFNet：移除了非线性激活单元且性能进一步提升。所提方案在SIDD降噪与GoPro去模糊任务上均达到了新的SOTA性能，同时计算量大幅降低。网络设计和特点如下图所示，采用带跳过连接的UNet作为整体架构，同时修改了Restormer块中的Transformer模块，并取消了激活函数，采取更简单有效的simplegate设计，运用更简单的通道注意力机制

![NAFNet](https://ai-studio-static-online.cdn.bcebos.com/699b87449c7e495f8655ae5ac8bc0eb77bed4d9cd828451e8939ddbc5732a704)

对模型更详细的介绍，可参考论文原文[Simple Baselines for Image Restoration](https://arxiv.org/pdf/2204.04676)，PaddleGAN中目前提供去噪任务的权重

## 2 如何使用

### 2.1 快速体验

安装`PaddleGAN`之后进入`PaddleGAN`文件夹下，运行如下命令即生成修复后的图像`./output_dir/Denoising/image_name.png`

```sh
python applications/tools/nafnet_denoising.py --images_path ${PATH_OF_IMAGE}
```
其中`PATH_OF_IMAGE`为你需要去噪的图像路径，或图像所在文件夹的路径。若需要使用自己的模型权重，则运行如下命令，其中`PATH_OF_MODEL`为模型权重的路径

```sh
python applications/tools/nafnet_denoising.py --images_path ${PATH_OF_IMAGE}  --weight_path ${PATH_OF_MODEL}
```

### 2.2 数据准备

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
用户也可以使用AI studio上的[SIDD数据](https://aistudio.baidu.com/aistudio/datasetdetail/149460)，但需要将文件夹`input_crops`与`gt_crops`重命名为`input`和`target`

### 2.3 训练
示例以训练Denoising的数据为例。如果想训练其他任务可以更换数据集并修改配置文件

```sh
python -u tools/main.py --config-file configs/nafnet_denoising.yaml
```

### 2.4 测试

测试模型：
```sh
python tools/main.py --config-file configs/nafnet_denoising.yaml --evaluate-only --load ${PATH_OF_WEIGHT}
```

## 3 结果展示

去噪
| 模型 | 数据集 | PSNR/SSIM |
|---|---|---|
| NAFNet | SIDD Val |  43.1468 / 0.9563 |

## 4 模型下载

| 模型 | 下载地址 |
|---|---|
| NAFNet| [NAFNet_Denoising](https://paddlegan.bj.bcebos.com/models/NAFNet_Denoising.pdparams) |



# 参考文献

- [Simple Baselines for Image Restoration](https://arxiv.org/pdf/2204.04676)

```
@article{chen_simple_nodate,
	title = {Simple {Baselines} for {Image} {Restoration}},
	abstract = {Although there have been significant advances in the field of image restoration recently, the system complexity of the state-of-the-art (SOTA) methods is increasing as well, which may hinder the convenient analysis and comparison of methods. In this paper, we propose a simple baseline that exceeds the SOTA methods and is computationally efficient. To further simplify the baseline, we reveal that the nonlinear activation functions, e.g. Sigmoid, ReLU, GELU, Softmax, etc. are not necessary: they could be replaced by multiplication or removed. Thus, we derive a Nonlinear Activation Free Network, namely NAFNet, from the baseline. SOTA results are achieved on various challenging benchmarks, e.g. 33.69 dB PSNR on GoPro (for image deblurring), exceeding the previous SOTA 0.38 dB with only 8.4\% of its computational costs; 40.30 dB PSNR on SIDD (for image denoising), exceeding the previous SOTA 0.28 dB with less than half of its computational costs. The code and the pretrained models will be released at github.com/megvii-research/NAFNet.},
	language = {en},
	author = {Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
	pages = {17}
}
```

