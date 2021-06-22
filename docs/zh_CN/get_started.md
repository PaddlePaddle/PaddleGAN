# 快速开始

PaddleGAN是飞桨生成对抗网络（GAN）开发套件，提供多种经典前沿网络的高性能复现，应用覆盖图像生成、风格迁移、动作驱动、影像超分及上色等多种领域。

本章节将以CycleGAN模型在Cityscapes数据集上的训练预测作为示例，教大家如何快速上手使用PaddleGAN。

**注意，PaddleGAN中所有的模型配置文件均可在 [./PaddleGAN/configs](https://github.com/PaddlePaddle/PaddleGAN/tree/develop/configs) 中找到。**

## 目录
- [安装](#安装)
- [数据准备](#数据准备)
- [训练](#训练)
  - [单卡训练](#1-单卡训练)
    - [参数](#参数)
    - [可视化训练](#可视化训练)
    - [恢复训练](#恢复训练)
  - [多卡训练](#2-多卡训练)
- [预测](#预测)

## 安装

关于安装配置运行环境，请参考[安装文档](./install.md)完成Paddle及PaddleGAN的安装。

在本演示案例中，假设用户将PaddleGAN的代码克隆并放置在 ’/home/paddle‘ 目录中。用户执行的命令操作均在 ’/home/paddle/PaddleGAN‘ 目录下完成。


## 数据准备

按照[数据准备文档](./data_prepare.md)准备Cityscapes数据集。

- 使用脚本下载Cityscapes数据集到 ~/.cache/ppgan 并软连接到 PaddleGAN/data/ 下：

```
python data/download_cyclegan_data.py --name cityscapes
```

## 训练

### 1. 单卡训练

```
python -u tools/main.py --config-file configs/cyclegan_cityscapes.yaml
```

#### 参数
- `--config-file (str)`: 配置文件的路径。此处用的是CycleGAN在Cityscapes数据集上训练的配置文件。
- 输出的日志，权重，可视化结果会默认保存在`./output_dir`中，可以通过配置文件中的`output_dir`参数修改：

```
output_dir: output_dir
```
<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/119620857-0b2ee200-be38-11eb-83c3-9c5c4c4cbedf.png' width=60%>
</div>


- 保存的文件夹会根据模型名字和时间戳自动生成一个新目录，目录示例如下：

```
output_dir
└── CycleGANModel-2020-10-29-09-21
    ├── epoch_1_checkpoint.pkl
    ├── log.txt
    └── visual_train
        ├── epoch001_fake_A.png
        ├── epoch001_fake_B.png
        ├── epoch001_idt_A.png
        ├── epoch001_idt_B.png
        ├── epoch001_real_A.png
        ├── epoch001_real_B.png
        ├── epoch001_rec_A.png
        ├── epoch001_rec_B.png
        ├── epoch002_fake_A.png
        ├── epoch002_fake_B.png
        ├── epoch002_idt_A.png
        ├── epoch002_idt_B.png
        ├── epoch002_real_A.png
        ├── epoch002_real_B.png
        ├── epoch002_rec_A.png
        └── epoch002_rec_B.png

```

#### 可视化训练

[飞桨VisualDL](https://github.com/PaddlePaddle/VisualDL)是针对深度学习模型开发所打造的可视化分析工具，提供关键指标的实时趋势可视化、样本训练中间过程可视化、网络结构可视化等等，更能直观展示超参与模型效果间关系，辅助实现高效调参。

以下操作请确保您已完成[VisualDL](https://github.com/PaddlePaddle/VisualDL)的安装，安装指南请见[VisualDL安装文档](https://github.com/PaddlePaddle/VisualDL/blob/develop/README_CN.md#%E5%AE%89%E8%A3%85%E6%96%B9%E5%BC%8F)。

**通过在配置文件 cyclegan_cityscapes.yaml 中添加参数`enable_visualdl: true`使用 [飞桨VisualDL](https://github.com/PaddlePaddle/VisualDL)对训练过程产生的指标或生成的图像进行记录，并运行相应命令对训练过程进行实时监控：**

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/119621184-68c32e80-be38-11eb-9830-95429db787cf.png' width=60%>
</div>

如果想要自定义[飞桨VisualDL](https://github.com/PaddlePaddle/VisualDL)可视化内容，可以到 [./PaddleGAN/ppgan/engine/trainer.py](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/ppgan/engine/trainer.py) 中进行修改。

本地启动命令：

```
visualdl --logdir output_dir/CycleGANModel-2020-10-29-09-21/
```
更多启动方式及可视化功能使用指南请见[VisualDL使用指南](https://github.com/PaddlePaddle/VisualDL/blob/develop/docs/components/README_CN.md)。

#### 恢复训练

在训练过程中默认会**保存上一个epoch的checkpoint在`output_dir`中，方便恢复训练。**

本次示例中，cyclegan的训练默认**每五个epoch会保存checkpoint**，如需更改，可以到**config文件中的`interval`**进行修改。

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/120147997-2758c780-c21a-11eb-9cf1-4288dbc01d22.png' width=60%>
</div>

```
python -u tools/main.py --config-file configs/cyclegan_cityscapes.yaml --resume your_checkpoint_path
```
- `--resume (str)`: 用来恢复训练的checkpoint路径（保存于上面配置文件中设置的output所在路径）。

### 2. 多卡训练

```
CUDA_VISIBLE_DEVICES=0,1 python -m paddle.distributed.launch tools/main.py --config-file configs/cyclegan_cityscapes.yaml
```

## 预测

```
python tools/main.py --config-file configs/cyclegan_cityscapes.yaml --evaluate-only --load your_weight_path
```
- `--evaluate-only`: 是否仅进行预测。
- `--load (str)`: 训练好的权重路径。
