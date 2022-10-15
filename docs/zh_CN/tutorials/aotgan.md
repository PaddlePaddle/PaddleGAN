# AOT GAN

## 1. 简介

本应用的 AOT GAN 模型出自论文《Aggregated Contextual Transformations for High-Resolution Image Inpainting》，其通过聚合不同膨胀率的空洞卷积学习到的图片特征，刷出了inpainting任务的新SOTA。模型推理效果如下：

![](https://ai-studio-static-online.cdn.bcebos.com/c3b71d7f28ce4906aa7cccb10ed09ae5e317513b6dbd471aa5cca8144a7fd593)

**论文:** [Aggregated Contextual Transformations for High-Resolution Image Inpainting](https://paperswithcode.com/paper/aggregated-contextual-transformations-for)

**参考repo:** [https://github.com/megvii-research/NAFNet](https://github.com/megvii-research/NAFNet)

## 2.快速体验

预训练模型权重文件 g.pdparams 可以从如下地址下载: （https://paddlegan.bj.bcebos.com/models/AotGan_g.pdparams）

输入一张 512x512 尺寸的图片和擦除 mask 给模型，输出一张补全（inpainting）的图片。预测代码如下：

```
python applications/tools/aotgan.py \
	--input_image_path data/aotgan/armani1.jpg \
	--input_mask_path data/aotgan/armani1.png \
	--weight_path test/aotgan/g.pdparams \
	--output_path output_dir/armani_pred.jpg \
	--config-file configs/aotgan.yaml
```

**参数说明:**
* input_image_path：输入图片路径
* input_mask_path：输入擦除 mask 路径
* weight_path：训练完成的模型权重存储路径，为 statedict 格式（.pdparams）的 Paddle 模型行权重文件
* output_path：预测生成图片的存储路径
* config-file：存储参数设定的yaml文件存储路径，与训练过程使用同一个yaml文件，预测参数由 predict 下字段设定

AI Studio 快速体验项目：（https://aistudio.baidu.com/aistudio/datasetdetail/165081）

## 3.训练

**数据准备:**

* 训练用的图片解压到项目路径下的 data/aotgan/train_img 文件夹内，可包含多层目录，dataloader会递归读取每层目录下的图片。训练用的mask图片解压到项目路径下的 data/aotgan/train_mask 文件夹内。
* 验证用的图片和mask图片相应的放到项目路径下的 data/aotgan/val_img 文件夹和 data/aotgan/val_mask 文件夹内。

数据集目录结构如下：

```
└─data
    └─aotgan
        ├─train_img
        ├─train_mask
        ├─val_img
        └─val_mask
```

* 训练预训练模型的权重使用了 Place365Standard 数据集的训练集图片，以及 NVIDIA Irregular Mask Dataset 数据集的测试集掩码图片。Place365Standard 的训练集为 160万张长或宽最小为 512 像素的图片。NVIDIA Irregular Mask Dataset 的测试集为 12000 张尺寸为 512 x 512 的不规则掩码图片。数据集下载链接：[Place365Standard](http://places2.csail.mit.edu/download.html)、[NVIDIA Irregular Mask Dataset](https://nv-adlr.github.io/publication/partialconv-inpainting)

### 3.1 gpu 单卡训练

`python -u tools/main.py --config-file configs/aotgan.yaml`

* config-file：训练使用的超参设置 yamal 文件的存储路径

### 3.2 gpu 多卡训练

```
!python -m paddle.distributed.launch \
    tools/main.py \
    --config-file configs/photopen.yaml \
    -o dataset.train.batch_size=6
```

* config-file：训练使用的超参设置 yamal 文件的存储路径
* -o dataset.train.batch_size=6：-o 设置参数覆盖 yaml 文件中的值，这里调整了 batch_size 参数

### 3.3 继续训练

```
python -u tools/main.py \
	--config-file configs/aotgan.yaml \
	--resume  output_dir/[path_to_checkpoint]/iter_[iternumber]_checkpoint.pdparams
```

* config-file：训练使用的超参设置 yamal 文件的存储路径
* resume：指定读取的 checkpoint 路径

### 3.4 实验结果展示

在Places365模型的验证集上的指标如下

|  mask   | PSNR  | SSIM  | download  |
|  ----  | ----  | ----  | ----  |
|  20-30%   | 26.04001  | 0.89011  | [download](https://paddlegan.bj.bcebos.com/models/AotGan_g.pdparams)  |

## 4. 参考链接与文献
@inproceedings{yan2021agg,
  author = {Zeng, Yanhong and Fu, Jianlong and Chao, Hongyang and Guo, Baining},
  title = {Aggregated Contextual Transformations for High-Resolution Image Inpainting},
  booktitle = {Arxiv},
  pages={-},
  year = {2020}
}
