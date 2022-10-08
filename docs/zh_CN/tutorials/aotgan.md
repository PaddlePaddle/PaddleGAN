# AOT GAN

## 1. 简介

本应用的 AOT GAN 模型出自论文《Aggregated Contextual Transformations for High-Resolution Image Inpainting》，其通过聚合不同膨胀率的空洞卷积学习到的图片特征，刷出了inpainting任务的新SOTA。模型推理效果如下：

![](https://ai-studio-static-online.cdn.bcebos.com/c3b71d7f28ce4906aa7cccb10ed09ae5e317513b6dbd471aa5cca8144a7fd593)

**论文:** [Aggregated Contextual Transformations for High-Resolution Image Inpainting](https://paperswithcode.com/paper/aggregated-contextual-transformations-for)

**参考repo:** [https://github.com/megvii-research/NAFNet](https://github.com/megvii-research/NAFNet)

## 2.快速体验

预训练模型权重文件 g.pdparams 可以从如下地址下载: （https://aistudio.baidu.com/aistudio/datasetdetail/165081）

输入一张 512x512 尺寸的图片和擦除 mask 给模型，输出一张补全（inpainting）的图片。预测代码如下：

```
python applications/tools/aotgan.py \
	--input_image_path test/aotgan/armani1.jpg \
	--input_mask_path test/aotgan/armani1.png \
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

## 3.训练

**数据准备:**

* 训练用的图片解压到项目路径下的 dataset/train_img 文件夹内，可包含多层目录，dataloader会递归读取每层目录下的图片。训练用的mask图片解压到项目路径下的 dataset/train_mask 文件夹内。
* 验证用的图片和mask图片相应的放到项目路径下的 dataset/val_img 文件夹和 dataset/val_mask 文件夹内。

数据集目录结构如下：

```
└─dataset
    ├─train_img
    ├─train_mask
    ├─val_img
    └─val_mask
```

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
	--resume  output_dir/aotgan-2022-10-08-18-00/iter_200_checkpoint.pdparams
```

* config-file：训练使用的超参设置 yamal 文件的存储路径
* resume：指定读取的 checkpoint 路径

## 4. 参考链接与文献
@inproceedings{yan2021agg,
  author = {Zeng, Yanhong and Fu, Jianlong and Chao, Hongyang and Guo, Baining},
  title = {Aggregated Contextual Transformations for High-Resolution Image Inpainting},
  booktitle = {Arxiv},
  pages={-},
  year = {2020}
}

