# 1 AnimeGANv2

## 1.1 原理介绍

[AnimeGAN](https://github.com/TachibanaYoshino/AnimeGANv2)基于2018年[CVPR论文CartoonGAN](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf)基础上对其进行了一些改进，主要消除了过度风格化以及颜色伪影区域的问题。对于具体原理可以参见作者[知乎文章](https://zhuanlan.zhihu.com/p/76574388?from_voters_page=true)。AnimeGANv2是作者在AnimeGAN的基础上添加了`total variation loss`的新模型。


## 1.2 如何使用

### 1.2.1 快速体验

安装`PaddleGAN`之后运行如下代码即生成风格化后的图像`output_dir/anime.png`，其中`PATH_OF_IMAGE`为你需要转换的图像路径。

```python
from ppgan.apps import AnimeGANPredictor
predictor = AnimeGANPredictor()
predictor.run(PATH_OF_IMAGE)
```

或者在终端中运行如下命令，也可获得相同结果：

```sh
python applications/tools/animeganv2.py --input_image ${PATH_OF_IMAGE}
```

### 1.2.1 数据准备

我们下载作者提供的训练数据，训练数据可以从[这里](https://github.com/TachibanaYoshino/AnimeGAN/releases/tag/dataset-1)下载。
下载后解压到data目录下：

```sh
wget https://github.com/TachibanaYoshino/AnimeGAN/releases/download/dataset-1/dataset.zip
cd PaddleGAN
unzip YOUR_DATASET_DIR/dataset.zip -d data/animedataset
```

解压完成后数据分布如下所示：

```sh
animedataset
├── Hayao
│   ├── smooth
│   └── style
├── Paprika
│   ├── smooth
│   └── style
├── Shinkai
│   ├── smooth
│   └── style
├── SummerWar
│   ├── smooth
│   └── style
├── test
│   ├── HR_photo
│   ├── label_map
│   ├── real
│   ├── test_photo
│   └── test_photo256
├── train_photo
└── val
```

### 1.2.2 训练
  示例以训练Hayao风格的数据为例。

  1.  为了保证模型具备生成原图的效果，需要预热模型:
  ```sh
  python tools/main.py --config-file configs/animeganv2_pretrain.yaml
  ```

  1.  预热模型完成后，训练风格迁移模型:
  **注意：** 必须先修改在`configs/animeganv2.yaml`中的`pretrain_ckpt`参数，确保指向正确的 **预热模型权重路径**
  设置`batch size=4`，`learning rate=0.0002`，在一个  GTX2060S GPU上训练30个epoch即可获得较好的效果，其他超参数请参考`configs/animeganv2.yaml`。

  ```sh
  python tools/main.py --config-file configs/animeganv2.yaml
  ```

  1.  改变目标图像的风格
  修改`configs/animeganv2.yaml`中的`style`参数即可改变风格(目前可选择`Hayao,Paprika,Shinkai,SummerWar`)。如果您想使用自己的数据集，可以在配置文件中修改数据集为您自己的数据集。

  **注意：** 修改目标风格后，必须计算目标风格数据集的像素均值，并修改`configs/animeganv2.yaml`中的`transform_anime->Add->value`参数。

  如下例子展示了如何获得`Hayao`风格图像的像素均值：
  ```sh
  python tools/animegan_picmean.py --dataset data/animedataset/Hayao/style
  image_num: 1792
  100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1792/1792 [00:04<00:00, 444.95it/s]
  RGB mean diff
  [-4.4346957 -8.665916  13.100612 ]
  ```

### 1.2.3 测试

测试模型：
```sh
python tools/main.py --config-file configs/animeganv2.yaml --evaluate-only --load ${PATH_OF_WEIGHT}
```

## 1.3 结果展示
| 原始图像                            | 风格化后图像                       |
| ----------------------------------- | ---------------------------------- |
| ![](../../imgs/animeganv2_test.jpg) | ![](../../imgs/animeganv2_res.jpg) |
