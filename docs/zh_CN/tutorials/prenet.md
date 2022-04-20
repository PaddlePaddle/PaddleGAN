# PReNet

## 1 简介
Progressive Image Deraining Networks: A Better and Simpler Baseline提出一种多阶段渐进的残差网络，每一个阶段都是resnet，每一res块的输入为上一res块输出和原始雨图，另外采用使用SSIM损失进行训练,进一步提升了网络的性能，网络总体简洁高效，在各种数据集上表现良好,为图像去雨提供了一个很好的基准。
<div align="center">
    <img src="https://github.com/simonsLiang/PReNet_paddle/blob/main/data/net.jpg" width=800">
</div>

## 2 如何使用

### 2.1 数据准备

  数据集(RainH.zip) 可以在[此处](https://pan.baidu.com/s/1_vxCatOV3sOA6Vkx1l23eA?pwd=vitu)下载,将其解压到./data路径下。

  数据集文件结构如下:

  ```
    ├── data
        ├── RainTrainH
            ├── rain
                ├── 1.png
                └── 2.png
                    .
                    .
            └── norain
                ├── 1.png
                └── 2.png
                    .
                    .
        └── Rain100H
            ├── rain
                ├── 001.png
                └── 002.png
                    .
                    .
            └── norain
                ├── 001.png
                └── 002.png
                    .
                    .
  ```

### 2.2 训练和测试


  训练模型：
  ```
     python -u tools/main.py --config-file configs/prenet.yaml
  ```

  测试模型：
  ```
     python tools/main.py --config-file configs/prenet.yaml --evaluate-only --load ${PATH_OF_WEIGHT}
  ```

## 3 预测结果

输入：

<div align="center">
    <img src="https://github.com/simonsLiang/PReNet_paddle/blob/main/data/rain-001.png" width=300">
</div>

输出：

<div align="center">
    <img src="https://github.com/simonsLiang/PReNet_paddle/blob/main/data/derain-rain-001.png" width=300">
</div>

## 4 模型参数下载
| 模型 | 数据集 |
|---|---|
| [PReNet](https://paddlegan.bj.bcebos.com/models/PReNet.pdparams)  | [RainH.zip](https://pan.baidu.com/s/1_vxCatOV3sOA6Vkx1l23eA?pwd=vitu) |




## 参考

- 1. [Progressive Image Deraining Networks: A Better and Simpler Baseline](https://arxiv.org/pdf/1901.09221v3.pdf)


```
@inproceedings{ren2019progressive,
   title={Progressive Image Deraining Networks: A Better and Simpler Baseline},
   author={Ren, Dongwei and Zuo, Wangmeng and Hu, Qinghua and Zhu, Pengfei and Meng, Deyu},
   booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
   year={2019},
 }
```
