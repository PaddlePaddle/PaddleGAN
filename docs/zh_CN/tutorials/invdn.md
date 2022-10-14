[English](../../en_US/tutorials/invdn.md) | 中文

# 可逆去噪网络（InvDN）：真实噪声移除的一个轻量级方案

**Invertible Denoising Network: A Light Solution for Real Noise Removal** (CVPR 2021) 论文复现

官方源码：[https://github.com/Yang-Liu1082/InvDN](https://github.com/Yang-Liu1082/InvDN)

论文地址：[https://arxiv.org/abs/2104.10546](https://arxiv.org/abs/2104.10546)

## 1、简介

InvDN利用可逆网络把噪声图片分成低解析度干净图片和高频潜在表示， 其中高频潜在表示中含有噪声信息和内容信息。由于可逆网络是无损的， 如果我们能够将高频表示中的噪声信息分离， 那么就可以将其和低解析度干净图片一起重构成原分辨率的干净图片。但实际上去除高频信息中的噪声是很困难的， 本文通过直接将带有噪声的高频潜在表示替换为在还原过程中从先验分布中采样的另一个表示，进而结合低解析度干净图片重构回原分辨率干净图片。本文所实现网络是轻量级的， 且效果较好。

![invdn](https://user-images.githubusercontent.com/51016595/195344773-9ea17ef5-9edd-4310-bfff-36049bbcefde.png)

## 2 如何使用

### 2.1 快速体验

安装`PaddleGAN`之后进入`PaddleGAN`文件夹下，运行如下命令即生成修复后的图像`./output_dir/Denoising/image_name.png`

```sh
python applications/tools/invdn_denoising.py --images_path ${PATH_OF_IMAGE}
```
其中`PATH_OF_IMAGE`为你需要去噪的图像路径，或图像所在文件夹的路径。

- 注意，作者原代码中，测试时使用了蒙特卡洛自集成（Monte Carlo self-ensemble）以提高性能，但是会拖慢速度。用户可以自由选择是否使用 `--disable_mc` 参数来关闭蒙特卡洛自集成以提高速度。（$test$ 时默认开启蒙特卡洛自集成，而 $train$ 和 $valid$ 时默认关闭蒙特卡洛自集成。）

### 2.2 数据准备

#### **训练数据**

本文所使用的数据集为SIDD，其中训练集为 [SIDD-Medium](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php)。按照论文要求，需要将数据集处理为 $512 \times 512$ 的 patches。此外，本文训练时需要产生低分辨率版本的GT图像，其尺寸为 $128 \times 128$。将低分辨率图像记作LQ。

已经处理好的数据，放在了 [Ai Studio](https://aistudio.baidu.com/aistudio/datasetdetail/172084) 里。

训练数据放在：`data/SIDD_Medium_Srgb_Patches_512/train/` 下。

#### **测试数据**

验证集为 [SIDD_valid](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php)。官网下载的验证集为 `./ValidationNoisyBlocksSrgb.mat 和 ./ValidationGtBlocksSrgb.mat`，建议转换为 $.png$ 格式更为方便。

已经转换好的数据，放在了 [Ai Studio](https://aistudio.baidu.com/aistudio/datasetdetail/172069) 里。

验证集数据放在：`data/SIDD_Valid_Srgb_Patches_256/valid/` 下。

- 经过处理之后，`PaddleGAN/data` 文件夹下的文件结构为
```sh
data
├─ SIDD_Medium_Srgb_Patches_512
│  └─ train
│      ├─ GT
│      │      0_0.PNG
│      │      ...
│      ├─ LQ
│      │      0_0.PNG
│      │      ...
│      └─ Noisy
│              0_0.PNG
│              ...
│
└─ SIDD_Valid_Srgb_Patches_256
    └─ valid
        ├─ GT
        │      0_0.PNG
        │      ...
        └─ Noisy
                0_0.PNG
                ...
```

### 2.3 训练

运行以下命令来快速开始训练：
```sh
python -u tools/main.py --config-file configs/invdn_denoising.yaml
```
- TIPS：
在复现时，为了保证总 $epoch$ 数目和论文配置相同，我们需要确保 $ total\_batchsize*iter == 1gpus*14bs*600000iters$。同时 $batchsize$ 改变时也要确保 $batchsize/learning\_rate == 14/0.0002$ 。
例如，在使用单机四卡时，将单卡 $batchsize$ 设置为14，此时实际的总 $batchsize$ 应为14*4，需要将总 $iters$ 设置为为150000，且学习率扩大到8e-4。

### 2.4 测试

运行以下命令来快速开始测试：
```sh
python tools/main.py --config-file configs/invdn_denoising.yaml --evaluate-only --load ${PATH_OF_WEIGHT}
```

## 3 结果展示

去噪
| 模型 | 数据集 | PSNR/SSIM |
|---|---|---|
| InvDN | SIDD |  39.29 / 0.956 |


## 4 模型下载

| 模型 | 下载地址 |
|---|---|
| InvDN| [InvDN_Denoising](https://paddlegan.bj.bcebos.com/models/InvDN_Denoising.pdparams) |



# 参考文献

- [https://arxiv.org/abs/2104.10546](https://arxiv.org/abs/2104.10546)

```
@article{liu2021invertible,
  title={Invertible Denoising Network: A Light Solution for Real Noise Removal},
  author={Liu, Yang and Qin, Zhenyue and Anwar, Saeed and Ji, Pan and Kim, Dongwoo and Caldwell, Sabrina and Gedeon, Tom},
  journal={arXiv preprint arXiv:2104.10546},
  year={2021}
}
```
