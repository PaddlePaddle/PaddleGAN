# 1.单幅遥感图像超分辨率重建

## 1.1 背景和原理介绍

 **意义与应用场景**：单幅影像超分辨率重建一直是low-level视觉领域中一个比较热门的任务，其可以成为修复老电影、老照片的技术手段，也可以为图像分割、目标检测等下游任务提供质量较高的数据。在遥感中的应用场景也比较广泛，例如：在**船舶检测和分类**等诸多遥感影像应用中，**提高遥感影像分辨率具有重要意义**。

**原理**：单幅遥感影像的超分辨率重建本质上与单幅影像超分辨率重建类似，均是使用RGB三通道的低分辨率影像生成纹理清晰的高分辨率影像。本项目复现的论文是[Yulun Zhang](http://yulunzhang.com/), [Kunpeng Li](https://kunpengli1994.github.io/), [Kai Li](http://kailigo.github.io/), [Lichen Wang](https://sites.google.com/site/lichenwang123/), [Bineng Zhong](https://scholar.google.de/citations?user=hvRBydsAAAAJ&hl=en), and [Yun Fu](http://www1.ece.neu.edu/~yunfu/), 发表在ECCV 2018上的论文[《Image Super-Resolution Using Very Deep Residual Channel Attention Networks》](https://arxiv.org/abs/1807.02758)。
作者提出了一个深度残差通道注意力网络（RCAN），引入一种通道注意力机制（CA），通过考虑通道之间的相互依赖性来自适应地重新调整特征。该模型取得优异的性能，因此本项目选择RCAN进行单幅遥感影像的x4超分辨率重建。

## 1.2 如何使用

### 1.2.1 数据准备
 本项目的训练分为两个阶段，第一个阶段使用[DIV2K数据集](https://data.vision.ee.ethz.ch/cvl/DIV2K/)进行预训练RCANx4模型，然后基于该模型再使用[遥感超分数据集合](https://aistudio.baidu.com/aistudio/datasetdetail/129011)进行迁移学习。
 - 关于DIV2K数据的准备方法参考[该文档](./single_image_super_resolution.md)
 - 遥感超分数据准备
    - 数据已经上传至AI studio中，该数据为从UC Merced Land-Use Dataset 21 级土地利用图像遥感数据集中抽取部分遥感影像，通过BI退化生成的HR-LR影像对用于训练超分模型，其中训练集6720对，测试集420对
    - 下载解压后的文件组织形式如下
    ```
    ├── RSdata_for_SR
        ├── train_HR
        ├── train_LR
        |    └──x4
        ├── test_HR
        ├── test_LR
        |    └──x4
    ```

### 1.2.2 DIV2K数据集上训练/测试

首先是在DIV2K数据集上训练RCANx4模型，并以Set14作为测试集。按照论文需要准备RCANx2作为初始化权重，可通过下表进行获取。

| 模型 | 数据集 | 下载地址 |
|---|---|---|
| RCANx2  | DIV2K | [RCANx2](https://paddlegan.bj.bcebos.com/models/RCAN_X2_DIV2K.pdparams)


将DIV2K数据按照 [该文档](./single_image_super_resolution.md)所示准备好后，执行以下命令训练模型，`--load`的参数为下载好的RCANx2模型权重所在路径。

```shell
python -u tools/main.py --config-file configs/rcan_rssr_x4.yaml --load ${PATH_OF_WEIGHT}
```

训练好后，执行以下命令可对测试集Set14预测，`--load`的参数为训练好的RCANx4模型权重
```shell
python tools/main.py --config-file configs/rcan_rssr_x4.yaml --evaluate-only --load ${PATH_OF_WEIGHT}
```

本项目在DIV2K数据集训练迭代第57250次得到的权重[RCAN_X4_DIV2K](https://pan.baidu.com/s/1rI7yUdD4T1DE0RZB5yHXjA)（提取码：aglw），在Set14数据集上测得的精度：`PSNR:28.8959 SSIM:0.7896`

### 1.2.3 遥感超分数据上迁移学习训练/测试
- 使用该数据集，需要修改`rcan_rssr_x4.yaml`文件中训练集与测试集的高分辨率图像路径和低分辨率图像路径，即文件中的`gt_folder`和`lq_folder`。
- 同时，由于使用了在DIV2K数据集上训练的RCAN_X4_DIV2K模型权重来进行迁移学习，所以训练的迭代次数`total_iters`也可以进行修改，并不需要很多次数的迭代就能有良好的效果。训练模型中`--load`的参数为下载好的RCANx4模型权重所在路径。

训练模型:
```shell
python -u tools/main.py --config-file configs/rcan_rssr_x4.yaml --load ${PATH_OF_RCANx4_WEIGHT}
```
测试模型：
```shell
python -u tools/main.py --config-file configs/rcan_rssr_x4.yaml --load ${PATH_OF_RCANx4_WEIGHT}
```

## 1.3 实验结果

- RCANx4遥感影像超分效果

<img src=../../imGs/RSSR.png></img>

- [RCAN遥感影像超分辨率重建 Ai studio 项目在线体验](https://aistudio.baidu.com/aistudio/projectdetail/3508912)

