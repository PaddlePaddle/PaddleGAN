## 数据准备

推荐把数据集软链接到 `$PaddleGAN/data`. 软链接后的目录结构如下图所示：

```
PaddleGAN
|-- configs
|-- data
|   |-- cityscapes
|   |   ├── test
|   |   ├── testA
|   |   ├── testB
|   |   ├── train
|   |   ├── trainA
|   |   └── trainB
|   ├── horse2zebra
|   |   ├── testA
|   |   ├── testB
|   |   ├── trainA
|   |   └── trainB
|   └── facades
|       ├── test
|       ├── train
|       └── val
|-- docs
|-- ppgan
|-- tools

```

### cyclegan 相关的数据集下载
cyclgan模型相关的数据集可以在[这里](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/)下载

### pix2pix 相关的数据集下载
pixel2pixel模型相关的数据集可以在[这里](hhttps://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/)下载
