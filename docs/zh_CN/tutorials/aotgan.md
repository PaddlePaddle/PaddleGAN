# GauGAN（加SimAM注意力的改进版）

## 1.简介：

本应用的模型出自论文《Semantic Image Synthesis with Spatially-Adaptive Normalization》，是一个像素风格迁移网络 Pix2PixHD，能够根据输入的语义分割标签生成照片风格的图片。为了解决模型归一化层导致标签语义信息丢失的问题，论文作者向 Pix2PixHD 的生成器网络中添加了 SPADE（Spatially-Adaptive Normalization）空间自适应归一化模块，通过两个卷积层保留了归一化时训练的缩放与偏置参数的空间维度，以增强生成图片的质量。

![](https://ai-studio-static-online.cdn.bcebos.com/4fc3036fdc18443a9dcdcddb960b5da1c689725bbfa84de2b92421a8640e0ee5)

此模型在 GauGAN 的 SPADE 模块上添加了无参的 SimAM 注意力模块，增强了生成图片的立体质感。

![](https://ai-studio-static-online.cdn.bcebos.com/94731023eab94b1b97b9ca80bd3b30830c918cf162d046bd88540dda450295a3)

## 2.快速体验

预训练模型可以从如下地址下载: （https://paddlegan.bj.bcebos.com/models/photopen.pdparams）

输入一张png格式的语义标签图片给模型，输出一张按标签语义生成的照片风格的图片。预测代码如下：

```
python applications/tools/photopen.py \
  --semantic_label_path test/sem.png \
  --weight_path test/n_g.pdparams \
  --output_path test/pic.jpg \
  --config-file configs/photopen.yaml
```

**参数说明:**
* semantic_label_path：输入的语义标签路径，为png图片文件
* weight_path：训练完成的模型权重存储路径，为 statedict 格式（.pdparams）的 Paddle 模型行权重文件
* output_path：预测生成图片的存储路径
* config-file：存储参数设定的yaml文件存储路径，与训练过程使用同一个yaml文件，预测参数由 predict 下字段设定

## 3.训练

**数据准备:**

数据集目录结构如下：

```
└─coco_stuff
    ├─train_img
    └─train_inst
```

coco_stuff 是数据集根目录可任意改变，其下的 train_img 子目录存放训练用的风景图片（一般jpg格式），train_inst 子目录下存放与风景图片文件名一一对应、尺寸相同的语义标签图片（一般png格式）。

### 3.1 gpu 单卡训练

`python -u tools/main.py --config-file configs/photopen.yaml`

* config-file：训练使用的超参设置 yamal 文件的存储路径

### 3.2 gpu 多卡训练

```
!python -m paddle.distributed.launch \
    tools/main.py \
    --config-file configs/photopen.yaml \
    -o model.generator.norm_G=spectralspadesyncbatch3x3 \
       model.batchSize=4 \
       dataset.train.batch_size=4
```

* config-file：训练使用的超参设置 yamal 文件的存储路径
* model.generator.norm_G：设置使用 syncbatch 归一化，使多个 GPU 中的数据一起进行归一化
* model.batchSize：设置模型的 batch size，一般为 GPU 个数的整倍数
* dataset.train.batch_size：设置数据读取的 batch size，要和模型的 batch size 一致

### 3.3 继续训练

`python -u tools/main.py --config-file configs/photopen.yaml --resume output_dir\photopen-2021-09-30-15-59\iter_3_checkpoint.pdparams`

* config-file：训练使用的超参设置 yamal 文件的存储路径
* resume：指定读取的 checkpoint 路径

## 4.模型效果展示

![](https://ai-studio-static-online.cdn.bcebos.com/72a4a6ede506436ebaa6fb6982aa899607a80e20a54f4b138fb7ae9673e12e6e)

## 5.参考

```
@inproceedings{park2019SPADE,
  title={Semantic Image Synthesis with Spatially-Adaptive Normalization},
  author={Park, Taesung and Liu, Ming-Yu and Wang, Ting-Chun and Zhu, Jun-Yan},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}

@InProceedings{pmlr-v139-yang21o,
    title = 	 {SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks},
    author =       {Yang, Lingxiao and Zhang, Ru-Yuan and Li, Lida and Xie, Xiaohua},
    booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
    pages = 	 {11863--11874},
    year = 	 {2021},
    editor = 	 {Meila, Marina and Zhang, Tong},
    volume = 	 {139},
    series = 	 {Proceedings of Machine Learning Research},
    month = 	 {18--24 Jul},
    publisher =    {PMLR},
    pdf = 	 {http://proceedings.mlr.press/v139/yang21o/yang21o.pdf},
    url = 	 {http://proceedings.mlr.press/v139/yang21o.html}
}
```
