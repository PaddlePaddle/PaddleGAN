# GauGAN（加SimAM注意力的改进版）

（这里文档主要简单介绍GauGAN的训练和预测，效果展示、论文简介等后面再完善）

## 1.简介：

本模型在 GauGAN 的 SPADE 模块上添加了无参的 SimAM 注意力模块，增强了生成图片的立体质感。

## 2.快速体验

预训练模型可以从如下地址下载: （上载后添加）

输入一张png格式的语义标签图片给模型，输出一张按标签语义生成的照片风格的图片。预测代码如下：

```
python applications/tools/photopen.py \
  --semantic_label_path test/sem.png \
  --weight_path test/n_g.pdparams \
  --output_path output_dir/pic.jpg \
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
       dataset.train.batch_size=4```

* config-file：训练使用的超参设置 yamal 文件的存储路径
* model.generator.norm_G：设置使用 syncbatch 归一化，使多个 GPU 中的数据一起进行归一化
* model.batchSize：设置模型的 batch size，一般为 GPU 个数的整倍数
* dataset.train.batch_size：设置数据读取的 batch size，要和模型的 batch size 一致

### 3.3 继续训练

`python -u tools/main.py --config-file configs/photopen.yaml --resume output_dir\photopen-2021-09-30-15-59\iter_3_checkpoint.pdparams`

* config-file：训练使用的超参设置 yamal 文件的存储路径
* resume：指定读取的 checkpoint 路径

## 4.模型效果展示

![](https://ai-studio-static-online.cdn.bcebos.com/aea83e49828c45168be390ff21339bb583dd2f043e954050b221bedd27ef6d9d)

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
