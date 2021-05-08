# First order motion model

## First order motion model原理

First order motion model的任务是image animation，给定一张源图片，给定一个驱动视频，生成一段视频，其中主角是源图片，动作是驱动视频中的动作。如下图所示，源图像通常包含一个主体，驱动视频包含一系列动作。

<div align="center">
  <img src="../../imgs/fom_demo.png" width="500"/>
</div>

以左上角的人脸表情迁移为例，给定一个源人物，给定一个驱动视频，可以生成一个视频，其中主体是源人物，视频中源人物的表情是由驱动视频中的表情所确定的。通常情况下，我们需要对源人物进行人脸关键点标注、进行表情迁移的模型训练。

但是这篇文章提出的方法只需要在同类别物体的数据集上进行训练即可，比如实现太极动作迁移就用太极视频数据集进行训练，想要达到表情迁移的效果就使用人脸视频数据集voxceleb进行训练。训练好后，我们使用对应的预训练模型就可以达到前言中实时image animation的操作。

## 联合人脸检测模型实现多人脸表情迁移

使用PaddleGAN提供的[人脸检测算法S3FD](https://github.com/PaddlePaddle/PaddleGAN/tree/develop/ppgan/faceutils/face_detection/detection)，将照片中多个人脸检测出来并进行表情迁移，实现多人同时换脸。

具体技术原理：

1. 使用S3FD人脸检测模型将照片中的每张人脸检测出来并抠出
2. 使用First Order Motion模型对抠出的每张人脸进行脸部表情迁移
3. 将完成表情迁移的人脸进行适当剪裁后贴回原照片位置

同时，PaddleGAN针对人脸的相关处理提供[faceutil工具](https://github.com/PaddlePaddle/PaddleGAN/tree/develop/ppgan/faceutils)，包括人脸检测、五官分割、关键点检测等能力。

## 使用方法
### 1 人脸测试
用户可上传一张单人/多人照片与驱动视频，并在如下命令中的source_image参数和driving_video参数分别换成自己的图片和视频路径，然后运行如下命令，即可完成单人/多人脸动作表情迁移，运行结果为命名为result.mp4的视频文件，保存在output文件夹中。

注意：使用多人脸时，尽量使用人脸间距较大的照片，效果更佳，也可通过手动调节ratio进行效果优化。

本项目中提供了原始图片和驱动视频供展示使用，运行的命令如下：

- 默认为单人脸:
```
cd applications/
python -u tools/first-order-demo.py  \
     --driving_video ../docs/imgs/fom_dv.mp4 \
     --source_image ../docs/imgs/fom_source_image.png \
     --ratio 0.4 \
     --relative --adapt_scale
```
- 多人脸：
```
cd applications/
python -u tools/first-order-demo.py  \
     --driving_video ../docs/imgs/fom_dv.mp4 \
     --source_image ../docs/imgs/fom_source_image_multi_person.jpg \
     --ratio 0.4 \
     --relative --adapt_scale \
     --multi_person

- driving_video: 驱动视频，视频中人物的表情动作作为待迁移的对象
- source_image: 原始图片，支持单人图片和多人图片，视频中人物的表情动作将迁移到该原始图片中的人物上
- relative: 指示程序中使用视频和图片中人物关键点的相对坐标还是绝对坐标，建议使用相对坐标，若使用绝对坐标，会导致迁移后人物扭曲变形
- adapt_scale: 根据关键点凸包自适应运动尺度
- ratio: 贴回驱动生成的人脸区域占原图的比例, 用户需要根据生成的效果调整该参数，尤其对于多人脸距离比较近的情况下需要调整改参数, 默认为0.4，调整范围是[0.4, 0.5]
- multi_person: 表示图片中有多张人脸，不加则默认为单人脸

### 2 训练
**数据集:**
- fashion 可以参考[这里](https://vision.cs.ubc.ca/datasets/fashion/)
- VoxCeleb 可以参考[这里](https://github.com/AliaksandrSiarohin/video-preprocessing)

**参数说明:**
- dataset_name.yaml: 需要配置自己的yaml文件及参数

- GPU单卡训练:
```
export CUDA_VISIBLE_DEVICES=0
python tools/main.py --config-file configs/dataset_name.yaml
```
- GPU多卡训练:
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch \
    tools/main.py \
    --config-file configs/dataset_name.yaml \

```

**例如:**
- GPU单卡训练:
```
export CUDA_VISIBLE_DEVICES=0
python tools/main.py --config-file configs/firstorder_fashion.yaml
```
- GPU多卡训练:
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch \
    tools/main.py \
    --config-file configs/firstorder_fashion.yaml \
```

**在线体验项目**

* **多人脸通用：https://aistudio.baidu.com/aistudio/projectdetail/1603391**
* **单人脸通用：https://aistudio.baidu.com/aistudio/projectdetail/1586056**

## 生成结果展示

![](../../imgs/first_order.gif)


## 参考文献

```
@InProceedings{Siarohin_2019_NeurIPS,
  author={Siarohin, Aliaksandr and Lathuilière, Stéphane and Tulyakov, Sergey and Ricci, Elisa and Sebe, Nicu},
  title={First Order Motion Model for Image Animation},
  booktitle = {Conference on Neural Information Processing Systems (NeurIPS)},
  month = {December},
  year = {2019}
}

```
