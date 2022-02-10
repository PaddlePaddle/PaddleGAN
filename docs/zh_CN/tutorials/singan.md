# SinGAN

## 简介

SinGAN是一种新的可以从单个自然图像中学习的无条件生成模型。该模型包含一个全卷积生成对抗网络的金字塔结构，每个生成对抗网络负责学习不同在不同比例的图像上的块分布。这允许生成任意大小和纵横比的新样本，具有显著的可变性，但同时保持训练图像的全局结构和精细纹理。与以往单一图像生成方案相比，该方法不局限于纹理图像，也没有条件（即从噪声中生成样本）。

## 使用方法

### 配置说明

我们为SinGAN提供了4个配置文件：

- `singan_universal.yaml`
- `singan_sr.yaml`
- `singan_animation.yaml`
- `singan_finetune.yaml`

其中`singan_universal.yaml`对所有任务都适用配置，`singan_sr.yaml`是官方建议的用于超分任务的配置，`singan_animation.yaml`是官方建议的用于“静图转动”任务的配置。本文档展示的结果均由`singan_universal.yaml`训练而来。对于手绘转照片任务，使用`singan_universal.yaml`训练后再用`singan_finetune.yaml`微调会得到更好的结果。

### 训练

启动训练：

```bash
python tools/main.py -c configs/singan_universal.yaml \
                     -o model.train_image=训练图片.png
```

为“手绘转照片”任务微调：

```bash
python tools/main.py -c configs/singan_finetune.yaml \
                     -o model.train_image=训练图片.png \
                     --load 已经训练好的模型.pdparams
```

### 测试
运行下面的命令，可以随机生成一张图片。需要注意的是，`训练图片.png`应当位于`data/singan`目录下，或者手动调整配置文件中`dataset.test.dataroot`的值。此外，这个目录中只能包含`训练图片.png`这一张图片。
```bash
python tools/main.py -c configs/singan_universal.yaml \
                     -o model.train_image=训练图片.png \
                     --load 已经训练好的模型.pdparams \
                     --evaluate-only
```

### 导出生成器权重

训练结束后，需要使用 ``tools/extract_weight.py`` 来从训练模型（包含了生成器和判别器）中提取生成器的权重来给`applications/tools/singan.py`进行推理，以实现SinGAN的各种应用。

```bash
python tools/extract_weight.py 训练过程中保存的权重文件.pdparams --net-name netG --output 生成器权重文件.pdparams
```

### 推理及结果展示

*注意：您可以下面的命令中的`--weight_path 生成器权重文件.pdparams`可以换成`--pretrained_model <model> `来体验训练好的模型，其中`<model>`可以是`trees`、`stone`、`mountains`、`birds`和`lightning`。*

#### 随机采样

```bash
python applications/tools/singan.py \
       --weight_path 生成器权重文件.pdparams \
       --mode random_sample \
       --scale_v 1 \ # vertical scale
       --scale_h 1 \ # horizontal scale
       --n_row 2 \
       --n_col 2
```

|训练图片|随机采样结果|
| ---- | ---- |
|![birds](https://user-images.githubusercontent.com/91609464/153211448-2614407b-a30b-467c-b1e5-7db88ff2ca74.png)|![birds-random_sample](https://user-images.githubusercontent.com/91609464/153211573-1af108ba-ad42-438a-94a9-e8f8f3e091eb.png)|

#### 图像编辑&风格和谐化

```bash
python applications/tools/singan.py \
       --weight_path 生成器权重文件.pdparams \
       --mode editing \ # or harmonization
       --ref_image 编辑后的图片.png \
       --mask_image 编辑区域标注图片.png \
       --generate_start_scale 2
```


|训练图片|编辑图片|编辑区域标注|SinGAN生成|
|----|----|----|----|
|![stone](https://user-images.githubusercontent.com/91609464/153211778-bb94d29d-a2b4-4d04-9900-89b20ae90b90.png)|![stone-edit](https://user-images.githubusercontent.com/91609464/153211867-df3d9035-d320-45ec-8043-488e9da49bff.png)|![stone-edit-mask](https://user-images.githubusercontent.com/91609464/153212047-9620f73c-58d9-48ed-9af7-a11470ad49c8.png)|![stone-edit-mask-result](https://user-images.githubusercontent.com/91609464/153211942-e0e639c2-3ea6-4ade-852b-73757b0bbab0.png)|

#### 超分

```bash
python applications/tools/singan.py \
       --weight_path 生成器权重文件.pdparams \
       --mode sr \
       --ref_image 待超分的图片亦即用于训练的图片.png \
       --sr_factor 4
```
|训练图片|超分结果|
| ---- | ---- |
|![mountains](https://user-images.githubusercontent.com/91609464/153212146-efbbbbd6-e045-477a-87ae-10f121341060.png)|![sr](https://user-images.githubusercontent.com/91609464/153212176-530b7075-e72b-4c05-ad3e-2f2cdfc76dea.png)|


#### 静图转动

```bash
python applications/tools/singan.py \
       --weight_path 生成器权重文件.pdparams \
       --mode animation \
       --animation_alpha 0.6 \ # this parameter determines how close the frames of the sequence remain to the training image
       --animation_beta 0.7 \ # this parameter controls the smoothness and rate of change in the generated clip
       --animation_frames 20 \ # frames of animation
       --animation_duration 0.1	# duration of each frame
```

|训练图片|动画效果|
| ---- | ---- |
|![lightning](https://user-images.githubusercontent.com/91609464/153212291-6f8976bd-e873-423e-ab62-77997df2df7a.png)|![animation](https://user-images.githubusercontent.com/91609464/153212372-0543e6d6-5842-472b-af50-8b22670270ae.gif)|


#### 手绘转照片
```bash
python applications/tools/singan.py \
       --weight_path 生成器权重文件.pdparams \
       --mode paint2image \
       --ref_image 手绘图片.png \
       --generate_start_scale 2
```
|训练图片|手绘图片|SinGAN生成|SinGAN微调后生成|
|----|----|----|----|
|![trees](https://user-images.githubusercontent.com/91609464/153212536-0bb6489d-d488-49e0-a6b5-90ef578c9e4f.png)|![trees-paint](https://user-images.githubusercontent.com/91609464/153212511-ef2c6bea-1f8c-4685-951b-8db589414dfe.png)|![trees-paint2image](https://user-images.githubusercontent.com/91609464/153212531-c080c705-fd58-4ade-aac6-e2134838a75f.png)|![trees-paint2image-finetuned](https://user-images.githubusercontent.com/91609464/153212529-51d8d29b-6b58-4f29-8792-4b2b04f9266e.png)|



## 参考文献

```
@misc{shaham2019singan,
      title={SinGAN: Learning a Generative Model from a Single Natural Image}, 
      author={Tamar Rott Shaham and Tali Dekel and Tomer Michaeli},
      year={2019},
      eprint={1905.01164},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

