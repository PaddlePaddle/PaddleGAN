
# LapStyle

 **LapStyle--拉普拉斯金字塔风格化网络**，是一种能够生成高质量风格化图的快速前馈风格化网络，能渐进地生成复杂的纹理迁移效果，同时能够在**512分辨率**下达到**100fps**的速度。可实现多种不同艺术风格的快速迁移，在艺术图像生成、滤镜等领域有广泛的应用。

本文档提供CVPR2021论文"Drafting and Revision: Laplacian Pyramid Network for Fast High-Quality Artistic Style Transfer"的官方代码。

## 1. 论文介绍

艺术风格迁移的目的是将一个实例图像的艺术风格迁移到一个内容图像。目前，基于优化的方法已经取得了很好的合成质量，但昂贵的时间成本限制了其实际应用。

同时，前馈方法仍然不能合成复杂风格，特别是存在全局和局部模式时。受绘制草图和修改细节这一常见绘画过程的启发，[论文](https://arxiv.org/pdf/2104.05376.pdf) 提出了一种新的前馈方法拉普拉斯金字塔网络（LapStyle）。

LapStyle首先通过绘图网络（Drafting Network）传输低分辨率的全局风格模式。然后通过修正网络（Revision Network）对局部细节进行高分辨率的修正，它根据拉普拉斯滤波提取的图像纹理和草图产生图像残差。通过叠加具有多个拉普拉斯金字塔级别的修订网络，可以很容易地生成更高分辨率的细节。最终的样式化图像是通过聚合所有金字塔级别的输出得到的。论文还引入了一个补丁鉴别器，以更好地对抗的学习局部风格。实验表明，该方法能实时合成高质量的风格化图像，并能正确生成整体风格模式。

![lapstyle_overview](https://user-images.githubusercontent.com/79366697/118654987-b24dc100-b81b-11eb-9430-d84630f80511.png)

## 2. 快速体验

PaddleGAN为大家提供了四种不同艺术风格的预训练模型，风格预览如下：

|                             原图                             | StarryNew                                                    | Stars                                                        | Ocean                                                        | Circuit                                                      |
| :----------------------------------------------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src='https://user-images.githubusercontent.com/48054808/130388598-1e2b27e7-be66-49df-84d5-57b4dc7730d6.png' width='300'/> | <img src='https://user-images.githubusercontent.com/48054808/130388606-78a3a682-2ae4-4753-a07c-671a46930de8.png' width='300'/> | <img src='https://user-images.githubusercontent.com/48054808/130388615-b04197b3-2fdf-4494-ad17-490afe0fd1cd.png' width='300'/> | <img src='https://user-images.githubusercontent.com/48054808/130388623-2eec0cca-fee1-47f0-8398-cae0171aa7a5.png' width='300'/> | <img src='https://user-images.githubusercontent.com/48054808/130388624-f27d0712-ba71-42b2-ada4-44bf60e36512.png' width='300'/> |

4个风格图像下载地址如下：
| [StarryNew](https://user-images.githubusercontent.com/79366697/118655415-1ec8c000-b81c-11eb-8002-90bf8d477860.png) | [Stars](https://user-images.githubusercontent.com/79366697/118655423-20928380-b81c-11eb-92bd-0deeb320ff14.png) | [Ocean](https://user-images.githubusercontent.com/79366697/118655407-1c666600-b81c-11eb-83a6-300ee1952415.png) | [Circuit](https://user-images.githubusercontent.com/79366697/118655399-196b7580-b81c-11eb-8bc5-d5ece80c18ba.jpg)|

只需运行下面的代码即可迁移至指定风格：

```
python applications/tools/lapstyle.py --content_img_path ${PATH_OF_CONTENT_IMG} --style_image_path ${PATH_OF_STYLE_IMG}
```
### **参数**

- `--content_img_path (str)`: 输入的内容图像路径。
- `--style_image_path (str)`: 输入的风格图像路径。
- `--output_path (str)`: 输出的图像路径，默认为`output_dir`。
- `--weight_path (str)`: 模型权重路径，设置`None`时会自行下载预训练模型，默认为`None`。
- `--style (str)`: 生成图像风格，当`weight_path`为`None`时，可以在`starrynew`, `circuit`, `ocean` 和 `stars`中选择，默认为`starrynew`。

## 3. 模型训练

配置文件参数详情：[Config文件使用说明](../config_doc.md)

### 3.1 数据准备

为了训练LapStyle，我们使用COCO数据集作为内容图像数据集。您可以从[starrynew](https://user-images.githubusercontent.com/79366697/118655415-1ec8c000-b81c-11eb-8002-90bf8d477860.png)，[ocean](https://user-images.githubusercontent.com/79366697/118655407-1c666600-b81c-11eb-83a6-300ee1952415.png)，[stars](https://user-images.githubusercontent.com/79366697/118655423-20928380-b81c-11eb-92bd-0deeb320ff14.png)或[circuit](https://user-images.githubusercontent.com/79366697/118655399-196b7580-b81c-11eb-8bc5-d5ece80c18ba.jpg)中选择一张风格图片，也可以任意选择您喜欢的图片作为风格图片。在开始训练与测试之前，记得修改配置文件的数据路径。

### 3.2 训练

示例以COCO数据为例。如果您想使用自己的数据集，可以在配置文件中修改数据集为您自己的数据集。

<div align="center">
  <img src="https://user-images.githubusercontent.com/48054808/130389113-981ef29c-12ac-4fcd-9a2c-c7fcbc7031be.png" width="300"/>
</div>



**注意，LapStyle模型训练暂时不支持Windows系统。**

(1) 首先在128*128像素下训练LapStyle的绘图网络（**Drafting Network**）:
```
python -u tools/main.py --config-file configs/lapstyle_draft.yaml
```

(2) 然后，在256*256像素下训练LapStyle的修正网络（**Revision Network**）:
```
python -u tools/main.py --config-file configs/lapstyle_rev_first.yaml --load ${PATH_OF_LAST_STAGE_WEIGHT}
```

(3) 最后，在512*512像素下再次训练LapStyle的修正网络（**Revision Network**）:
```
python -u tools/main.py --config-file configs/lapstyle_rev_second.yaml --load ${PATH_OF_LAST_STAGE_WEIGHT}
```

### 3.3 测试

测试时需要将配置文件中的`validate/save_img`参数改成`True`以保存输出图像。
测试训练好的模型，您可以直接测试 "lapstyle_rev_second"，因为它包含了之前步骤里的训练权重：
```
python tools/main.py --config-file configs/lapstyle_rev_second.yaml --evaluate-only --load ${PATH_OF_WEIGHT}
```

## 4. 结果展示

| Style | Stylized Results |
| --- | --- |
| ![starrynew](https://user-images.githubusercontent.com/79366697/118655415-1ec8c000-b81c-11eb-8002-90bf8d477860.png) | ![chicago_stylized_starrynew](https://user-images.githubusercontent.com/79366697/118655671-59325d00-b81c-11eb-93a3-4fcc24680124.png)|
| ![ocean](https://user-images.githubusercontent.com/79366697/118655407-1c666600-b81c-11eb-83a6-300ee1952415.png) | ![chicago_ocean_512](https://user-images.githubusercontent.com/79366697/118655625-4cae0480-b81c-11eb-83ec-30936ed3df65.png)|
| ![stars](https://user-images.githubusercontent.com/79366697/118655423-20928380-b81c-11eb-92bd-0deeb320ff14.png) | ![chicago_stylized_stars_512](https://user-images.githubusercontent.com/79366697/118655638-50da2200-b81c-11eb-9223-58d5df022fa5.png)|
| ![circuit](https://user-images.githubusercontent.com/79366697/118655399-196b7580-b81c-11eb-8bc5-d5ece80c18ba.jpg) | ![chicago_stylized_circuit](https://user-images.githubusercontent.com/79366697/118655660-56376c80-b81c-11eb-87f2-64ae5a82375c.png)|


## 5. 模型下载

PaddleGAN中提供四个风格的预训练模型下载：

| 模型 | 风格 | 下载地址 |
|---|---|---|
| lapstyle_circuit  | circuit | [lapstyle_circuit](https://paddlegan.bj.bcebos.com/models/lapstyle_circuit.pdparams)
| lapstyle_ocean  | ocean | [lapstyle_ocean](https://paddlegan.bj.bcebos.com/models/lapstyle_ocean.pdparams)
| lapstyle_starrynew  | starrynew | [lapstyle_starrynew](https://paddlegan.bj.bcebos.com/models/lapstyle_starrynew.pdparams)
| lapstyle_stars  | stars | [lapstyle_stars](https://paddlegan.bj.bcebos.com/models/lapstyle_stars.pdparams)


# References

```
@article{lin2021drafting,
  title={Drafting and Revision: Laplacian Pyramid Network for Fast High-Quality Artistic Style Transfer},
  author={Lin, Tianwei and Ma, Zhuoqi and Li, Fu and He, Dongliang and Li, Xin and Ding, Errui and Wang, Nannan and Li, Jie and Gao, Xinbo},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```
