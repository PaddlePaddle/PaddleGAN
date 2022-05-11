[English](../../en_US/tutorials/gpen.md) | 中文

## GPEN 盲人脸修复模型


## 1、简介

GPEN模型是一个盲人脸修复模型。作者将前人提出的 StyleGAN V2 的解码器嵌入模型，作为GPEN的解码器；用DNN重新构建了一种简单的编码器，为解码器提供输入。这样模型在保留了 StyleGAN V2 解码器优秀的性能的基础上，将模型的功能由图像风格转换变为了盲人脸修复。模型的总体结构如下图所示：

![img](../../imgs/gpen_1.jpg)

对模型更详细的介绍，和参考repo可查看以下AI Studio项目[链接]([GPEN盲人脸修复模型复现 - 飞桨AI Studio (baidu.com)](https://aistudio.baidu.com/aistudio/projectdetail/3936241?contributionType=1))的最新版本。




## 2、准备工作

### 2.1 数据集准备

GPEN模型训练集是经典的FFHQ人脸数据集,共70000张1024 x 1024高分辨率的清晰人脸图片，测试集是CELEBA-HQ数据集，共2000张高分辨率人脸图片。详细信息可以参考**数据集网址:** [FFHQ](https://github.com/NVlabs/ffhq-dataset) ，[CELEBA-HQ](https://github.com/tkarras/progressive_growing_of_gans) 。以下给出了具体的下载链接：

**原数据集下载地址：**

**FFHQ ：**           https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL?usp=drive_open

**CELEBA-HQ：** https://drive.google.com/drive/folders/0B4qLcYyJmiz0TXY1NG02bzZVRGs?resourcekey=0-arAVTUfW9KRhN-irJchVKQ&usp=sharing



由于FFHQ原数据集过大，也可以从以下链接下载256分辨率的FFHQ数据集：

[Flickr-Faces-HQ Dataset (FFHQ) 256x256 - 飞桨AI Studio (baidu.com)](https://aistudio.baidu.com/aistudio/datasetdetail/111879)



**下载后，文件组织形式如下**

```
|-- data/GPEN
	|-- train
		|-- 00000
			|-- 00000.png
			|-- 00001.png
			|-- ......
			|-- 00999.png
		|-- 01000
			|-- ......
		|-- ......
		|-- 69000
            |-- ......
                |-- 69999.png
	|-- test
		|-- 2000张png图片
```



### 2.2 模型准备

**模型参数文件及训练日志下载地址：**

链接：https://pan.baidu.com/s/1MORzH2C58xlAXrMI5MyJow    提取码：pyda 


从链接中下载模型参数,并放到项目根目录下的data/gpen/weights文件夹下，具体文件结构如下所示：

**文件结构**


```
data/gpen/weights
    |-- model_ir_se50_2.pdparams #计算id_loss需要加载的facenet的模型参数文件
    |-- weight_pretrain.pdparams #256分辨率的包含生成器和判别器的模型参数文件，其中只有生成器的参数是训练好的参数，参                                  #数文件的格式与3.1训练过程中保存的参数文件格式相同。3.2、3.3.1、4.1也需要用到该参数文件
    |-- g_ema.pdparams           #256分辨率的仅包含生成器模型参数文件，与3.3.1中生成的参数文件格式相同，在3.3.2中用到
```



## 3、开始使用

### 3.1 模型训练

在控制台输入以下代码，开始训练：

 ```shell
 python tools/main.py -c configs/gpen_256_ffhq.yaml
 ```



### 3.2 模型评估

对模型进行评估时，在控制台输入以下代码，下面代码中使用上面提到的下载的模型参数：

 ```shell
python tools/main.py -c configs/gpen_256_ffhq.yaml -o dataset.test.amount=2000 --load data/gpen/weights/weight_pretrain.pdparams --evaluate-only
 ```

如果要在自己提供的模型上进行测试，请修改 --load  后面的路径。



### 3.3 模型预测

#### 3.3.1 导出生成器权重

训练结束后，需要使用 ``tools/extract_weight.py`` 来从训练模型（包含了生成器和判别器）中提取生成器的权重来给`applications/tools/gpen.py`进行推理，以实现GPEN模型的各种应用。输入以下命令来提取生成器的权重：

```bash
python tools/extract_weight.py data/gpen/weights/weight_pretrain.pdparams --net-name g_ema --output data/gpen/weights/g_ema.pdparams
```



#### 3.3.2 对单张图像进行处理

提取完生成器的权重后，输入以下命令可对--test_img路径下图片进行测试。修改--seed参数，可生成不同的退化图像，展示出更丰富的效果。可修改--test_img后的路径为你想测试的任意图片。

```bash
python applications/tools/gpen.py --test_img data/gpen/lite_data/15006.png --seed=100 --weight_path data/gpen/weights/g_ema.pdparams
```

以下是样例图片和对应的修复图像，从左到右依次是退化图像、生成的图像和原始清晰图像：

<p align='center'>
<img src="../../imgs/gpen_2.png" height="256px" width='768px' >




输出示例如下:

```
result saved in : output_dir/gpen_predict.png
        FID: 92.11730631094356
        PSNR:19.014782083825743
```



## 4. Tipc

### 4.1 导出inference模型

```bash
python tools/export_model.py -c configs/gpen_256_ffhq.yaml --inputs_size=1,3,256,256 --load data/gpen/weights/weight_pretrain.pdparams
```

上述命令将生成预测所需的模型结构文件`gpenmodel_g_ema.pdmodel`和模型权重文件`gpenmodel_g_ema.pdiparams`以及`gpenmodel_g_ema.pdiparams.info`文件，均存放在`inference_model/`目录下。也可以修改--load 后的参数为你想测试的模型参数文件。



### 4.2 使用预测引擎推理

```bash
python tools/inference.py --model_type GPEN --seed 100 -c configs/gpen_256_ffhq.yaml -o dataset.test.dataroot="./data/gpen/lite_data/" --output_path test_tipc/output/ --model_path inference_model/gpenmodel_g_ema
```

推理结束会默认保存下模型生成的修复图像在test_tipc/output/GPEN目录下，并载test_tipc/output/GPEN/metric.txt中输出测试得到的FID值。


默认输出如下:

```
Metric fid: 187.0158
```

注：由于对高清图片进行退化的操作具有一定的随机性，所以每次测试的结果都会有所不同。为了保证测试结果一致，在这里我固定了随机种子，使每次测试时对图片都进行相同的退化操作。



### 4.3 调用脚本两步完成训推一体测试

测试基本训练预测功能的`lite_train_lite_infer`模式，运行：

```shell
# 准备数据
bash test_tipc/prepare.sh ./test_tipc/configs/GPEN/train_infer_python.txt 'lite_train_lite_infer'
# 运行测试
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/GPEN/train_infer_python.txt 'lite_train_lite_infer'
```



## 5. LICENSE

本项目的发布受[Apache 2.0 license](https://github.com/PaddlePaddle/models/blob/release/2.2/community/repo_template/LICENSE)许可认证。



## 7、参考文献与链接

论文地址：https://paperswithcode.com/paper/gan-prior-embedded-network-for-blind-face

参考repo Github：https://github.com/yangxy/GPEN

论文复现指南-CV方向：https://github.com/PaddlePaddle/models/blob/release%2F2.2/tutorials/article-implementation/ArticleReproduction_CV.md

readme文档模板：https://github.com/PaddlePaddle/models/blob/release/2.2/community/repo_template/README.md
