# PSGAN

## 1. PSGAN原理
PSGAN模型的任务是妆容迁移， 即将任意参照图像上的妆容迁移到不带妆容的源图像上。很多人像美化应用都需要这种技术。近来的一些妆容迁移方法大都基于生成对抗网络（GAN）。它们通常采用 CycleGAN 的框架，并在两个数据集上进行训练，即无妆容图像和有妆容图像。但是，现有的方法存在一个局限性：只在正面人脸图像上表现良好，没有为处理源图像和参照图像之间的姿态和表情差异专门设计模块。PSGAN是一种全新的姿态稳健可感知空间的生生成对抗网络。PSGAN 主要分为三部分：妆容提炼网络（MDNet）、注意式妆容变形（AMM）模块和卸妆-再化妆网络（DRNet）。这三种新提出的模块能让 PSGAN 具备上述的完美妆容迁移模型所应具备的能力。
![](../imgs/psgan_arc.png)

## 2. 使用方法
### 2.1 测试
运行如下命令，就可以完成妆容迁移，程序运行成功后，会在当前文件夹生成妆容迁移后的图片文件。本项目中提供了原始图片和参考供展示使用，具体命令如下所示：

```
cd applications/
python tools/ps_demo.py \  
  --config-file configs/makeup.yaml \
  --model_path /your/model/path \
  --source_path  /your/source/image/path  \
  --reference_dir /your/ref/image/path
```
** 参数说明: **
- config-file: PSGAN网络到参数配置文件，格式为yaml
- model_path: 训练完成保存下来网络权重文件的路径
- source_path: 未化妆的原始图片文件全路径，包含图片文件名字
- reference_dir: 化妆的参考图片文件路径，不包含图片文件名字

### 2.2 训练
1. 从百度网盘下载原始换妆数据[data](https://pan.baidu.com/s/1ZF-DN9PvbBteOSfQodWnyw)(密码:rtdd)到PaddleGAN文件夹, 并解压。然后下载landmarks数据[lmks]()，解压后的landmarks文件夹替换原始换妆数据中的landmarks文件夹, train_makeup.txt文件替换原始换妆数据集中的makeup.txt文件, train_non-makeup.txt文件替换原始换妆数据集中的non-makeup.txt文件
2. `python tools/main.py --config-file configs/makeup.yaml`

注意：训练时makeup.yaml文件中`isTrain`参数值为`True`, 测试时修改该参数值为`False`.

### 2.3 模型
---|:--:|:--:|:--:|:--:|:--:
Model|Dataset|BatchSize|Inference speed|Download
PSGAN|MT-Dataset| 1 | -- | [model]()

## 3. 妆容迁移结果展示
![](../imgs/makeup_shifter.png)
