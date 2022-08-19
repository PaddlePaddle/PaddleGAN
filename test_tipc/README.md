
# 飞桨训推一体认证

## 1. 简介

飞桨除了基本的模型训练和预测，还提供了支持多端多平台的高性能推理部署工具。本文档提供了PaddleGAN中所有模型的飞桨训推一体认证 (Training and Inference Pipeline Certification(TIPC)) 信息和测试工具，方便用户查阅每种模型的训练推理部署打通情况，并可以进行一键测试。

## 2. 汇总信息

打通情况汇总如下，已填写的部分表示可以使用本工具进行一键测试，未填写的表示正在支持中。

**字段说明：**
- 基础训练预测：包括模型训练、Paddle Inference Python预测。
- 更多训练方式：包括多机多卡、混合精度。
- 模型压缩：包括裁剪、离线/在线量化、蒸馏。
- 其他预测部署：包括Paddle Inference C++预测、Paddle Serving部署、Paddle-Lite部署等。

更详细的mkldnn、Tensorrt等预测加速相关功能的支持情况可以查看各测试工具的[更多教程](#more)。

| 算法论文 | 模型名称 | 模型类型 | 基础<br>训练预测 | 更多<br>训练方式 | 模型压缩 |  其他预测部署  |
| :--- | :--- |  :----:  | :--------: |  :----  |   :----  |   :----  |
| Pix2Pix |Pix2Pix | 生成  | 支持 | 多机多卡  | | |
| CycleGAN |CycleGAN | 生成  | 支持 | 多机多卡  | | |
| StyleGAN2 |StyleGAN2 | 生成  | 支持 | 多机多卡  | | |
| FOMM |FOMM | 生成  | 支持 | 多机多卡  | | |
| BasicVSR |BasicVSR | 超分  | 支持 | 多机多卡  | | |
|PP-MSVSR|PP-MSVSR | 超分|
|SinGAN|SinGAN | 生成| 支持 |




## 3. 一键测试工具使用
### 目录介绍

```shell
test_tipc/
├── configs/  # 配置文件目录
	├── basicvsr_reds.yaml             # 测试basicvsr模型训练的yaml文件
	├── cyclegan_horse2zebra.yaml      # 测试cyclegan模型训练的yaml文件
	├── firstorder_vox_256.yaml        # 测试fomm模型训练的yaml文件
	├── pix2pix_facedes.yaml           # 测试pix2pix模型训练的yaml文件
	├── stylegan_v2_256_ffhq.yaml      # 测试stylegan模型训练的yaml文件

	├── ...  
├── results/   # 预先保存的预测结果，用于和实际预测结果进行精读比对
	├── python_basicvsr_results_fp32.txt            # 预存的basicvsr模型python预测fp32精度的结果
	├── python_cyclegan_results_fp32.txt            # 预存的cyclegan模型python预测fp32精度的结果
	├── python_pix2pix_results_fp32.txt             # 预存的pix2pix模型python预测的fp32精度的结果
	├── python_stylegan2_results_fp32.txt           # 预存的stylegan2模型python预测的fp32精度的结果
	├── ...
├── prepare.sh                        # 完成test_*.sh运行所需要的数据和模型下载
├── test_train_inference_python.sh    # 测试python训练预测的主程序
├── compare_results.py                # 用于对比log中的预测结果与results中的预存结果精度误差是否在限定范围内
└── readme.md                         # 使用文档
```

### 测试流程
使用本工具，可以测试不同功能的支持情况，以及预测结果是否对齐，测试流程如下：

![img](https://user-images.githubusercontent.com/79366697/185377097-a0f852a8-2d78-45ae-84ba-ae71b799d738.png)

1. 运行prepare.sh准备测试所需数据和模型；
2. 运行要测试的功能对应的测试脚本`test_*.sh`，产出log，由log可以看到不同配置是否运行成功；
3. 用`compare_results.py`对比log中的预测结果和预存在results目录下的结果，判断预测精度是否符合预期（在误差范围内）。

其中，有4个测试主程序，功能如下：
- `test_train_inference_python.sh`：测试基于Python的模型训练、评估、推理等基本功能。


<a name="more"></a>
#### 更多教程
各功能测试中涉及混合精度、裁剪、量化等训练相关，及mkldnn、Tensorrt等多种预测相关参数配置，请点击下方相应链接了解更多细节和使用教程：  
- [test_train_inference_python 使用](docs/test_train_inference_python.md)  测试基于Python的模型训练、推理等基本功能。
- [test_inference_cpp 使用](docs/test_inference_cpp.md) 测试基于C++的模型推理功能。
