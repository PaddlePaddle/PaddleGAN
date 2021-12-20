# TensorRT预测部署教程
TensorRT是NVIDIA提出的用于统一模型部署的加速库，可以应用于V100、JETSON Xavier等硬件，它可以极大提高预测速度。Paddle TensorRT教程请参考文档[使用Paddle-TensorRT库预测](https://paddle-inference.readthedocs.io/en/latest/optimize/paddle_trt.html#)

## 1. 安装PaddleInference预测库
- Python安装包，请从[这里](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/Tables.html#whl-release) 下载带有tensorrt的安装包进行安装

- CPP预测库，请从[这里](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/build_and_install_lib_cn.html) 下载带有TensorRT编译的预测库

- 如果Python和CPP官网没有提供已编译好的安装包或预测库，请参考[源码安装](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/linux-compile.html) 自行编译

**注意：**
- 您的机器上TensorRT的版本需要跟您使用的预测库中TensorRT版本保持一致。
- PaddleGAN中部署预测要求TensorRT版本 > 7.0。

## 2. 导出模型
模型导出具体请参考文档[PaddleGAN模型导出教程](../EXPORT_MODEL.md)。

## 3. 开启TensorRT加速
### 3.1 配置TensorRT
在使用Paddle预测库构建预测器配置config时，打开TensorRT引擎就可以了：

```
config->EnableUseGpu(100, 0); // 初始化100M显存，使用GPU ID为0
config->GpuDeviceId();        // 返回正在使用的GPU ID
// 开启TensorRT预测，可提升GPU预测性能，需要使用带TensorRT的预测库
config->EnableTensorRtEngine(1 << 20             /*workspace_size*/,
                             batch_size        /*max_batch_size*/,
                             3                 /*min_subgraph_size*/,
                             AnalysisConfig::Precision::kFloat32 /*precision*/,
                             false             /*use_static*/,
                             false             /*use_calib_mode*/);

```

### 3.2 TensorRT固定尺寸预测

以`msvsr`为例，使用固定尺寸输入预测：
```
python tools/inference.py --model_path=/root/to/model --config-file /root/to/config --run_mode trt_fp32 --min_subgraph_size  20 --mode_type msvsr
```

## 4、常见问题QA
**Q:** 提示没有`tensorrt_op`</br>
**A:** 请检查是否使用带有TensorRT的Paddle Python包或预测库。

**Q:** 提示`op out of memory`</br>
**A:** 检查GPU是否是别人也在使用，请尝试使用空闲GPU

**Q:** 提示`some trt inputs dynamic shape info not set`</br>
**A:** 这是由于`TensorRT`会把网络结果划分成多个子图，我们只设置了输入数据的动态尺寸，划分的其他子图的输入并未设置动态尺寸。有两个解决方法：

- 方法一：通过增大`min_subgraph_size`，跳过对这些子图的优化。根据提示，设置min_subgraph_size大于并未设置动态尺寸输入的子图中OP个数即可。
`min_subgraph_size`的意思是，在加载TensorRT引擎的时候，大于`min_subgraph_size`的OP才会被优化，并且这些OP是连续的且是TensorRT可以优化的。

- 方法二：找到子图的这些输入，按照上面方式也设置子图的输入动态尺寸。

**Q:** 如何打开日志</br>
**A:** 预测库默认是打开日志的，只要注释掉`config.disable_glog_info()`就可以打开日志

**Q:** 开启TensorRT，预测时提示Slice on batch axis is not supported in TensorRT</br>
**A:** 请尝试使用动态尺寸输入
