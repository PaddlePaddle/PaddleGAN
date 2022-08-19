# C++预测功能测试

C++预测功能测试的主程序为`test_inference_cpp.sh`，可以测试基于C++预测库的模型推理功能。

## 1. 测试结论汇总

| 模型类型 |device | batchsize | tensorrt | mkldnn | cpu多线程 |
|  :----:   |  :----: |   :----:   |  :----:  |   :----:   |  :----:  |
| 正常模型 | GPU | 1 | - | - | - |
| 正常模型 | CPU | 1 | - | fp32 | 支持 |

## 2. 测试流程
运行环境配置请参考[文档](../../docs/zh_CN/install.md)的内容安装PaddleGAN，TIPC推荐的环境：
- PaddlePaddle=2.3.1
- CUDA=10.2
- cuDNN=7.6.5

### 2.1 功能测试
先运行`prepare.sh`准备数据和模型，然后运行`test_inference_cpp.sh`进行测试，msvsr模型的具体测试如下

```bash
# 准备模型和数据
bash test_tipc/test_inference_cpp.sh test_tipc/configs/msvsr/inference_cpp.txt
# cpp推理测试，可修改inference_cpp.txt配置累测试不同配置下的推理结果
bash test_tipc/test_inference_cpp.sh test_tipc/configs/msvsr/inference_cpp.txt
```

运行预测指令后，在`test_tipc/output`文件夹下自动会保存运行日志和输出结果，包括以下文件：

```shell
test_tipc/output
    ├── infer_cpp/results_cpp_infer.log    # 运行指令状态的日志
    ├── infer_cpp/infer_cpp_GPU.log        # 使用GPU推理测试的日志
    ├── infer_cpp/infer_cpp_CPU_use_mkldnn_threads_1.log    # 使用CPU开启mkldnn，thread为1的推理测试日志
    ├── output.mp4     # 视频超分预测结果
......
```
其中results_cpp_infer.log中包含了每条指令的运行状态，如果运行成功会输出：

```
Run successfully with command - ./deploy/cpp_infer/build/vsr --model_path=./inference/msvsr/multistagevsrmodel_generator.pdmodel --param_path=./inference/msvsr/multistagevsrmodel_generator.pdiparams --video_path=./data/low_res.mp4 --output_dir=./test_tipc/output/msvsr --frame_num=2 --device=GPU --gpu_id=1 --use_mkldnn=True --cpu_threads=1 > ./test_tipc/output/infer_cpp/infer_cpp_GPU.log 2>&1!
......
```
如果运行失败，会输出：
```
Run failed with command - ./deploy/cpp_infer/build/vsr --model_path=./inference/msvsr/multistagevsrmodel_generator.pdmodel --param_path=./inference/msvsr/multistagevsrmodel_generator.pdiparams --video_path=./data/low_res.mp4 --output_dir=./test_tipc/output/msvsr --frame_num=2 --device=GPU --gpu_id=1 --use_mkldnn=True --cpu_threads=1 > ./test_tipc/output/infer_cpp/infer_cpp_GPU.log 2>&1!
......
```
可以根据results_cpp_infer.log中的内容判定哪一个指令运行错误。
