# 视频补帧

针对老视频的流畅度提升，PaddleGAN提供了[DAIN](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/apis/apps.md#ppganappsdainpredictor)模型接口。

## DAIN

[DAIN](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/apis/apps.md#ppganappsdainpredictor)模型通过探索深度的信息来显式检测遮挡。并且开发了一个深度感知的流投影层来合成中间流。在视频补帧方面有较好的效果。

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/117925889-76ea5880-b32a-11eb-9917-17cea27b64d0.png'>
</div>

```
ppgan.apps.DAINPredictor(
                        output='output',
                        weight_path=None,
                        time_step=None,
                        use_gpu=True,
                        remove_duplicates=False)
```
### 参数

- `output (str，可选的)`: 输出的文件夹路径，默认值：`output`.
- `weight_path (None，可选的)`: 载入的权重路径，如果没有设置，则从云端下载默认的权重到本地。默认值：`None`。
- `time_step (int)`: 补帧的时间系数，如果设置为0.5，则原先为每秒30帧的视频，补帧后变为每秒60帧。
- `remove_duplicates (bool，可选的)`: 是否删除重复帧，默认值：`False`.

### 使用方式
**1. API预测**

除了定义输入视频路径外，此接口还需定义time_step，同时，目前API预测方式只支持在静态图下运行，需加上启动静态图命令，后续会支持动态图，敬请期待~

```
paddle.enable_static()

from ppgan.apps import DAINPredictor
dain = DAINPredictor(output='output', time_step=0.5)
# 测试一个视频文件
dain.run("/home/aistudio/Peking_input360p_clip6_5s.mp4",)
paddle.disable_static()

paddle.disable_static()

```

**2. 命令行预测**

```
!python applications/tools/video-enhance.py --input /home/aistudio/Peking_input360p_clip6_5s.mp4 \ #原视频路径
                               --process_order DAIN \
                               --output output_dir #成品视频所在的路径
```
### 在线项目体验
**1. [老北京城影像修复](https://aistudio.baidu.com/aistudio/projectdetail/1161285)**
