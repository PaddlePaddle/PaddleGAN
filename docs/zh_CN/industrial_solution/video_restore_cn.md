# 智能影像修复

PaddleGAN提供一系列影像修复能力，包括 **[图片上色](./photo_color_cn.md)、[视频上色](./video_color_cn.md)、[图片分辨率提升](./photo_sr_cn.md)、[视频分辨率提升](./video_sr_cn.md)**，以及 **[视频流畅度提升](./video_frame_cn.md)**（提高视频播放流畅度）三大功能，使得历史影像恢复往日鲜活的色彩，清晰流畅的呈现于我们眼前。

在未来，PaddleGAN也将不断补充与优化影像修复的能力，比如增加去噪、图像修复等功能，还请大家敬请期待！

## **一行代码快速进行影像修复**

```
cd applications
python tools/video-enhance.py --input you_video_path.mp4 --process_order DAIN DeOldify PPMSVSR --output output_dir
```

### **参数**

- `--input (str)`: 输入的视频路径。
- `--output (str)`: 输出的视频路径。
- `--process_order`: 调用的模型名字和顺序，比如输入为 `DAIN DeOldify PPMSVSR`，则会顺序调用 `DAINPredictor` `DeOldifyPredictor` `PPMSVSRPredictor` 。

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/117925494-e9a70400-b329-11eb-9f38-a48ef946a3a4.gif' width='600'>
</div>

## 详细教程
* 视频修复
  * [视频上色](./video_color_cn.md)
  * [视频分辨率提升](./video_sr_cn.md)
  * [视频流畅度提升](./video_frame_cn.md)

* 照片修复
  * [图片上色](./photo_color_cn.md)
  * [图片分辨率提升](./photo_sr_cn.md)


## 在线体验
为了让大家快速体验影像修复的能力，PaddleGAN在飞桨人工智能学习与实训平台AI Studio准备了完整的实现步骤及详细代码，同时，AI Studio还为大家准备了免费的GPU算力，大家登录即可亲自实践 **[老北京城影像修复](https://aistudio.baidu.com/aistudio/projectdetail/1161285)** 的项目，快上手体验吧！

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/117924001-a0ee4b80-b327-11eb-8ab8-189f4afb8c23.png'>
</div>
