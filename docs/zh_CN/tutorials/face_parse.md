# 人脸解析

## 1. 人脸解析简介

人脸解析是语义图像分割的一种特殊情况，人脸解析是计算人脸图像中不同语义成分(如头发、嘴唇、鼻子、眼睛等)的像素级标签映射。给定一个输入的人脸图像，人脸解析将为每个语义成分分配一个像素级标签。我们利用BiseNet来解决这个问题。人脸解析工具在很多任务中都有应用，如识别，动画以及合成等。这个工具我们目前应用在换妆模型上。
## 2. 使用方法

### 2.1 测试

运行如下命令，可以完成人脸解析任务，程序运行成功后，会在`output`文件夹生成解析后的图片文件。具体命令如下所示：
```
cd applications
python tools/face_parse.py --input_image ../docs/imgs/face.png
```

**参数:**

- input_image: 输入待解析的图片文件路径

## 3. 结果展示
![](../../imgs/face_parse_out.png)

### 4. 参考文献

```
@misc{yu2018bisenet,
      title={BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation},
      author={Changqian Yu and Jingbo Wang and Chao Peng and Changxin Gao and Gang Yu and Nong Sang},
      year={2018},
      eprint={1808.00897},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
