# Applications接口说明

ppgan.apps包含超分、插针、上色、换妆、图像动画生成、人脸解析等应用，接口使用简洁，并内置了已训练好的模型，可以直接用来做应用。

## 公共用法

### CPU和GPU的切换

默认情况下，如果是GPU设备、并且安装了PaddlePaddle的GPU环境包，则默认使用GPU进行推理。否则，如果安装的是CPU环境包，则使用CPU进行推理。如果需要手动切换CPU、GPU，可以通过以下方式:


```
import paddle
paddle.set_device('cpu')
#paddle.set_device('gpu')

# from ppgan.apps import DeOldifyPredictor
# deoldify = DeOldifyPredictor()
# deoldify.run("docs/imgs/test_old.jpeg")
```

## ppgan.apps.DeOldifyPredictor

```python
ppgan.apps.DeOldifyPredictor(output='output', weight_path=None, render_factor=32)
```

> 构建DeOldify实例。DeOldify是一个基于GAN的老照片上色模型。该接口可以对图片或视频做上色。建议视频使用mp4格式。
>
> **示例**
>
> ```python
> from ppgan.apps import DeOldifyPredictor
> deoldify = DeOldifyPredictor()
> deoldify.run("docs/imgs/test_old.jpeg")
> ```

> **参数**
>
> > - output (str):  设置输出图片的保存路径，默认是output。注意，保存路径为设置output/DeOldify。
> > - weight_path (str): 指定模型路径，默认是None，则会自动下载内置的已经训练好的模型。
> > - artistic (bool): 是否使用偏"艺术性"的模型。"艺术性"的模型有可能产生一些有趣的颜色，但是毛刺比较多。
> > - render_factor (int): 图片渲染上色时的缩放因子，图片会缩放到边长为16xrender_factor的正方形， 再上色，例如render_factor默认值为32，输入图片先缩放到(16x32=512) 512x512大小的图片。通常来说，render_factor越小，计算速度越快，颜色看起来也更鲜活。较旧和较低质量的图像通常会因降低渲染因子而受益。渲染因子越高，图像质量越好，但颜色可能会稍微褪色。

### run

```python
run(input)
```

> 构建实例后的执行接口。

> **参数**
>
> > - input (str|np.ndarray|Image.Image): 输入的图片或视频文件。如果是图片，可以是图片的路径、np.ndarray、或PIL.Image类型。如果是视频，只能是视频文件路径。
> >
>
> **返回值**
>
> > - tuple(pred_img(np.array), out_paht(str)): 当属输入时图片时，返回预测后的图片，类型PIL.Image，以及图片的保存的路径。
> > - tuple(frame_path(str), out_path(str)): 当输入为视频时，frame_path为视频每帧上色后保存的图片路径，out_path为上色后视频的保存路径。

### run_image

```python
run_image(img)
```

> 图片上色的接口。

> **参数**
>
> > - img (str|np.ndarray|Image.Image): 输入图片，可以是图片的路径、np.ndarray、或PIL.Image类型。
> >
>
> **返回值**
>
> > - pred_img(PIL.Image): 返回预测后的图片，为PIL.Image类型。

### run_video

```python
run_video(video)
```

> 视频上色的接口。

> **参数**
>
> > - Video (str): 输入视频文件的路径。
>
> **返回值**
>
> > - tuple(frame_path(str), out_path(str)):  frame_path为视频每帧上色后保存的图片路径，out_path为上色后视频的保存路径。



## ppgan.apps.DeepRemasterPredictor

```python
ppgan.apps.DeepRemasterPredictor(output='output', weight_path=None, colorization=False, reference_dir=None, mindim=360)
```

> 构建DeepRemasterPredictor实例。DeepRemaster是一个基于GAN的老照片/视频修复、上色模型，该模型可以提供一个参考色的图片作为输入。该接口目前只支持视频输入，建议使用mp4格式。
>
> **示例**
>
> ```
> from ppgan.apps import DeepRemasterPredictor
> deep_remaster = DeepRemasterPredictor()
> deep_remaster.run("docs/imgs/test_old.jpeg")
> ```
>
>

> **参数**
>
> > - output (str):  设置输出图片的保存路径，默认是output。注意，保存路径为设置output/DeepRemaster。
> > - weight_path (str): 指定模型路径，默认是None，则会自动下载内置的已经训练好的模型。
> > - colorization (bool):  是否打开上色功能，默认是False，既不打开，只执行修复功能。
> > - reference_dir(str|None): 打开上色功能时，输入参考色图片路径，也可以不设置参考色图片。
> > - mindim(int):  预测前图片会进行缩放，最小边长度。

### run

```python
run(video_path)
```

> 构建实例后的执行接口。

> **参数**
>
> > - video_path (str): 输入视频文件路径。
> >
> > 返回值
> >
> > - tuple(str, str)): 返回两个str类型，前者是视频上色后每帧图片的保存路径，后者是上色之后的视频保存路径。



## ppgan.apps.RealSRPredictor

```python
ppgan.apps.RealSRPredictor(output='output', weight_path=None)
```

> 构建RealSR实例。RealSR: Real-World Super-Resolution via Kernel Estimation and Noise Injection发表于CVPR 2020 Workshops的基于真实世界图像训练的超分辨率模型。此接口对输入图片或视频做4倍的超分辨率。建议视频使用mp4格式。
>
> **用例**
>
> ```
> from ppgan.apps import RealSRPredictor
> sr = RealSRPredictor()
> sr.run("docs/imgs/test_sr.jpeg")
> ```

> **参数**
>
> > - output (str):  设置输出图片的保存路径，默认是output。注意，保存路径为设置output/RealSR。
> > - weight_path (str): 指定模型路径，默认是None，则会自动下载内置的已经训练好的模型。

```python
run(video_path)
```

> 构建实例后的执行接口。

> **参数**
>
> > - video_path (str): 输入视频文件路径。
> >
>
> **返回值**
>
> > - tuple(pred_img(np.array), out_paht(str)): 当属输入时图片时，返回预测后的图片，类型PIL.Image，以及图片的保存的路径。
> > - tuple(frame_path(str), out_path(str)): 当输入为视频时，frame_path为超分后视频每帧图片的保存路径，out_path为超分后的视频保存路径。

### run_image

```python
run_image(img)
```

> 图片超分的接口。

> **参数**
>
> > - img (str|np.ndarray|Image.Image): 输入图片，可以是图片的路径、np.ndarray、或PIL.Image类型。
>
> **返回值**
>
> > - pred_img(PIL.Image): 返回预测后的图片，为PIL.Image类型。

### run_video

```python
run_video(video)
```

> 视频超分的接口。

> **参数**
>
> > - Video (str): 输入视频文件的路径。
>
> **返回值**
>
> > - tuple(frame_path(str), out_path(str)):  frame_path为超分后视频每帧图片的保存路径，out_path为超分后的视频保存路径。



## ppgan.apps.EDVRPredictor

```python
ppgan.apps.EDVRPredictor(output='output', weight_path=None)
```

> 构建RealSR实例。EDVR: Video Restoration with Enhanced Deformable Convolutional Networks，论文链接: https://arxiv.org/abs/1905.02716  ，是一个针对视频超分的模型。该接口，对视频做2倍的超分。建议视频使用mp4格式。
>
> **示例**
>
> ```
> from ppgan.apps import EDVRPredictor
> sr = EDVRPredictor()
> # 测试一个视频文件
> sr.run("docs/imgs/test.mp4")
> ```

> **参数**
>
> > - output (str):  设置输出图片的保存路径，默认是output。注意，保存路径为设置output/EDVR。
> > - weight_path (str): 指定模型路径，默认是None，则会自动下载内置的已经训练好的模型。

```python
run(video_path)
```

> 构建实例后的执行接口。

> **参数**
>
> > - video_path (str): 输入视频文件路径。
>
> **返回值**
>
> > - tuple(str, str): 前者超分后的视频每帧图片的保存路径，后者为做完超分的视频路径。



## ppgan.apps.DAINPredictor

```python
ppgan.apps.DAINPredictor(output='output', weight_path=None，time_step=None, use_gpu=True, key_frame_thread=0，remove_duplicates=False)
```

> 构建插帧DAIN模型的实例。DAIN: Depth-Aware Video Frame Interpolation，论文链接: https://arxiv.org/abs/1904.00830 ，对视频做插帧，获得帧率更高的视频。
>
> **示例**
>
> ```
> from ppgan.apps import DAINPredictor
> dain = DAINPredictor()
> # 测试一个视频文件
> dain.run("docs/imgs/test.mp4")
> ```

> **参数**
>
> > - output_path (str):  设置预测输出的保存路径，默认是output。注意，保存路径为设置output/DAIN。
> > - weight_path (str):  指定模型路径，默认是None，则会自动下载内置的已经训练好的模型。
> > - time_step (float): 帧率变化的倍数为 1./time_step，例如，如果time_step为0.5，则2倍插针，为0.25，则为4倍插帧。
> > - use_gpu (bool): 是否使用GPU做预测，默认是True。
> > - remove_duplicates (bool): 是否去除重复帧，默认是False。

```python
run(video_path)
```

> 构建实例后的执行接口。

> **参数**
>
> > - video_path (str): 输入视频文件路径。
>
> **返回值**
>
> > - tuple(str, str): 当输入为视频时，frame_path为视频每帧上色后保存的图片路径，out_path为上色后视频的保存路径。



## ppgan.apps.FirstOrderPredictor

```python
ppgan.apps.FirstOrderPredictor(output='output', weight_path=None，config=None, relative=False, adapt_scale=False，find_best_frame=False, best_frame=None)
```

> 构建FirsrOrder模型的实例，此模型用来做Image Animation，即给定一张源图片和一个驱动视频，生成一段视频，其中主体是源图片，动作是驱动视频中的动作。论文是First Order Motion Model for Image Animation，论文链接: https://arxiv.org/abs/2003.00196 。
>
> **示例**
>
> ```
> from ppgan.apps import FirstOrderPredictor
> animate = FirstOrderPredictor()
> # 测试一个视频文件
> animate.run("source.png"，"driving.mp4")
> ```

> **参数**
>
> > - output_path (str):  设置预测输出的保存路径，默认是output。注意，保存路径为设置output/result.mp4。
> > - weight_path (str):  指定模型路径，默认是None，则会自动下载内置的已经训练好的模型。
> > - config (dict|str|None): 设置模型的参数，可以是字典类型或YML文件，默认值是None，采用的默认的参数。当权重默认是None时，config也需采用默认值None。否则，这里的配置和对应权重保持一致
> > - relative (bool):  使用相对还是绝对关键点坐标，默认是False。
> > - adapt_scale (bool): 是否基于关键点凸包的自适应运动，默认是False。
> > - find_best_frame (bool): 是否从与源图片最匹配的帧开始生成，仅仅适用于人脸应用，需要人脸对齐的库。
> > - best_frame (int): 设置起始帧数，默认是None，从第1帧开始(从1开始计数)。

```python
run(source_image，driving_video)
```

> 构建实例后的执行接口，预测视频保存位置为output/result.mp4。

> **参数**
>
> > - source_image (str): 输入源图片。
> > - driving_video (str): 输入驱动视频，支持mp4格式。
>
> **返回值**
>
> > 无。

## ppgan.apps.FaceParsePredictor

```pyhton
ppgan.apps.FaceParsePredictor(output_path='output')
```
> 构建人脸解析模型实例，此模型用来做人脸解析， 即给定一个输入的人脸图像，人脸解析将为每个语义成分(如头发、嘴唇、鼻子、耳朵等)分配一个像素级标签。我们用BiseNet来完成这项任务。论文是 BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation, 论文链接: https://arxiv.org/abs/1808.00897v1.

> **参数:**
>
> > - input_image: 输入待解析的图片文件路径

> **示例:**
>
> ```
> from ppgan.apps import FaceParsePredictor
> parser = FaceParsePredictor()
> parser.run('docs/imgs/face.png')
> ```
> **返回值:**
> > - mask(numpy.ndarray): 返回解析完成的人脸成分mask矩阵, 数据类型为numpy.ndarray
