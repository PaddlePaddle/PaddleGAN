# Introduction of Prediction Interface

PaddleGAN（ppgan.apps）provides prediction APIs covering multiple applications, including super resolution, video frame interpolation, colorization, makeup shifter, image animation, face parsing, etc. The integral pre-trained high-performance models enable users' flexible and efficient usage and inference.

* Colorization:
  * [DeOldify](#ppgan.apps.DeOldifyPredictor)
  * [DeepRemaster](#ppgan.apps.DeepRemasterPredictor)
* Super Resolution:
  * [RealSR](#ppgan.apps.RealSRPredictor)
  * [PPMSVSR](#ppgan.apps.PPMSVSRPredictor)
  * [PPMSVSRLarge](#ppgan.apps.PPMSVSRLargePredictor)
  * [EDVR](#ppgan.apps.EDVRPredictor)
  * [BasicVSR](#ppgan.apps.BasicVSRPredictor)
  * [IconVSR](#ppgan.apps.IconVSRPredictor)
  * [BasiVSRPlusPlus](#ppgan.apps.BasiVSRPlusPlusPredictor)
* Video Frame Interpolation:
  * [DAIN](#ppgan.apps.DAINPredictor)
* Motion Driving:
  * [FirstOrder](#ppgan.apps.FirstOrderPredictor)
* Face:
  * [FaceFaceParse](#ppgan.apps.FaceParsePredictor)
* Image Animation:
  * [AnimeGAN](#ppgan.apps.AnimeGANPredictor)
* Lip-syncing:
  * [Wav2Lip](#ppgan.apps.Wav2LipPredictor)


## Public Usage

### Switch of CPU and GPU

By default, GPU devices with the [PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/windows-pip.html) GPU environment package installed conduct inference by using GPU. If the CPU environment package is installed, CPU is used for inference.

If manual switch of CPU and GPU is needed，you can do the following:


```
import paddle
paddle.set_device('cpu') #set as CPU
#paddle.set_device('gpu') #set as GPU
```

## ppgan.apps.DeOldifyPredictor

```python
ppgan.apps.DeOldifyPredictor(output='output', weight_path=None, render_factor=32)
```

> Build the instance of DeOldify. DeOldify is a coloring model based on GAN. The interface supports the colorization of images or videos. The recommended video format is mp4.
>
> **Example**
>
> ```python
> from ppgan.apps import DeOldifyPredictor
> deoldify = DeOldifyPredictor()
> deoldify.run("docs/imgs/test_old.jpeg")
> ```
> **Parameters**
>
> > - output (str): path of the output image, default: output. Note that the save path should be set as output/DeOldify.
> > - weight_path (str): path of the model, default: None，pre-trained integral model will then be automatically downloaded.
> > - artistic (bool): whether to use "artistic" model, which may produce interesting colors, but there are more glitches.
> > - render_factor (int): the zoom factor during image rendering and colorization. The image will be zoomed to a square with side length of 16xrender_factor before being colorized. For example, with a default value of 32，the entered image will  be resized to  (16x32=512) 512x512. Normally，the smaller the render_factor，the faster the computation and the more vivid the colors. Therefore, old images with low quality usually benefits from lowering the value of rendering factor. The higher the value, the better the image quality, but the color may fade slightly.
### run

```python
run(input)
```

> The execution interface after building the instance.
> **Parameters**
>
> > - input (str|np.ndarray|Image.Image): the input image or video files。For images, it could be its path, np.ndarray, or PIL.Image type. For videos, it could only be the file path.
>
>**Return Value**
>
>> - tuple(pred_img(np.array), out_paht(str)): for image input, return the predicted image, PIL.Image type and the path where the image is saved.
> > - tuple(frame_path(str), out_path(str)): for video input, frame_path is the save path of the images after colorizing each frame of the video, and out_path is the save path of the colorized video.
### run_image

```python
run_image(img)
```

> The interface of image colorization.
> **Parameters**
>
> > - img (str|np.ndarray|Image.Image): input image，it could be the path of the image, np.ndarray, or PIL.Image type.
>
>**Return Value**
>
>> - pred_img(PIL.Image): return the predicted image, PIL.Image type.
### run_video

```python
run_video(video)
```

> The interface of video colorization.
> **Parameters**
>
> > - Video (str): path of the input video files.
>
> **Return Value**
>
> > - tuple(frame_path(str), out_path(str)):  frame_path is the save path of the images after colorizing each frame of the video, and out_path is the save path of the colorized video.


## ppgan.apps.DeepRemasterPredictor

```python
ppgan.apps.DeepRemasterPredictor(output='output', weight_path=None, colorization=False, reference_dir=None, mindim=360)
```

> Build the instance of DeepRemasterPredictor. DeepRemaster is a GAN-based coloring and restoring model, which can provide input reference frames. Only video input is available now, and the recommended format is mp4.
>
> **Example**
>
> ```
> from ppgan.apps import DeepRemasterPredictor
> deep_remaster = DeepRemasterPredictor()
> deep_remaster.run("docs/imgs/test_old.jpeg")
> ```
>
>
> **Parameters**
>
> > - output (str): path of the output image, default: output. Note that the path should be set as output/DeepRemaster.
> > - weight_path (str): path of the model, default: None，pre-trained integral model will then be automatically downloaded.
> > - colorization (bool):  whether to enable the coloring function, default: False, only the restoring function will be executed.
> > - reference_dir(str|None): path of the reference frame when the coloring function is on, no reference frame is also allowed.
> > - mindim(int):  minimum side length of the resized image before prediction.
### run

```python
run(video_path)
```

> The execution interface after building the instance.
> **Parameters**
>
> > - video_path (str): path of the video file.
> >
> > **Return Value**
> >
> > - tuple(str, str)): return two types of str, the former is the save path of each frame of the colorized video, the latter is the save path of the colorized video.


## ppgan.apps.RealSRPredictor

```python
ppgan.apps.RealSRPredictor(output='output', weight_path=None)
```

> Build the instance of RealSR。RealSR, Real-World Super-Resolution via Kernel Estimation and Noise Injection, is launched by CVPR 2020 Workshops in its super resolution model based on real-world images training. The interface imposes 4x super resolution on the input image or video. The recommended video format is mp4.
>
> *Note: the size of the input image should be less than 1000x1000pix。
>
> **Example**
>
> ```
> from ppgan.apps import RealSRPredictor
> sr = RealSRPredictor()
> sr.run("docs/imgs/test_sr.jpeg")
> ```
> **Parameters**
>
> > - output (str):  path of the output image, default: output. Note that the path should be set as output/RealSR.
> > - weight_path (str): path of the model, default: None，pre-trained integral model will then be automatically downloaded.
```python
run(video_path)
```

> The execution interface after building the instance.
> **Parameters**
>
> > - video_path (str): path of the video file.
>
>**Return Value**
>
>> - tuple(pred_img(np.array), out_paht(str)): for image input, return the predicted image, PIL.Image type and the path where the image is saved.
> > - tuple(frame_path(str), out_path(str)): for video input, frame_path is the save path of each frame of the video after super resolution,  and out_path is the save path of the video after super resolution.
### run_image

```python
run_image(img)
```

> The interface of image super resolution.
> **Parameter**
>
> > - img (str|np.ndarray|Image.Image): input image, it could be the path of the image, np.ndarray, or PIL.Image type.
>
> **Return Value**
>
> > - pred_img(PIL.Image):  return the predicted image, PIL.Image type.
### run_video

```python
run_video(video)
```

> The interface of video super resolution.
> **Parameter**
>
> > - Video (str): path of the video file.
>
> **Return Value**
>
> > - tuple(frame_path(str), out_path(str)): frame_path is the save path of each frame of the video after super resolution,  and out_path is the save path of the video after super resolution.



## ppgan.apps.PPMSVSRPredictor

```python
ppgan.apps.PPMSVSRPredictor(output='output', weight_path=None, num_frames=10)
```

> Build the instance of PPMSVSR. PPMSVSR is a multi-stage VSR deep architecture. For more details, see the paper, PP-MSVSR: Multi-Stage Video Super-Resolution (https://arxiv.org/pdf/2112.02828.pdf).  The interface imposes 4x super resolution on the input video. The recommended video format is mp4.
>
> **Parameter**
>
> ```
> from ppgan.apps import PPMSVSRPredictor
> sr = PPMSVSRPredictor()
> # test a video file
> sr.run("docs/imgs/test.mp4")
> ```
> **参数**
>
> > - output (str):  path of the output image, default: output. Note that the path should be set as output/EDVR.
> > - weight_path (str): path of the model, default: None，pre-trained integral model will then be automatically downloaded.
> > - num_frames (int): the number of input frames of the PPMSVSR model, the default value: 10. Note that the larger the num_frames, the better the effect of the video after super resolution.
```python
run(video_path)
```

> The execution interface after building the instance.
> **Parameter**
>
> > - video_path (str): path of the video files.
>
> **Return Value**
>
> > - tuple(str, str): the former is the save path of each frame of the video after super resolution, the latter is the save path of the video after super resolution.


## ppgan.apps.PPMSVSRLargePredictor

```python
ppgan.apps.PPMSVSRLargePredictor(output='output', weight_path=None, num_frames=10)
```

> Build the instance of PPMSVSRLarge. PPMSVSRLarge is a Large PPMSVSR model. For more details, see the paper, PP-MSVSR: Multi-Stage Video Super-Resolution (https://arxiv.org/pdf/2112.02828.pdf).  The interface imposes 4x super resolution on the input video. The recommended video format is mp4.
>
> **Parameter**
>
> ```
> from ppgan.apps import PPMSVSRLargePredictor
> sr = PPMSVSRLargePredictor()
> # test a video file
> sr.run("docs/imgs/test.mp4")
> ```
> **参数**
>
> > - output (str):  path of the output image, default: output. Note that the path should be set as output/EDVR.
> > - weight_path (str): path of the model, default: None，pre-trained integral model will then be automatically downloaded.
> > - num_frames (int): the number of input frames of the PPMSVSR model, the default value: 10. Note that the larger the num_frames, the better the effect of the video after super resolution.
```python
run(video_path)
```

> The execution interface after building the instance.
> **Parameter**
>
> > - video_path (str): path of the video files.
>
> **Return Value**
>
> > - tuple(str, str): the former is the save path of each frame of the video after super resolution, the latter is the save path of the video after super resolution.

## ppgan.apps.EDVRPredictor

```python
ppgan.apps.EDVRPredictor(output='output', weight_path=None)
```

> Build the instance of EDVR. EDVR is a model designed for video super resolution. For more details, see the paper, EDVR: Video Restoration with Enhanced Deformable Convolutional Networks (https://arxiv.org/abs/1905.02716).  The interface imposes 4x super resolution on the input video. The recommended video format is mp4.
>
> **Parameter**
>
> ```
> from ppgan.apps import EDVRPredictor
> sr = EDVRPredictor()
> # test a video file
> sr.run("docs/imgs/test.mp4")
> ```
> **参数**
>
> > - output (str):  path of the output image, default: output. Note that the path should be set as output/EDVR.
> > - weight_path (str): path of the model, default: None，pre-trained integral model will then be automatically downloaded.
```python
run(video_path)
```

> The execution interface after building the instance.
> **Parameter**
>
> > - video_path (str): path of the video files.
>
> **Return Value**
>
> > - tuple(str, str): the former is the save path of each frame of the video after super resolution, the latter is the save path of the video after super resolution.


## ppgan.apps.BasicVSRPredictor

```python
ppgan.apps.BasicVSRPredictor(output='output', weight_path=None, num_frames=10)
```

> Build the instance of BasicVSR. BasicVSR is a model designed for video super resolution. For more details, see the paper, BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond (https://arxiv.org/pdf/2012.02181.pdf).  The interface imposes 4x super resolution on the input video. The recommended video format is mp4.
>
> **Parameter**
>
> ```
> from ppgan.apps import BasicVSRPredictor
> sr = BasicVSRPredictor()
> # test a video file
> sr.run("docs/imgs/test.mp4")
> ```
> **参数**
>
> > - output (str):  path of the output image, default: output. Note that the path should be set as output/EDVR.
> > - weight_path (str): path of the model, default: None，pre-trained integral model will then be automatically downloaded.
> > - num_frames (int): the number of input frames of the PPMSVSR model, the default value: 10. Note that the larger the num_frames, the better the effect of the video after super resolution.
```python
run(video_path)
```

> The execution interface after building the instance.
> **Parameter**
>
> > - video_path (str): path of the video files.
>
> **Return Value**
>
> > - tuple(str, str): the former is the save path of each frame of the video after super resolution, the latter is the save path of the video after super resolution.

## ppgan.apps.IconVSRPredictor

```python
ppgan.apps.IconVSRPredictor(output='output', weight_path=None, num_frames=10)
```

> Build the instance of IconVSR. IconVSR is a VSR model expanded by BasicVSR. For more details, see the paper, BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond (https://arxiv.org/pdf/2012.02181.pdf).  The interface imposes 4x super resolution on the input video. The recommended video format is mp4.
>
> **Parameter**
>
> ```
> from ppgan.apps import IconVSRPredictor
> sr = IconVSRPredictor()
> # test a video file
> sr.run("docs/imgs/test.mp4")
> ```
> **参数**
>
> > - output (str):  path of the output image, default: output. Note that the path should be set as output/EDVR.
> > - weight_path (str): path of the model, default: None，pre-trained integral model will then be automatically downloaded.
> > - num_frames (int): the number of input frames of the PPMSVSR model, the default value: 10. Note that the larger the num_frames, the better the effect of the video after super resolution.
```python
run(video_path)
```

> The execution interface after building the instance.
> **Parameter**
>
> > - video_path (str): path of the video files.
>
> **Return Value**
>
> > - tuple(str, str): the former is the save path of each frame of the video after super resolution, the latter is the save path of the video after super resolution.


## ppgan.apps.BasiVSRPlusPlusPredictor

```python
ppgan.apps.BasiVSRPlusPlusPredictor(output='output', weight_path=None, num_frames=10)
```

> Build the instance of BasiVSRPlusPlus. BasiVSRPlusPlus is a model designed for video super resolution. For more details, see the paper, BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment (https://arxiv.org/pdf/2104.13371v1.pdf).  The interface imposes 4x super resolution on the input video. The recommended video format is mp4.
>
> **Parameter**
>
> ```
> from ppgan.apps import BasiVSRPlusPlusPredictor
> sr = BasiVSRPlusPlusPredictor()
> # test a video file
> sr.run("docs/imgs/test.mp4")
> ```
> **参数**
>
> > - output (str):  path of the output image, default: output. Note that the path should be set as output/EDVR.
> > - weight_path (str): path of the model, default: None，pre-trained integral model will then be automatically downloaded.
> > - num_frames (int): the number of input frames of the PPMSVSR model, the default value: 10. Note that the larger the num_frames, the better the effect of the video after super resolution.
```python
run(video_path)
```

> The execution interface after building the instance.
> **Parameter**
>
> > - video_path (str): path of the video files.
>
> **Return Value**
>
> > - tuple(str, str): the former is the save path of each frame of the video after super resolution, the latter is the save path of the video after super resolution.



## ppgan.apps.DAINPredictor

```python
ppgan.apps.DAINPredictor(output='output', weight_path=None，time_step=None, use_gpu=True, key_frame_thread=0，remove_duplicates=False)
```

> Build the instance of DAIN model. DAIN supports video frame interpolation, producing videos with higher frame rate. For more details, see the paper, DAIN: Depth-Aware Video Frame interpolation (https://arxiv.org/abs/1904.00830).
>
> *Note: The interface is only available in static graph, add the following codes to enable static graph before using it：
>
> ```
> import paddle
> paddle.enable_static() #enable static graph
> paddle.disable_static() #disable static graph
> ```
>
> **Example**
>
> ```
> from ppgan.apps import DAINPredictor
> dain = DAINPredictor(time_step=0.5) # With no defualt value, time_step need to be manually specified
> # test a video file
> dain.run("docs/imgs/test.mp4")
> ```
> **Parameters**
>
> > - output_path (str):  path of the predicted output, default: output. Note that the path should be set as output/DAIN.
> > - weight_path (str):  path of the model, default: None, pre-trained integral model will then be automatically downloaded.
> > - time_step (float): the frame rate changes by a factor of 1./time_step, e.g. 2x frames if time_step is 0.5 and 4x frames if it is 0.25.
> > - use_gpu (bool): whether to make predictions by using GPU, default: True.
> > - remove_duplicates (bool): whether to remove duplicates, default: False.
```python
run(video_path)
```

> The execution interface after building the instance.
> **Parameters**
>
> > - video_path (str): path of the video file.
>
> **Return Value**
>
> > - tuple(str, str): for video input, frame_path is the save path of the image after colorizing each frame of the video, and out_path is the save path of the colorized video.


## ppgan.apps.FirstOrderPredictor

```python
ppgan.apps.FirstOrderPredictor(output='output', weight_path=None，config=None, relative=False, adapt_scale=False，find_best_frame=False, best_frame=None)
```

> Build the instance of FirstOrder model. The model is dedicated to Image Animation, i.e., generating a video sequence so that an object in a source image is animated according to the motion of a driving video.
>
> For more details, see paper, First Order Motion Model for Image Animation (https://arxiv.org/abs/2003.00196) .
>
> **Example**
>
> ```
> from ppgan.apps import FirstOrderPredictor
> animate = FirstOrderPredictor()
> # test a video file
> animate.run("source.png"，"driving.mp4")
> ```
> **Parameters**
>
> > - output_path (str):  path of the predicted output, default: output. Note that the path should be set as output/result.mp4.
> > - weight_path (str):  path of the model, default: None, pre-trained integral model will then be automatically downloaded.
> > - config (dict|str|None): model configuration, it can be a dictionary type or a YML file, and the default value None is adopted. When the weight is None by default, the config also needs to adopt the default value None. otherwise, the configuration here should be consistent with the corresponding weight.
> > - relative (bool):  indicate whether the relative or absolute coordinates of key points in the video are used in the program, default: False.
> > - adapt_scale (bool): adapt movement scale based on convex hull of key points, default: False.
> > - find_best_frame (bool): whether to start generating from the frame that best matches the source image, which exclusively applies to face applications and requires libraries with face alignment.
> > - best_frame (int): set the number of the starting frame, default: None, that is, starting from the first frame(counting from 1).
```python
run(source_image，driving_video)
```

> The execution interface after building the instance, the predicted video is save in output/result.mp4.
> **Parameters**
>
> > - source_image (str): input the source image。
> > - driving_video (str): input the driving video, mp4 format recommended.
>
> **Return Value**
>
> > None.
## ppgan.apps.FaceParsePredictor

```pyhton
ppgan.apps.FaceParsePredictor(output_path='output')
```
> Build the instance of the face parsing model. The model is devoted to address the task of distributing a pixel-wise label to each semantic components (e.g. hair, lips, nose, ears, etc.) in accordance with the input facial image. The task proceeds with the help of BiseNet.
>
> For more details, see the paper, BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation (https://arxiv.org/abs/1808.00897v1).
>
> *Note: dlib package is needed for this interface, use the following codes to install it：
>
> ```
> pip install dlib
> ```
> It may take long to install this package under Windows, please be patient.
>
> **Parameters:**
>
> > - input_image: path of the input image to be parsed
> > - output_path: path of the output to be saved
> **Example:**
>
> ```
> from ppgan.apps import FaceParsePredictor
> parser = FaceParsePredictor()
> parser.run('docs/imgs/face.png')
> ```
> **Return Value:**
>
> > - mask(numpy.ndarray): return the mask matrix of the parsed facial components, data type: numpy.ndarray.
## ppgan.apps.AnimeGANPredictor

```pyhton
ppgan.apps.AnimeGANPredictor(output_path='output_dir',weight_path=None,use_adjust_brightness=True)
```
> Adopt the AnimeGAN v2 to realize the animation of scenery images.
>
> For more details, see the paper, AnimeGAN: A Novel Lightweight GAN for Photo Animation (https://link.springer.com/chapter/10.1007/978-981-15-5577-0_18).
> **Parameters:**
>
> > - input_image: path of the input image to be parsed.
> **Example:**
>
> ```
> from ppgan.apps import AnimeGANPredictor
> predictor = AnimeGANPredictor()
> predictor.run('docs/imgs/animeganv2_test.jpg')
> ```
> **Return Value:**
>
> > - anime_image(numpy.ndarray): return the stylized scenery image.

## ppgan.apps.MiDaSPredictor

```pyhton
ppgan.apps.MiDaSPredictor(output=None, weight_path=None)
```

> MiDaSv2 is a monocular depth estimation model (see https://github.com/intel-isl/MiDaS). Monocular depth estimation is a method used to compute depth from a singe RGB image.
>
> For more details, see the paper Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer (https://arxiv.org/abs/1907.01341v3).
> **Example**
>
> ```python
> from ppgan.apps import MiDaSPredictor
> # if set output, will write depth pfm and png file in output/MiDaS
> model = MiDaSPredictor()
> prediction = model.run()
> ```
>
> Color display of the depth image:
>
> ```python
> import numpy as np
> import PIL.Image as Image
> import matplotlib as mpl
> import matplotlib.cm as cm
>
> vmax = np.percentile(prediction, 95)
> normalizer = mpl.colors.Normalize(vmin=prediction.min(), vmax=vmax)
> mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
> colormapped_im = (mapper.to_rgba(prediction)[:, :, :3] * 255).astype(np.uint8)
> im = Image.fromarray(colormapped_im)
> im.save('test_disp.jpeg')
> ```
>
> **Parameters:**
>
> > - output (str): path of the output, if it is None, no pfm and png depth image will be saved.
> > - weight_path (str): path of the model, default: None, pre-trained integral model will then be automatically downloaded.
> **Return Value:**
>
> > - prediction (numpy.ndarray): return the prediction.
> > - pfm_f (str): return the save path of pfm files if the output path is set.
> > - png_f (str): return the save path of png files if the output path is set.

## ppgan.apps.Wav2LipPredictor

```python
ppgan.apps.Wav2LipPredictor(face=None, ausio_seq=None, outfile=None)
```

> Build the instance for the Wav2Lip model, which is used for lip generation, i.e., achieving the synchronization of lip movements on a talking face video and the voice from an input audio.
>
> For more details, see the paper, A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild (http://arxiv.org/abs/2008.10010).
>
> **Example**
>
> ```
> from ppgan.apps import Wav2LipPredictor
> import ppgan
> predictor = Wav2LipPredictor()
> predictor.run('/home/aistudio/先烈.jpeg', '/home/aistudio/pp_guangquan_zhenzhu46s.mp4','wav2lip')
> ```
> **Parameters:**
> - face (str): path of images or videos containing human face.
> - audio_seq (str): path of the input audio, any processable format in ffmpeg is supported, including `.wav`, `.mp3`, `.m4a` etc.
> - outfile (str): path of the output video file.
>**Return Value**
>
>> None
