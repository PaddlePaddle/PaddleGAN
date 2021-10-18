# Image Colorization
PaddleGAN provides [DeOldify](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/apis/apps.md#ppganappsdeoldifypredictor) model for image colorization.

## DeOldifyPredictor

[DeOldify](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/apis/apps.md#ppganappsdeoldifypredictor) generates the adversarial network with a self-attentive mechanism. The generator is a U-NET structured network with better effects in image/video coloring.

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/117925538-fd526a80-b329-11eb-8924-8f2614fcd9e6.png'>
</div>

### Parameters

- `output (str，可选的)`: path of the output folder, default value: `output`
- `weight_path (None, optional)`: path to load weights, if not set, the default weights will be downloaded locally from the cloud. Default value:`None`
- `artistic (bool)`: whether or not to use the "artistic" model. "Artistic" models are likely to produce some interesting colors, but with some burrs.
- `render_factor (int)`: This parameter will be multiplied by 16 and used as the resize value for the input frame. If the value is set to 32, the input frame will be resized to a size of (32 * 16, 32 * 16) and fed into the network.


### Usage
**1. API Prediction**

```
from ppgan.apps import DeOldifyPredictor
deoldify = DeOldifyPredictor()
deoldify.run("/home/aistudio/先烈.jpg") #原图片所在路径
```
*`run` interface is a common interface for images/videos, since the object here is an image, the interface of `run_image` is suitable.

[Complete API interface usage instructions]()

**2. Command-Line Prediction**

```
!python applications/tools/video-enhance.py --input /home/aistudio/先烈.jpg \ #Original image path
                               --process_order DeOldify \ #Order of processing of the original image
                               --output output_dir #Path of the final image
```

### Experience Online Projects
**1. [Old Beijing City Video Restoration](https://aistudio.baidu.com/aistudio/projectdetail/1161285)**

**2. [PaddleGAN ❤️ 520 Edition](https://aistudio.baidu.com/aistudio/projectdetail/1956943?channelType=0&channel=0)**
