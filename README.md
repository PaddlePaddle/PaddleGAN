
English | [简体中文](./README_cn.md)

# PaddleGAN

PaddleGAN provides developers with high-performance implementation of classic and SOTA Generative Adversarial Networks, and supports developers to quickly build, train and deploy GANs for academic, entertainment and industrial usage.

GAN-Generative Adversarial Network, was praised by "the Father of Convolutional Networks"  **Yann LeCun (Yang Likun)**  as **[One of the most interesting ideas in the field of computer science in the past decade]**. It's the one research area in deep learning that AI researchers are most concerned about.

<div align='center'>
  <img src='./docs/imgs/ppgan.jpg'>
</div>

[![License](https://img.shields.io/badge/license-Apache%202-red.svg)](LICENSE)![python version](https://img.shields.io/badge/python-3.6+-orange.svg)

## 🎪 Hot Activities

- 🔥 **2021.7.9-2021.9** 🔥

  **💙 AI Creation Camp 💙**

  **You can implement any abilities in PaddleGAN with Wechaty to create your own chat robot 🤖 !**

  **A plenty of gifts 🎁 waiting for you!**

  **💰First Prize: 30,000RMB**

  **🎮 Second Prize: PS5**

  **🕶 Third Prize: VR Glass**

  **🏵 Most Popular Prize: 3D Printer**

  **Still hezitating? Click here and sign up!**  **https://aistudio.baidu.com/aistudio/competition/detail/98**

- 2021.4.15~4.22

  GAN 7 Days Course Camp: Baidu Senior Research Developers help you learn the basic and advanced GAN knowledge in 7 days!

  **Courses videos and related materials: https://aistudio.baidu.com/aistudio/course/introduce/16651**

## 🚀 Recent Updates

- 👩‍🚀**A Space Odyssey ：[LapStyle](./docs/zh_CN/tutorials/lap_style.md) image translation take you travel around the universe**👨‍🚀

  - **[Online Toturials](https://aistudio.baidu.com/aistudio/projectdetail/2343740?contributionType=1)**

    <div align='center'>
      <img src='https://user-images.githubusercontent.com/48054808/133392621-9a552c46-841b-4fe4-bb24-7b0cbf86616c.gif' width='250'/>
      <img src='https://user-images.githubusercontent.com/48054808/133392630-c5329c4c-bc10-406e-a853-812a2b1f0fa6.gif' width='250'/>
      <img src='https://user-images.githubusercontent.com/48054808/133392652-f4811b1e-0676-4402-808b-a4c96c611368.gif' width='250'/>
    </div>

- 🧙‍♂️**Latest Creative Project：create magic/dynamic profile for your student ID in Hogwarts** 🧙‍♀️

  - **[Online Toturials](https://aistudio.baidu.com/aistudio/projectdetail/2288888?channelType=0&channel=0)**

    <div align='center'>
      <img src='https://ai-studio-static-online.cdn.bcebos.com/da1c51844ac048aa8d4fa3151be95215eee75d8bb488409d92ec17285b227c2c' width='250'/>
    </div>
  
- **💞Add Face Morphing function💞: you can perfectly merge any two faces and make the new face get any facial expressions!**

  - Tutorials: https://aistudio.baidu.com/aistudio/projectdetail/2254031

    <div align='center'>
      <img src='https://user-images.githubusercontent.com/48054808/128299870-66a73bb3-57a4-4985-aadc-8ddeab048145.gif' width='300'/>
    </div>

- **Publish a new version of First Oder Motion model by having two impressive features:**
  - High resolution 512x512 
  - Face Enhancement
  - Tutorials: https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/motion_driving.md 
  
- **New image translation ability--transfer photo into oil painting style:**

  - Complete tutorials for deployment: https://github.com/wzmsltw/PaintTransformer

    <div align='center'>
      <img src='https://user-images.githubusercontent.com/48054808/129904830-8b87e310-ea51-4aff-b29b-88920ee82447.png' width='700'/>
    </div>

## Document Tutorial

#### **Installation**

* Environment dependence: 
   - PaddlePaddle >= 2.1.0
   - Python >= 3.6
   - CUDA >= 10.1
* [Full installation tutorial](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/install.md)

#### **Starter Tutorial**

- [Quick start](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/get_started.md)
- [Data Preparation](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/data_prepare.md)
- [Instruction of APIs](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/apis/apps.md)
- [Instruction of Config Files](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/config_doc.md)

## Model Tutorial

* [Pixel2Pixel](./docs/en_US/tutorials/pix2pix_cyclegan.md)
* [CycleGAN](./docs/en_US/tutorials/pix2pix_cyclegan.md)
* [LapStyle](./docs/en_US/tutorials/lap_style.md)
* [PSGAN](./docs/en_US/tutorials/psgan.md)
* [First Order Motion Model](./docs/en_US/tutorials/motion_driving.md)
* [FaceParsing](./docs/en_US/tutorials/face_parse.md)
* [AnimeGANv2](./docs/en_US/tutorials/animegan.md)
* [U-GAT-IT](./docs/en_US/tutorials/ugatit.md)
* [Photo2Cartoon](./docs/en_US/tutorials/photo2cartoon.md)
* [Wav2Lip](./docs/en_US/tutorials/wav2lip.md)
* [Single Image Super Resolution(SISR)](./docs/en_US/tutorials/single_image_super_resolution.md)
* [Video Super Resolution(VSR)](./docs/en_US/tutorials/video_super_resolution.md)
* [StyleGAN2](./docs/en_US/tutorials/styleganv2.md)
* [Pixel2Style2Pixel](./docs/en_US/tutorials/pixel2style2pixel.md)


## Composite Application

* [Video restore](./docs/en_US/tutorials/video_restore.md)

## Online Tutorial

You can run those projects in the [AI Studio](https://aistudio.baidu.com/aistudio/projectoverview/public/1?kw=paddlegan) to learn how to use the models above:

|Online Tutorial      |    link  |
|--------------|-----------|
|Motion Driving-multi-personal "Mai-ha-hi" | [Click and Try](https://aistudio.baidu.com/aistudio/projectdetail/1603391) |
|Restore the video of Beijing hundreds years ago|[Click and Try](https://aistudio.baidu.com/aistudio/projectdetail/1161285)|
|Motion Driving-When "Su Daqiang" sings "unravel" |[Click and Try](https://aistudio.baidu.com/aistudio/projectdetail/1048840)|

## Examples

### Face Morphing 

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/129020371-75de20d1-705b-44b1-8254-e09710124244.gif'width='700' />
</div>

### Image Translation

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/119464966-d5c1c000-bd75-11eb-9696-9bb75357229f.gif'width='700' height='200'/>
</div>


### Old video restore
<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/119469496-fc81f580-bd79-11eb-865a-5e38482b1ae8.gif' width='700'/>
</div>



### Motion driving
<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/119469551-0a377b00-bd7a-11eb-9117-e4871c8fb9c0.gif' width='700'>
</div>


### Super resolution

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/119469753-3e12a080-bd7a-11eb-9cde-4fa01b3201ab.png'width='700' height='250'/>
</div>



### Makeup shifter

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/119469834-4ff44380-bd7a-11eb-93b6-05b705dcfbf2.png'width='700' height='250'/>
</div>



### Face cartoonization

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/119469952-6bf7e500-bd7a-11eb-89ad-9a78b10bd4ab.png'width='700' height='250'/>
</div>



### Realistic face cartoonization

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/119470028-7f0ab500-bd7a-11eb-88e9-78a6b9e2e319.png'width='700' height='250'/>
</div>



### Photo animation

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/119470099-9184ee80-bd7a-11eb-8b12-c9400fe01266.png'width='700' height='250'/>
</div>



### Lip-syncing

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/119470166-a6618200-bd7a-11eb-9f98-58052ce21b14.gif'width='700'>
</div>



## Changelog

- v0.1.0 (2020.11.02)
  - Release first version, supported models include Pixel2Pixel, CycleGAN, PSGAN. Supported applications include video frame interpolation, super resolution, colorize images and videos, image animation.
  - Modular design and friendly interface.

## Community

Scan OR Code below to join [PaddleGAN QQ Group：1058398620], you can get offical technical support  here and communicate with other developers/friends. Look forward to your participation!

<div align='center'>
  <img src='./docs/imgs/qq.png'width='250' height='300'/>
</div>

### PaddleGAN Special Interest Group（SIG）

It was first proposed and used by [ACM（Association for Computing Machinery)](https://en.wikipedia.org/wiki/Association_for_Computing_Machinery) in 1961. Top International open source organizations including [Kubernates](https://kubernetes.io/) all adopt the form of SIGs, so that members with the same specific interests can share, learn knowledge and develop projects. These members do not need to be in the same country/region or the same organization, as long as they are like-minded, they can all study, work, and play together with the same goals~

PaddleGAN SIG is such a developer organization that brings together people who interested in GAN. There are frontline developers of PaddlePaddle, senior engineers from the world's top 500, and students from top universities at home and abroad.

We are continuing to recruit developers interested and capable to join us building this project and explore more useful and interesting applications together.

SIG contributions:

- [zhen8838](https://github.com/zhen8838): contributed to AnimeGANv2.
- [Jay9z](https://github.com/Jay9z): contributed to DCGAN and updated install docs, etc.
- [HighCWu](https://github.com/HighCWu): contributed to c-DCGAN and WGAN. Support to use `paddle.vision.datasets`.
- [hao-qiang](https://github.com/hao-qiang) & [ minivision-ai ](https://github.com/minivision-ai): contributed to the photo2cartoon project.


## Contributing

Contributions and suggestions are highly welcomed. Most contributions require you to agree to a [Contributor License Agreement (CLA)](https://cla-assistant.io/PaddlePaddle/PaddleGAN) declaring.
When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA. Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.
For more, please reference [contribution guidelines](docs/en_US/contribute.md).

## License
PaddleGAN is released under the [Apache 2.0 license](LICENSE).
