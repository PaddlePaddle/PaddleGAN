#    AIStudio ID: 郭呱呱1997

#                 老北京城影像修复

###########################################下面是对自己上传的老视频的修复处理#########################

# 安装ppgan
# 当前目录在: /home/aistudio/, 这个目录也是左边文件和文件夹所在的目录
# 克隆最新的PaddleGAN仓库到当前目录
# !git clone https://github.com/PaddlePaddle/PaddleGAN.git
# 如果从github下载慢可以从gitee clone：
!git clone https://gitee.com/paddlepaddle/PaddleGAN.git
%cd PaddleGAN/
!pip install -v -e .

# 导入一些可视化需要的包
import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import warnings
warnings.filterwarnings("ignore")

# 定义一个展示视频的函数
def display(driving, fps, size=(8, 6)):
    fig = plt.figure(figsize=size)

    ims = []
    for i in range(len(driving)):
        cols = []
        cols.append(driving[i])

        im = plt.imshow(np.concatenate(cols, axis=1), animated=True)
        plt.axis('off')
        ims.append([im])

    video = animation.ArtistAnimation(fig, ims, interval=1000.0/fps, repeat_delay=1000)

    plt.close()
    return video
    
# 展示一下输入的视频, 如果视频太大，时间会非常久，可以跳过这个步骤
# Video 的分辨率或者时长等参数不合适，可能导致运行出错
# Video2 是经过自己前期处理后的视频，主要调整了视频画面大小，对时长也进行了剪辑，之前剪辑的时候没注意，可能不是整数秒，这样可能导致后面截取帧的时候最后卡在98%，所以个人认为最好是严格的整数秒。
video_path = '/home/aistudio/MyVideo2.mp4'
video_frames = imageio.mimread(video_path, memtest=False)

# 获得视频的原分辨率
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
    
HTML(display(video_frames, fps).to_html5_video())


# 使用插帧(DAIN), 上色(DeOldify), 超分(EDVR)这三个模型对该视频进行修复
# input参数表示输入的视频路径
# output表示处理后的视频的存放文件夹
# proccess_order 表示使用的模型和顺序（目前支持）
%cd /home/aistudio/PaddleGAN/applications/
!python tools/video-enhance.py --input /home/aistudio/MyVideo2.mp4 \
                               --process_order DAIN DeOldify EDVR \
                               --output output_dir
                               
# 展示一下处理好的视频, 如果视频太大，时间会非常久，可以下载下来看
# 这个路径可以查看上个code cell的最后打印的output video path
output_video_path = '/home/aistudio/PaddleGAN/applications/output_dir/EDVR/MyVideo2_deoldify_out_edvr_out.mp4'

video_frames = imageio.mimread(output_video_path, memtest=False)

# 获得视频的原分辨率
cap = cv2.VideoCapture(output_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
    

HTML(display(video_frames, fps, size=(16, 12)).to_html5_video())

