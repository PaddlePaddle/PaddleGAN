import moviepy.editor as mp
import time


if __name__ == '__main__':
    start = time.time()
    input_video = '/home/user/paddle/PaddleGAN/data/baseline.MP4'
    target_video = '/home/user/paddle/PaddleGAN/output/selfie2with GFPGAN preprocess with bg enchance.mp4'
    output_video = '/home/user/paddle/PaddleGAN/output/selfie2with GFPGAN preprocess with bg enchance with audio.mp4'
    videoclip_1 = mp.VideoFileClip(input_video)
    videoclip_2 = mp.VideoFileClip(target_video)
    audio_1 = videoclip_1.audio

    videoclip_3 = videoclip_2.set_audio(audio_1)
    videoclip_3.write_videofile(output_video, audio_codec="aac")
    print(time.time() - start)
