# 模型说明
# 目前包含DAIN(插帧模型)，DeOldify(上色模型)，DeepRemaster(去噪与上色模型)，EDVR(基于连续帧(视频)超分辨率模型)，RealSR(基于图片的超分辨率模型)
# 参数说明
# input 输入视频的路径
# output 输出视频保存的路径
# proccess_order 要使用的模型及顺序

python tools/video-enhance.py \
--input input.mp4  --output output --proccess_order DeOldify RealSR
