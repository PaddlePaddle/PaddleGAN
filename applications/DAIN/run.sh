cd pwcnet/correlation_op
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`python -c 'import paddle; print(paddle.sysconfig.get_lib())'`
export PYTHONPATH=$PYTHONPATH:`pwd`
cd ../../

# VID_PATH=/workspace/codes/colorization_paddle_net_weights/video/Peking_input360p.mp4
VID_PATH=/workspace/codes/colorization_paddle_net_weights/video/Peking_input360p_clip_5_15.mp4
OUT_PATH=output
MODEL_PATH=DAIN_paddle_weight

#CUDA_VISIBLE_DEVICES=1 python demo.py \
#    --time_step 0.125 \
#    --video_path=$VID_PATH \
#    --output_path=$OUT_PATH \
#    --saved_model=$MODEL_PATH

CUDA_VISIBLE_DEVICES=5 python predict.py \
    --time_step 0.5 \
    --video_path=$VID_PATH \
    --output_path=$OUT_PATH \
    --saved_model=$MODEL_PATH
