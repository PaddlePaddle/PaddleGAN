cd pwcnet/correlation_op
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`python -c 'import paddle; print(paddle.sysconfig.get_lib())'`
export PYTHONPATH=$PYTHONPATH:`pwd`
cd ../../

VID_PATH=/paddle/work/github/DAIN/data/CBA.mp4
OUT_PATH=output
MODEL_PATH=DAIN_paddle_weight

#CUDA_VISIBLE_DEVICES=1 python demo.py \
#    --time_step 0.125 \
#    --video_path=$VID_PATH \
#    --output_path=$OUT_PATH \
#    --saved_model=$MODEL_PATH

CUDA_VISIBLE_DEVICES=2 python predict.py \
    --time_step 0.125 \
    --video_path=$VID_PATH \
    --output_path=$OUT_PATH \
    --saved_model=$MODEL_PATH
