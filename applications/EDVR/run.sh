# examples of running programs:
# bash ./run.sh inference EDVR ./configs/edvr_L.yaml
# bash ./run.sh predict EDvR ./cofings/edvr_L.yaml

# configs should be ./configs/xxx.yaml

mode=$1
name=$2
configs=$3

save_inference_dir="./data/inference_model"
use_gpu=True
fix_random_seed=False
log_interval=1
valid_interval=1

weights="./weights/paddle_state_dict_L.npz"


export CUDA_VISIBLE_DEVICES=6   #0,1,5,6 fast,  2,3,4,7 slow
# export FLAGS_fast_eager_deletion_mode=1
# export FLAGS_eager_delete_tensor_gb=0.0
# export FLAGS_fraction_of_gpu_memory_to_use=0.98

if [ "$mode"x == "predict"x ]; then
    echo $mode $name $configs $weights
    if [ "$weights"x != ""x ]; then
        python predict.py --model_name=$name \
                          --config=$configs \
                          --log_interval=$log_interval \
                          --video_path='' \
                          --use_gpu=$use_gpu
    else
        python predict.py --model_name=$name \
                          --config=$configs \
                          --log_interval=$log_interval \
                          --use_gpu=$use_gpu \
                          --video_path=''
    fi
fi

