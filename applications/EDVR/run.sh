# examples of running programs:
# bash ./run.sh train CTCN ./configs/ctcn.yaml
# bash ./run.sh eval NEXTVLAD ./configs/nextvlad.yaml
# bash ./run.sh predict NONLOCAL ./cofings/nonlocal.yaml

# mode should be one of [train, eval, predict, inference]
# name should be one of [AttentionCluster, AttentionLSTM, NEXTVLAD, NONLOCAL, TSN, TSM, STNET, CTCN]
# configs should be ./configs/xxx.yaml

mode=$1
name=$2
configs=$3

pretrain="./tmp/name_map/paddle_state_dict.npz" # set pretrain model path if needed
resume="" # set pretrain model path if needed
save_dir="./data/checkpoints"
save_inference_dir="./data/inference_model"
use_gpu=True
fix_random_seed=False
log_interval=1
valid_interval=1

#weights="./data/checkpoints/EDVR_epoch721.pdparams" #set the path of weights to enable eval and predicut, just ignore this when training

#weights="./data/checkpoints_with_tsa/EDVR_epoch821.pdparams"
weights="./weights/paddle_state_dict_L.npz"
#weights="./weights/"


#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=4,5,6,7   #0,1,5,6 fast,  2,3,4,7 slow
#export CUDA_VISIBLE_DEVICES=7
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98

if [ "$mode" == "train" ]; then
    echo $mode $name $configs $resume $pretrain
    if [ "$resume"x != ""x ]; then
        python train.py --model_name=$name \
                        --config=$configs \
                        --resume=$resume \
                        --log_interval=$log_interval \
                        --valid_interval=$valid_interval \
                        --use_gpu=$use_gpu \
                        --save_dir=$save_dir \
                        --fix_random_seed=$fix_random_seed
    elif [ "$pretrain"x != ""x ]; then
        python train.py --model_name=$name \
                        --config=$configs \
                        --pretrain=$pretrain \
                        --log_interval=$log_interval \
                        --valid_interval=$valid_interval \
                        --use_gpu=$use_gpu \
                        --save_dir=$save_dir \
                        --fix_random_seed=$fix_random_seed
    else
        python train.py --model_name=$name \
                        --config=$configs \
                        --log_interval=$log_interval \
                        --valid_interval=$valid_interval \
                        --use_gpu=$use_gpu \
                        --save_dir=$save_dir \
                        --fix_random_seed=$fix_random_seed
    fi
elif [ "$mode"x == "eval"x ]; then
    echo $mode $name $configs $weights
    if [ "$weights"x != ""x ]; then
        python eval.py --model_name=$name \
                       --config=$configs \
                       --log_interval=$log_interval \
                       --weights=$weights \
                       --use_gpu=$use_gpu
    else
        python eval.py --model_name=$name \
                       --config=$configs \
                       --log_interval=$log_interval \
                       --use_gpu=$use_gpu
    fi
elif [ "$mode"x == "predict"x ]; then
    echo $mode $name $configs $weights
    if [ "$weights"x != ""x ]; then
        python predict.py --model_name=$name \
                          --config=$configs \
                          --log_interval=$log_interval \
                          --weights=$weights \
                          --video_path='' \
                          --use_gpu=$use_gpu
    else
        python predict.py --model_name=$name \
                          --config=$configs \
                          --log_interval=$log_interval \
                          --use_gpu=$use_gpu \
                          --video_path=''
    fi
elif [ "$mode"x == "inference"x ]; then
    echo $mode $name $configs $weights
    if [ "$weights"x != ""x ]; then
        python inference_model.py --model_name=$name \
                                  --config=$configs \
                                  --weights=$weights \
                                  --use_gpu=$use_gpu \
                                  --save_dir=$save_inference_dir
    else
        python inference_model.py --model_name=$name \
                                  --config=$configs \
                                  --use_gpu=$use_gpu \
                                  --save_dir=$save_inference_dir
    fi
else
    echo "Not implemented mode " $mode
fi

