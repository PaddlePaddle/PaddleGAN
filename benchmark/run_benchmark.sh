#!/usr/bin/env bash
set -xe
# 运行示例：CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} 500 ${model_mode}
# 参数说明
function _set_params(){
    model_item=${1:-"model_item"} 
    base_batch_size=${2:-"2"}  
    fp_item=${3:-"fp32"}        # fp32|fp16
    run_process_type=${4:-"SingleP"}
    run_mode=${5:-"DP"} 
    device_num=${6:-"N1C1"}
    profiling=${PROFILING:-"false"}
    model_repo="PaddleGAN"
    ips_unit="samples/sec"
    skip_steps=10 
    keyword="ips:"        
    convergence_key="G_idt_A_loss:"
    # 以下为通用执行命令，无特殊可不用修改
    model_name=${model_item}_bs${base_batch_size}_${fp_item}_${run_process_type}_${run_mode}
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}  # （必填） TRAIN_LOG_DIR  benchmark框架设置该参数为全局变量
    profiling_log_path=${PROFILING_LOG_DIR:-$(pwd)}  # （必填） PROFILING_LOG_DIR benchmark框架设置该参数为全局变量
    speed_log_path=${LOG_PATH_INDEX_DIR:-$(pwd)}
    train_log_file=${run_log_path}/${model_repo}_${model_name}_${device_num}_log
    profiling_log_file=${profiling_log_path}/${model_repo}_${model_name}_${device_num}_profiling
    speed_log_file=${speed_log_path}/${model_repo}_${model_name}_${device_num}_speed
}


function _train(){
    batch_size=${base_batch_size}  # 如果模型跑多卡单进程时,请在_train函数中计算出多卡需要的bs
    echo "current CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, model_name=${model_name}, device_num=${device_num}, is profiling=${profiling}"
    if [ ${profiling} = "true" ];then
            add_options="--profiler_options=\"batch_range=[10,20];state=GPU;tracer_option=Default;profile_path=model.profile\""
            log_file=${profiling_log_file}
        else
            add_options=""
            log_file=${train_log_file}
    fi
    if [ ${fp_item} = "fp16" ]; then
        use_fp16_cmd="--fp16"
    fi
    
    if [[ ${model_item} = "FOMM" ]];then
        train_cmd="--config-file=benchmark/configs/${model_item}.yaml
               -o dataset.train.batch_size=${batch_size} epochs=2 ${add_options}"
    else
        train_cmd="--config-file=benchmark/configs/${model_item}.yaml
               -o dataset.train.batch_size=${batch_size} ${add_options}"
    fi
            
    
# 以下不用修改
    case ${run_mode} in
    DP) if [[ ${run_process_type} = "SingleP" ]];then
            echo "run ${run_mode} ${run_process_type}"
            train_cmd="python -u tools/main.py ${train_cmd}" 
        elif [[ ${run_process_type} = "MultiP" ]];then
            rm -rf ./mylog
            train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --gpus=$CUDA_VISIBLE_DEVICES \
                  tools/main.py ${train_cmd}" 
        else  
            echo "run ${run_mode} ${run_process_type} error", exit 1
        fi
        ;;
    DP1-MP1-PP1)  echo "run run_mode: DP1-MP1-PP1" ;;
    *) echo "choose run_mode "; exit 1;
    esac
    echo "train_cmd: ${train_cmd}  log_file: ${log_file}"
    timeout 15m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
    else
        echo -e "${model_name}, SUCCESS"
    fi
    trap 'for pid in $(jobs -pr); do kill -KILL $pid; done' INT QUIT TERM

    if [ ${run_process_type} = "MultiP" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi

}

source ${BENCHMARK_ROOT}/scripts/run_model.sh   # 在该脚本中会对符合benchmark规范的log使用analysis.py 脚本进行性能数据解析;如果不联调只想要产出训练log可以注掉本行,提交时需打开
_set_params $@
# _train       # 如果只产出训练log,不解析,可取消注释
_run     # 该函数在run_model.sh中,执行时会调用_train; 如果不联调只产出训练log可以注掉本行,提交时需打开

