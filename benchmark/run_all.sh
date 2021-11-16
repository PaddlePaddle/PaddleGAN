
#!usr/bin/env bash

export BENCHMARK_ROOT=/workspace
run_env=$BENCHMARK_ROOT/run_env
log_date=`date "+%Y.%m%d.%H%M%S"`
frame=paddle2.1.3
cuda_version=10.2
save_log_dir=${BENCHMARK_ROOT}/logs/${frame}_${log_date}_${cuda_version}/

if [[ -d ${save_log_dir} ]]; then
    rm -rf ${save_log_dir}
fi

# this for update the log_path coding mat
export TRAIN_LOG_DIR=${save_log_dir}/train_log
mkdir -p ${TRAIN_LOG_DIR}
log_path=${TRAIN_LOG_DIR}

################################# 配置python, 如:
rm -rf $run_env
mkdir $run_env
echo `which python3.7`
ln -s $(which python3.7)m-config  $run_env/python3-config
ln -s $(which python3.7) $run_env/python
ln -s $(which pip3.7) $run_env/pip

export PATH=$run_env:${PATH}
cd $BENCHMARK_ROOT
pip install -v -e .


function parse_yaml {
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      if (indent == 0) {
         model_mode_list[model_num]=$2;
         printf("model_mode_list[%d]=%s\n",(model_num), $2);
         printf("model_num=%d\n", (model_num+1));
         model_num=(model_num+1);
      }
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) >= 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s=\"%s\"\n",vn, $2, $3);
      }
   }'
}
eval $(parse_yaml "benchmark/benchmark.yaml")

profile=${1:-"off"}

for model_mode in ${model_mode_list[@]}; do
      eval fp_item_list='$'"${model_mode}_fp_item"
      eval bs_list='$'"${model_mode}_bs_item"
      eval config='$'"${model_mode}_config"
      eval total_iters='$'"${model_mode}_total_iters"
      eval epochs='$'"${model_mode}_epochs"
      eval dataset_web='$'"${model_mode}_dataset_web"
      eval dataset='$'"${model_mode}_dataset"
      eval log_interval='$'"${model_mode}_log_interval"
      if [ -n "$dataset" ]; then
          cp -r ${dataset} data/
      else
          wget ${dataset_web} -O data/${model_mode}.tar
          tar -vxf data/${model_mode}.tar -C data/
      fi
      if [ -n "$total_iters" ]; then
          mode="total_iters"
          max_iter=$total_iters
      else
          mode="epochs"
          max_iter=$epochs
      fi
      echo ${epochs}
      for fp_item in ${fp_item_list[@]}; do
          for bs_item in ${bs_list[@]}
            do
            echo "index is speed, 1gpus, begin, ${model_name}"
            run_mode=sp
            CUDA_VISIBLE_DEVICES=0 benchmark/run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} ${mode} ${max_iter} ${model_mode} ${config} ${log_interval} ${profile}  #  (5min)
            sleep 60
            echo "index is speed, 8gpus, run_mode is multi_process, begin, ${model_name}"
            run_mode=mp
            basicvsr_name=basicvsr
            if [ ${model_mode} = ${basicvsr_name} ]; then
                CUDA_VISIBLE_DEVICES=0,1,2,3 bash benchmark/run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} ${mode} ${max_iter} ${model_mode} ${config} ${log_interval} ${profile}
            else
                CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash benchmark/run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} ${mode} ${max_iter} ${model_mode} ${config} ${log_interval} ${profile}
            fi
            sleep 60
            done
      done
done
