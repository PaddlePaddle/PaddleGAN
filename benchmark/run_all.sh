
#!usr/bin/env bash

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


for model_mode in ${model_mode_list[@]}; do
      eval fp_item_list='$'"${model_mode}_fp_item"
      eval bs_list='$'"${model_mode}_bs_item"
      eval config='$'"${model_mode}_config"
      eval total_iters='$'"${model_mode}_total_iters"
      eval epochs='$'"${model_mode}_epochs"
      eval dataset_web='$'"${model_mode}_dataset_web"
      eval log_interval='$'"${model_mode}_log_interval"
      wget ${dataset_web} -O data/${model_mode}.tar
      tar -vxf data/${model_mode}.tar -C data/
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
            CUDA_VISIBLE_DEVICES=0 benchmark/run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} ${mode} ${max_iter} ${model_mode} ${config} ${log_interval}   #  (5min)
            sleep 60
            echo "index is speed, 8gpus, run_mode is multi_process, begin, ${model_name}"
            run_mode=mp
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash benchmark/run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} ${mode} ${max_iter} ${model_mode} ${config} ${log_interval}
            sleep 60
            done
      done
done
