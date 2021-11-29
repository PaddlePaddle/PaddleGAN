
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
