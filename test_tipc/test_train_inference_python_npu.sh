#!/bin/bash
source test_tipc/common_func.sh

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

function func_parser_config() {
    strs=$1
    IFS=" "
    array=(${strs})
    tmp=${array[2]}
    echo ${tmp}
}

BASEDIR=$(dirname "$0")
REPO_ROOT_PATH=$(readlinkf ${BASEDIR}/../)

FILENAME=$1

# change gpu to npu in tipc txt configs
sed -i "s/state=GPU/state=NPU/g" $FILENAME
sed -i "s/--device:gpu/--device:npu/g" $FILENAME
sed -i "s/--benchmark:True/--benchmark:False/g" $FILENAME
dataline=`cat $FILENAME`

# parser params
IFS=$'\n'
lines=(${dataline})

# change total iters/epochs for npu/xpu to accelaration
modelname=$(func_parser_value "${lines[1]}")
echo $modelname
if  [ $modelname == "Pix2pix" ]; then
    sed -i "s/lite_train_lite_infer=10/lite_train_lite_infer=1/g" $FILENAME
    sed -i "s/-o log_config.interval=1/-o log_config.interval=1 snapshot_config.interval=1/g" $FILENAME
fi

if  [ $modelname == "edvr" ]; then
    sed -i "s/lite_train_lite_infer=100/lite_train_lite_infer=10/g" $FILENAME
    sed -i "s/snapshot_config.interval=25/snapshot_config.interval=5/g" $FILENAME
fi

# change gpu to npu in execution script
sed -i 's/\"gpu\"/\"npu\"/g' test_tipc/test_train_inference_python.sh
sed -i 's/--gpus/--npus/g' test_tipc/test_train_inference_python.sh

# pass parameters to test_train_inference_python.sh
cmd="bash test_tipc/test_train_inference_python.sh ${FILENAME} $2"
echo $cmd
eval $cmd
