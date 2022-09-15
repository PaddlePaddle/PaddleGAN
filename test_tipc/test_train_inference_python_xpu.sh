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
sed -i "s/state=GPU/state=XPU/g" $FILENAME
sed -i "s/--device:gpu/--device:xpu/g" $FILENAME
sed -i "s/--benchmark:True/--benchmark:False/g" $FILENAME
dataline=`cat $FILENAME`

# change gpu to npu in execution script
sed -i 's/\"gpu\"/\"xpu\"/g' test_tipc/test_train_inference_python.sh
sed -i 's/--gpus/--xpus/g' test_tipc/test_train_inference_python.sh

# parser params
IFS=$'\n'
lines=(${dataline})

# pass parameters to test_train_inference_python.sh
cmd="bash test_tipc/test_train_inference_python.sh ${FILENAME} $2"
echo $cmd
eval $cmd
