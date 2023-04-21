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

# change total iters/epochs for npu/xpu to accelaration
modelname=$(echo $FILENAME | cut -d '/' -f3)
if  [ $modelname == "Pix2pix" ]; then
    sed -i "s/lite_train_lite_infer=10/lite_train_lite_infer=1/g" $FILENAME
fi

if  [ $modelname == "edvr" ]; then
    sed -i "s/lite_train_lite_infer=100/lite_train_lite_infer=10/g" $FILENAME
fi

# change gpu to npu in execution script
sed -i 's/\"gpu\"/\"npu\"/g' test_tipc/test_train_inference_python.sh
sed -i 's/--gpus/--npus/g' test_tipc/test_train_inference_python.sh

# parser params
IFS=$'\n'
lines=(${dataline})

# pass parameters to test_train_inference_python.sh
cmd="bash test_tipc/test_train_inference_python.sh ${FILENAME} $2"
echo $cmd
eval $cmd
