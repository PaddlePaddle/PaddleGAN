#!/bin/bash

BASEDIR=$(dirname "$0")

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

REPO_ROOT_PATH=$(readlinkf ${BASEDIR}/../)

config_files=$(find ${REPO_ROOT_PATH}/test_tipc/configs -name "train_infer_python.txt")
for file in ${config_files}; do
   echo $file
   sed -i 's/--device:gpu/--device:npu/g' $file
done
