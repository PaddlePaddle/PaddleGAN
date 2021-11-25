import numpy as np
import os
import subprocess
import json
import argparse
import glob


def init_args():
    parser = argparse.ArgumentParser()
    # params for testing assert allclose
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--gt_file", type=str, default="")
    parser.add_argument("--log_file", type=str, default="")
    parser.add_argument("--precision", type=str, default="fp32")
    return parser

def parse_args():
    parser = init_args()
    return parser.parse_args()

def load_from_file(gt_file):
    if not os.path.exists(gt_file):
        raise ValueError("The log file {} does not exists!".format(gt_file))
    with open(gt_file, 'r') as f:
        data = f.readlines()
        f.close()
    parser_gt = {}
    for line in data:
        metric_name, result = line.strip("\n").split(":")
        parser_gt[metric_name] = float(result)
    return parser_gt

if __name__ == "__main__":
    # Usage:
    # python3.7 test_tipc/compare_results.py --gt_file=./test_tipc/results/*.txt  --log_file=./test_tipc/output/*/*.txt

    args = parse_args()

    gt_collection = load_from_file(args.gt_file)
    pre_collection = load_from_file(args.log_file)

    for metric in pre_collection.keys():
        try:
            np.testing.assert_allclose(
                np.array(pre_collection[metric]), np.array(gt_collection[metric]), atol=args.atol, rtol=args.rtol)
            print(
                "Assert allclose passed! The results of {} are consistent!".
                format(metric))
        except Exception as E:
            print(E)
            raise ValueError(
                "The results of {} are inconsistent!".
                format(metric))