import paddle
import os
import sys
sys.path.insert(0, os.getcwd())
from ppgan.apps import MPRPredictor
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path",
                        type=str,
                        default='output_dir',
                        help="path to output image dir")

    parser.add_argument("--weight_path",
                        type=str,
                        default=None,
                        help="path to model checkpoint path")

    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="sample random seed for model's image generation")

    parser.add_argument('--images_path', 
                        default=None, 
                        required=True,
                        type=str, 
                        help='Single image or images directory.')

    parser.add_argument('--task', 
                        required=True, 
                        type=str, 
                        help='Task to run', 
                        choices=['Deblurring', 'Denoising', 'Deraining'])
                        
    parser.add_argument("--cpu",
                        dest="cpu",
                        action="store_true",
                        help="cpu mode.")

    args = parser.parse_args()

    if args.cpu:
        paddle.set_device('cpu')

    predictor = MPRPredictor(
        images_path=args.images_path,
        output_path=args.output_path,
        weight_path=args.weight_path,
        seed=args.seed,
        task=args.task
    )
    predictor.run()
