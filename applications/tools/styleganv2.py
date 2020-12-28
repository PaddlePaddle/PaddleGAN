import paddle
import os
import sys
sys.path.insert(0, os.getcwd())
from ppgan.apps import StyleGANv2Predictor
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

    parser.add_argument("--model_type",
                        type=str,
                        default=None,
                        help="type of model for loading pretrained model")

    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="sample random seed for model's image generation")
                        
    parser.add_argument("--size",
                        type=int,
                        default=1024,
                        help="resolution of output image")
                        
    parser.add_argument("--style_dim",
                        type=int,
                        default=512,
                        help="number of style dimension")
                        
    parser.add_argument("--n_mlp",
                        type=int,
                        default=8,
                        help="number of mlp layer depth")
                        
    parser.add_argument("--channel_multiplier",
                        type=int,
                        default=2,
                        help="number of channel multiplier")

    parser.add_argument("--n_row",
                        type=int,
                        default=3,
                        help="row number of output image grid")

    parser.add_argument("--n_col",
                        type=int,
                        default=5,
                        help="column number of output image grid")

    parser.add_argument("--cpu",
                        dest="cpu",
                        action="store_true",
                        help="cpu mode.")

    args = parser.parse_args()

    if args.cpu:
        paddle.set_device('cpu')

    predictor = StyleGANv2Predictor(
        output_path=args.output_path,
        weight_path=args.weight_path,
        model_type=args.model_type,
        seed=args.seed,
        size=args.size,
        style_dim=args.style_dim,
        n_mlp=args.n_mlp,
        channel_multiplier=args.channel_multiplier
    )
    predictor.run(args.n_row, args.n_col)
