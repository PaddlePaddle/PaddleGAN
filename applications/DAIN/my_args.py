import os
import datetime
import argparse
import numpy
import networks

modelnames = networks.__all__
# import datasets
datasetNames = ('Vimeo_90K_interp')  #datasets.__all__

parser = argparse.ArgumentParser(description='DAIN')

parser.add_argument('--debug', action='store_true', help='Enable debug mode')
parser.add_argument('--netName',
                    type=str,
                    default='DAIN',
                    choices=modelnames,
                    help='model architecture: ' + ' | '.join(modelnames) +
                    ' (default: DAIN)')

parser.add_argument('--datasetName',
                    default='Vimeo_90K_interp',
                    choices=datasetNames,
                    nargs='+',
                    help='dataset type : ' + ' | '.join(datasetNames) +
                    ' (default: Vimeo_90K_interp)')
parser.add_argument('--video_path',
                    default='',
                    help='the path of selected videos')
parser.add_argument('--output_path', default='', help='the output root path')

parser.add_argument('--seed',
                    type=int,
                    default=1,
                    help='random seed (default: 1)')

parser.add_argument('--batch_size',
                    '-b',
                    type=int,
                    default=1,
                    help='batch size (default:1)')
parser.add_argument('--channels',
                    '-c',
                    type=int,
                    default=3,
                    choices=[1, 3],
                    help='channels of images (default:3)')
parser.add_argument('--filter_size',
                    '-f',
                    type=int,
                    default=4,
                    help='the size of filters used (default: 4)',
                    choices=[2, 4, 6, 5, 51])

parser.add_argument('--time_step',
                    type=float,
                    default=0.5,
                    help='choose the time steps')
parser.add_argument(
    '--alpha',
    type=float,
    nargs='+',
    default=[0.0, 1.0],
    help=
    'the ration of loss for interpolated and rectified result (default: [0.0, 1.0])'
)
parser.add_argument('--frame_rate',
                    type=int,
                    default=None,
                    help='frame rate of the input video')

parser.add_argument('--patience',
                    type=int,
                    default=5,
                    help='the patience of reduce on plateou')
parser.add_argument('--factor',
                    type=float,
                    default=0.2,
                    help='the factor of reduce on plateou')

parser.add_argument('--saved_model',
                    type=str,
                    default='',
                    help='path to the model weights')
parser.add_argument('--no-date',
                    action='store_true',
                    help='don\'t append date timestamp to folder')
parser.add_argument('--use_cuda',
                    default=True,
                    type=bool,
                    help='use cuda or not')
parser.add_argument('--use_cudnn', default=1, type=int, help='use cudnn or not')

# args = parser.parse_args()
