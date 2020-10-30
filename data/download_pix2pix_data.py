import os
import argparse

from ppgan.utils.download import get_path_from_url

PIX2PIX_URL_ROOT = 'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/'

parser = argparse.ArgumentParser(description='download datasets')
parser.add_argument('--name',
                    type=str,
                    required=True,
                    help='dataset name, \
                    support dataset name: cityscapes, night2day, edges2handbags, \
                    edges2shoes, facades, maps')

if __name__ == "__main__":
    args = parser.parse_args()

    data_url = PIX2PIX_URL_ROOT + args.name + '.tar.gz'

    path = get_path_from_url(data_url)

    dst = os.path.join('data', args.name)
    print('symlink {} to {}'.format(path, dst))
    os.symlink(path, dst)
