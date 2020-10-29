import os
import argparse

from ppgan.utils.download import get_path_from_url

CYCLEGAN_URL_ROOT = 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/'

parser = argparse.ArgumentParser(description='download datasets')
parser.add_argument('--name',
                    type=str,
                    required=True,
                    help='dataset name, \
                    support dataset name: apple2orange, summer2winter_yosemite, \
                    horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, \
                    vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, \
                    ae_photos, cityscapes')

if __name__ == "__main__":
    args = parser.parse_args()

    data_url = CYCLEGAN_URL_ROOT + args.name + '.zip'

    if args.name == 'cityscapes':
        data_url = 'https://paddlegan.bj.bcebos.com/datasets/cityscapes.zip'

    path = get_path_from_url(data_url)

    dst = os.path.join('data', args.name)
    print('symlink {} to {}'.format(path, dst))
    os.symlink(path, dst)
