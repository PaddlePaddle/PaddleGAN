import os, sys
import glob
import shutil


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def combine_frames(input, interpolated, combined, num_frames):
    frames1 = sorted(glob.glob(os.path.join(input, '*.png')))
    frames2 = sorted(glob.glob(os.path.join(interpolated, '*.png')))
    num1 = len(frames1)
    num2 = len(frames2)
    # assert (num1 - 1) * num_frames == num2
    for i in range(num1):
        src = frames1[i]
        imgname = int(src.split('/')[-1].split('.')[-2])
        assert i == imgname
        dst = os.path.join(combined, '{:08d}.png'.format(i * (num_frames + 1)))
        shutil.copy2(src, dst)
        if i < num1 - 1:
            try:
                for k in range(num_frames):
                    src = frames2[i * num_frames + k]
                    dst = os.path.join(
                        combined,
                        '{:08d}.png'.format(i * (num_frames + 1) + k + 1))
                    shutil.copy2(src, dst)
            except Exception as e:
                print(e)
                print(len(frames2), num_frames, i, k, i * num_frames + k)
