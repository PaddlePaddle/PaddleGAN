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


def dump_frames_ffmpeg(vid_path, outpath, r=None, ss=None, t=None):
    ffmpeg = ['ffmpeg ', ' -y -loglevel ', ' error ']
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(outpath, vid_name)

    if not os.path.exists(out_full_path):
        os.makedirs(out_full_path)

    # video file name
    outformat = out_full_path + '/%08d.png'

    if ss is not None and t is not None and r is not None:
        cmd = ffmpeg + [
            ' -ss ',
            ss,
            ' -t ',
            t,
            ' -i ',
            vid_path,
            ' -r ',
            r,
            # ' -f ', ' image2 ',
            #                        ' -s ', ' 960*540 ',
            ' -qscale:v ',
            ' 0.1 ',
            ' -start_number ',
            ' 0 ',
            # ' -qmax ', ' 1 ',
            outformat
        ]
    else:
        cmd = ffmpeg + [' -i ', vid_path, ' -start_number ', ' 0 ', outformat]

    cmd = ''.join(cmd)

    if os.system(cmd) == 0:
        pass
    else:
        print('ffmpeg process video: {} error'.format(vid_name))

    sys.stdout.flush()
    return out_full_path


def frames_to_video_ffmpeg(framepath, videopath, r):
    ffmpeg = ['ffmpeg ', ' -y -loglevel ', ' error ']
    cmd = ffmpeg + [
        ' -r ', r, ' -f ', ' image2 ', ' -i ', framepath, ' -vcodec ',
        ' libx264 ', ' -pix_fmt ', ' yuv420p ', ' -crf ', ' 16 ', videopath
    ]
    cmd = ''.join(cmd)

    if os.system(cmd) == 0:
        pass
    else:
        print('ffmpeg process video: {} error'.format(videopath))

    sys.stdout.flush()


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
