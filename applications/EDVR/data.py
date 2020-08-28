import cv2

import numpy as np

def read_img(path, size=None, is_gt=False):
    """read image by cv2
    return: Numpy float32, HWC, BGR, [0,1]"""
    # print('debug:', path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    
    if img.shape[2] > 3:
        img = img[:, :, :3] 
    return img 

def get_test_neighbor_frames(crt_i, N, max_n, padding='new_info'):
    """Generate an index list for reading N frames from a sequence of images
    Args:
        crt_i (int): current center index
        max_n (int): max number of the sequence of images (calculated from 1)
        N (int): reading N frames
        padding (str): padding mode, one of replicate | reflection | new_info | circle
            Example: crt_i = 0, N = 5
            replicate: [0, 0, 0, 1, 2]
            reflection: [2, 1, 0, 1, 2]
            new_info: [4, 3, 0, 1, 2]
            circle: [3, 4, 0, 1, 2]

    Returns:
        return_l (list [int]): a list of indexes
    """
    max_n = max_n - 1
    n_pad = N // 2
    return_l = []

    for i in range(crt_i - n_pad, crt_i + n_pad + 1):
        if i < 0:
            if padding == 'replicate':
                add_idx = 0
            elif padding == 'reflection':
                add_idx = -i
            elif padding == 'new_info':
                add_idx = (crt_i + n_pad) + (-i)
            elif padding == 'circle':
                add_idx = N + i
            else:
                raise ValueError('Wrong padding mode')
        elif i > max_n:
            if padding == 'replicate':
                add_idx = max_n
            elif padding == 'reflection':
                add_idx = max_n * 2 - i
            elif padding == 'new_info':
                add_idx = (crt_i - n_pad) - (i - max_n)
            elif padding == 'circle':
                add_idx = i - N
            else:
                raise ValueError('Wrong padding mode')
        else:
            add_idx = i
        return_l.append(add_idx)
    # name_b = '{:08d}'.format(crt_i)    
    return return_l


class EDVRDataset:
    def __init__(self, frame_paths):
        self.frames = frame_paths


    def __getitem__(self, index):
        indexs = get_test_neighbor_frames(index, 5, len(self.frames))
        frame_list = []
        for i in indexs:
            img = read_img(self.frames[i])
            frame_list.append(img)

        img_LQs = np.stack(frame_list, axis=0)
        print('img:', img_LQs.shape)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        img_LQs = np.transpose(img_LQs, (0, 3, 1, 2)).astype('float32')

        return img_LQs, self.frames[index]

    def __len__(self):
        return len(self.frames)