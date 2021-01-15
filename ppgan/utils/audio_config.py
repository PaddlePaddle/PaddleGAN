from easydict import EasyDict as edict

_C = edict()

_C.num_mels = 80
_C.rescale = True
_C.rescaling_max = 0.9
_C.use_lws = False
_C.n_fft = 800
_C.hop_size = 200
_C.win_size = 800
_C.sample_rate = 16000
_C.frame_shift_ms = None
_C.signal_normalization = True
_C.allow_clipping_in_normalization = True
_C.symmetric_mels = True
_C.max_abs_value = 4.
_C.preemphasize = True
_C.preemphasis = 0.97
_C.min_level_db = -100
_C.ref_level_db = 20
_C.fmin = 55
_C.fmax = 7600
_C.fps = 25


def get_audio_config():
    return _C
