from .config import AttrDict

_audio_cfg = AttrDict()

_audio_cfg.num_mels = 80
_audio_cfg.rescale = True
_audio_cfg.rescaling_max = 0.9
_audio_cfg.use_lws = False
_audio_cfg.n_fft = 800
_audio_cfg.hop_size = 200
_audio_cfg.win_size = 800
_audio_cfg.sample_rate = 16000
_audio_cfg.frame_shift_ms = None
_audio_cfg.signal_normalization = True
_audio_cfg.allow_clipping_in_normalization = True
_audio_cfg.symmetric_mels = True
_audio_cfg.max_abs_value = 4.
_audio_cfg.preemphasize = True
_audio_cfg.preemphasis = 0.97
_audio_cfg.min_level_db = -100
_audio_cfg.ref_level_db = 20
_audio_cfg.fmin = 55
_audio_cfg.fmax = 7600
_audio_cfg.fps = 25


def get_audio_config():
    return _audio_cfg
