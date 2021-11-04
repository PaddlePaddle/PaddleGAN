# code was heavily based on https://github.com/Rudrabha/Wav2Lip
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/Rudrabha/Wav2Lip#license-and-citation

import numpy as np
from scipy import signal
from scipy.io import wavfile
from paddle.utils import try_import
from .audio_config import get_audio_config

audio_config = get_audio_config()


def load_wav(path, sr):
    librosa = try_import('librosa')
    return librosa.core.load(path, sr=sr)[0]


def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    #proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))


def save_wavenet_wav(wav, path, sr):
    librosa = try_import('librosa')
    librosa.output.write_wav(path, wav, sr=sr)


def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav


def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav


def get_hop_size():
    hop_size = audio_config.hop_size
    if hop_size is None:
        assert audio_config.frame_shift_ms is not None
        hop_size = int(audio_config.frame_shift_ms / 1000 *
                       audio_config.sample_rate)
    return hop_size


def linearspectrogram(wav):
    D = _stft(
        preemphasis(wav, audio_config.preemphasis, audio_config.preemphasize))
    S = _amp_to_db(np.abs(D)) - audio_config.ref_level_db

    if audio_config.signal_normalization:
        return _normalize(S)
    return S


def melspectrogram(wav):
    D = _stft(
        preemphasis(wav, audio_config.preemphasis, audio_config.preemphasize))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - audio_config.ref_level_db

    if audio_config.signal_normalization:
        return _normalize(S)
    return S


def _lws_processor():
    import lws
    return lws.lws(audio_config.n_fft,
                   get_hop_size(),
                   fftsize=audio_config.win_size,
                   mode="speech")


def _stft(y):
    if audio_config.use_lws:
        return _lws_processor(audio_config).stft(y).T
    else:
        librosa = try_import('librosa')
        return librosa.stft(y=y,
                            n_fft=audio_config.n_fft,
                            hop_length=get_hop_size(),
                            win_length=audio_config.win_size)


##########################################################
#Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r


##########################################################
#Librosa correct padding
def librosa_pad_lr(x, fsize, fshift):
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]


# Conversions
_mel_basis = None


def _linear_to_mel(spectogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)


def _build_mel_basis():
    assert audio_config.fmax <= audio_config.sample_rate // 2
    librosa = try_import('librosa')
    return librosa.filters.mel(audio_config.sample_rate,
                               audio_config.n_fft,
                               n_mels=audio_config.num_mels,
                               fmin=audio_config.fmin,
                               fmax=audio_config.fmax)


def _amp_to_db(x):
    min_level = np.exp(audio_config.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)


def _normalize(S):
    if audio_config.allow_clipping_in_normalization:
        if audio_config.symmetric_mels:
            return np.clip(
                (2 * audio_config.max_abs_value) *
                ((S - audio_config.min_level_db) /
                 (-audio_config.min_level_db)) - audio_config.max_abs_value,
                -audio_config.max_abs_value, audio_config.max_abs_value)
        else:
            return np.clip(
                audio_config.max_abs_value * ((S - audio_config.min_level_db) /
                                              (-audio_config.min_level_db)), 0,
                audio_config.max_abs_value)

    assert S.max() <= 0 and S.min() - audio_config.min_level_db >= 0
    if audio_config.symmetric_mels:
        return (2 * audio_config.max_abs_value) * (
            (S - audio_config.min_level_db) /
            (-audio_config.min_level_db)) - audio_config.max_abs_value
    else:
        return audio_config.max_abs_value * ((S - audio_config.min_level_db) /
                                             (-audio_config.min_level_db))


def _denormalize(D):
    if audio_config.allow_clipping_in_normalization:
        if audio_config.symmetric_mels:
            return (((np.clip(D, -audio_config.max_abs_value,
                              audio_config.max_abs_value) +
                      audio_config.max_abs_value) * -audio_config.min_level_db /
                     (2 * audio_config.max_abs_value)) +
                    audio_config.min_level_db)
        else:
            return ((np.clip(D, 0, audio_config.max_abs_value) *
                     -audio_config.min_level_db / audio_config.max_abs_value) +
                    audio_config.min_level_db)

    if audio_config.symmetric_mels:
        return (((D + audio_config.max_abs_value) * -audio_config.min_level_db /
                 (2 * audio_config.max_abs_value)) + audio_config.min_level_db)
    else:
        return ((D * -audio_config.min_level_db / audio_config.max_abs_value) +
                audio_config.min_level_db)
