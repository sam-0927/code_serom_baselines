import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class MelSpectrogramLoss(nn.Module):
    """
    L1 loss on log-mel-spectrogram.

    log_mel = log(MelSpectrogram(x) + eps)
    loss    = L1(log_mel_estimated, log_mel_target)

    Parameters
    ----------
    sample_rate  : audio sample rate (default 16000)
    n_fft        : FFT size
    hop_length   : hop size in samples
    win_length   : window size in samples
    n_mels       : number of mel filter banks
    f_min        : lowest mel frequency
    f_max        : highest mel frequency (None → sample_rate / 2)
    eps          : floor before log to avoid log(0)
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft:       int = 1024,
        hop_length:  int = 256,
        win_length:  int = 1024,
        n_mels:      int = 80,
        f_min:       float = 0.0,
        f_max:       float | None = None,
        eps:         float = 1e-5,
    ):
        super().__init__()
        self.eps = eps
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max if f_max is not None else sample_rate / 2,
            power=1.0,          # amplitude spectrogram
        )

    def forward(self, estimated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        estimated : (B, T)  generated waveform
        target    : (B, T)  clean reference waveform

        Returns
        -------
        Scalar L1 loss on log-mel-spectrograms.
        """
        mel_e = self.mel_transform(estimated)           # (B, n_mels, T_frames)
        mel_t = self.mel_transform(target)
        log_mel_e = (mel_e + self.eps).log()
        log_mel_t = (mel_t + self.eps).log()
        return F.l1_loss(log_mel_e, log_mel_t)