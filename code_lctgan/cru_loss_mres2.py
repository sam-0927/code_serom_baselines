amp = lambda x,c: torch.pow(torch.abs(x),c)
phs_aware = lambda mag, phs, c: torch.pow(mag, c)*torch.exp(phs*1j)

# istft
from torchaudio.transforms import Spectrogram, InverseSpectrogram
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
win = lambda x: torch.sqrt(torch.hann_window(x)).to(device)
from_spec = InverseSpectrogram(n_fft=512, hop_length=256, window_fn = win).to(device)
to_spec1 = Spectrogram(n_fft=320, hop_length=160, power=None, window_fn = win).to(device)
to_spec2 = Spectrogram(n_fft=512, hop_length=256, power=None, window_fn = win).to(device)
to_spec3 = Spectrogram(n_fft=768, hop_length=384, power=None, window_fn = win).to(device)
to_spec4 = Spectrogram(n_fft=256, hop_length=128, power=None, window_fn = win).to(device)

def cruse_loss(spec_pred, S, l=0.3, c=0.3, reduction='mean'):
    
    # spec_pred: [batch, channel, time', frequency']
     # back and forth between F-T
    spec_pred_temp = spec_pred.permute(0,1,3,2) # -> (N, c, F, T)
    S = S.permute(0,1,3,2) # -> (N, c, F, T)
    sig_pred_temp = from_spec(spec_pred_temp)
    s = from_spec(S)# output: (N, c, T)
    error = 0
    for i in range(3):
        if i < 1:
            spec_pred_re = to_spec1(sig_pred_temp).permute(0,1,3,2)
            S = to_spec1(s).permute(0,1,3,2)# -> (N, c, T, F)
        elif i < 2:
            spec_pred_re = to_spec2(sig_pred_temp).permute(0,1,3,2)
            S = to_spec2(s).permute(0,1,3,2)# -> (N, c, T, F)
        elif i < 3:
            spec_pred_re = to_spec3(sig_pred_temp).permute(0,1,3,2)
            S = to_spec3(s).permute(0,1,3,2)# -> (N, c, T, F)
        else:
            spec_pred_re = to_spec4(sig_pred_temp).permute(0,1,3,2)
            S = to_spec4(s).permute(0,1,3,2)# -> (N, c, T, F)
        spec_pred_re[torch.abs(spec_pred_re)<1e-8] = 1e-8+0j
        S[torch.abs(S)<1e-8] = 1e-8+0j
        diff = phs_aware(torch.abs(spec_pred_re), torch.angle(spec_pred_re) ,c) - phs_aware(torch.abs(S), torch.angle(S) ,c)
        stft_cnst_delta = amp(diff, 2)
        amp_delta = amp(amp(spec_pred_re, c) - amp(S, c), 2)    
        errori = (1-l)*amp_delta + l*stft_cnst_delta 
        errori = torch.mean(errori)
        if i == 1:
            error = error + 2 * errori
        else:
            error = error + errori

    return error