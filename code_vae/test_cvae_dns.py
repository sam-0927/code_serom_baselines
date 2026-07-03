import os
import numpy as np
import re
import torch
import soundfile as sf
from pathlib import Path
from scipy.io import wavfile
from torchaudio.transforms import Spectrogram, InverseSpectrogram
from large_cpx import Cruse
from large_vae_cpx import VAE

# ================= Configuration =================
MODEL_PATH = './output_cvae_model/checkpoint_best.pth'
DNS_ROOT = '/workspace/DB/DNS-Challenge-2020/datasets/test_set/synthetic'
DIR_WRITE = './result_cvae_dns'

REVERB_TAGS = ['no_reverb', 'with_reverb']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model
model_pred = Cruse().to(device)
model = VAE().to(device)

# Load pre-trained weights
model_pred_checkpoint = torch.load('output_predictive_model/checkpoint_epoch200.pth', map_location=device)
model_pred.load_state_dict(model_pred_checkpoint['model_state_dict'])
model_checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(model_checkpoint['model_state_dict'])

model_pred.eval()
model.eval()

# STFT transforms
win_fn = lambda x: torch.sqrt(torch.hann_window(x)).to(device)
to_spec = Spectrogram(n_fft=512, hop_length=256, power=None, window_fn=win_fn).to(device)
from_spec = InverseSpectrogram(n_fft=512, hop_length=256, window_fn=win_fn).to(device)

# ================= Processing =================
for reverb_tag in REVERB_TAGS:
    noisy_dir = os.path.join(DNS_ROOT, reverb_tag, 'noisy')
    clean_dir = os.path.join(DNS_ROOT, reverb_tag, 'clean')
    out_dir = os.path.join(DIR_WRITE, reverb_tag)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    noisy_files = sorted(Path(noisy_dir).glob('*.wav'))
    print(f'\n[{reverb_tag}] {len(noisy_files)} files')

    for k, noisy_path in enumerate(noisy_files):
        fname = noisy_path.stem  # e.g. clnsp102_traffic_248091_3_snr0_tl-21_fileid_268

        # Parse fileid and snr from filename
        fileid_match = re.search(r'fileid_(\d+)', fname)
        snr_match = re.search(r'snr(-?\d+)', fname)
        if not fileid_match:
            print(f'  [SKIP] fileid not found: {fname}')
            continue
        fileid = fileid_match.group(1)
        snr = snr_match.group(1) if snr_match else 'unknown'

        clean_path = Path(clean_dir) / f'clean_fileid_{fileid}.wav'
        if not clean_path.exists():
            print(f'  [SKIP] clean not found: {clean_path.name}')
            continue

        # Load noisy wav
        sr, audio = wavfile.read(str(noisy_path))
        input_wav = torch.from_numpy(audio.astype(np.float32) / (2**15)).to(device)
        
        with torch.no_grad():
            # Waveform -> Spectrogram: (F, T) -> (1, 1, T, F)
            X = to_spec(input_wav).permute(1, 0).unsqueeze(0).unsqueeze(0).cfloat()

            # Magnitude compression (power 0.3) + RI concat
            inputs_cpx = torch.pow(torch.abs(X), 0.3) * torch.exp(1j * torch.angle(X))
            model_input = torch.cat((torch.real(inputs_cpx), torch.imag(inputs_cpx)), dim=1)

            # Inference
            outputs_p = model_pred(model_input)
            outputs_p = torch.complex(outputs_p[:,0,:,:], outputs_p[:,1,:,:]).unsqueeze(1) * inputs_cpx
            outputs_p = torch.cat((torch.real(outputs_p), torch.imag(outputs_p)), dim=1)
            vae_in = torch.cat((outputs_p, model_input), dim=1)
            outputs, re_mu, im_mu, r, re_s, im_s = model(vae_in)
            outputs = torch.complex(outputs[:,0,:,:].unsqueeze(1), outputs[:,1,:,:].unsqueeze(1)) * inputs_cpx
            outputs = torch.pow(torch.abs(outputs), (10/3)) * torch.exp(1j * torch.angle(outputs))
            
            out_spec = outputs.squeeze(0).permute(0, 2, 1)
            enhanced_wav = from_spec(out_spec).squeeze()

        enhanced_np = (enhanced_wav.cpu().numpy() * (2**15)).astype(np.int16)
        wavfile.write(os.path.join(out_dir, f'{fname}_enh.wav'), sr, enhanced_np)

        # Save clean reference
        clean_audio, sr_clean = sf.read(str(clean_path))
        clean_np = (clean_audio * (2**15)).astype(np.int16)
        wavfile.write(os.path.join(out_dir, f'{fname}_clean.wav'), sr_clean, clean_np)

        if k % 50 == 0:
            print(f'  [{k}/{len(noisy_files)}] snr={snr} | {noisy_path.name}')

print('\nDone.')
