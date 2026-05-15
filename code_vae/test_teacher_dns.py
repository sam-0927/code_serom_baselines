import os
import re
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from scipy.io import wavfile
from torchaudio.transforms import Spectrogram, InverseSpectrogram
from teacher_pred import Teacher_Pred

# ================= Configuration =================
MODEL_PATH = './output/checkpoint_best.pth'
DNS_ROOT = '/workspace/DB/DNS-Challenge-2020/datasets/test_set/synthetic'
DIR_WRITE = 'result_vae_dns'

REVERB_TAGS = ['no_reverb', 'with_reverb']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model
model = Teacher_Pred()
checkpoint = torch.load(MODEL_PATH, map_location=device)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.to(device)
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
            mask = model(model_input)
            mask_cpx = torch.complex(mask[:, 0], mask[:, 1]).unsqueeze(1)

            # Apply mask + de-compress (power 10/3)
            enhanced_cpx = mask_cpx * inputs_cpx
            enhanced_final = torch.pow(torch.abs(enhanced_cpx), 10/3) * torch.exp(1j * torch.angle(enhanced_cpx))

            # Inverse STFT: (1, 1, T, F) -> (1, F, T) -> waveform
            out_spec = enhanced_final.squeeze(0).permute(0, 2, 1)
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
