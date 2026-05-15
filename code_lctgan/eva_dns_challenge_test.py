import os
import re
import torch
import numpy as np
from pathlib import Path
from scipy.io import wavfile
import torchaudio.transforms as T
from lct_la1n import Cruse

# ================= Configuration =================
MODEL_PATH = './output/checkpoint_best.pth'
DNS_TEST_ROOT = '/workspace/DB/DNS-Challenge-2020/datasets/test_set/synthetic'
DIR_WRITE = './result_lctgan_dns'

REVERB_CONDITIONS = ['no_reverb', 'with_reverb']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model
model = Cruse()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Spectrogram transforms
win_fn = lambda x: torch.sqrt(torch.hann_window(x)).to(device)
to_spec = T.Spectrogram(n_fft=512, hop_length=256, power=None, window_fn=win_fn).to(device)
from_spec = T.InverseSpectrogram(n_fft=512, hop_length=256, window_fn=win_fn).to(device)

# ================= Processing =================
for reverb_tag in REVERB_CONDITIONS:
    noisy_dir = os.path.join(DNS_TEST_ROOT, reverb_tag, 'noisy')
    clean_dir = os.path.join(DNS_TEST_ROOT, reverb_tag, 'clean')
    out_dir = os.path.join(DIR_WRITE, reverb_tag)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    noisy_files = sorted(Path(noisy_dir).glob('*.wav'))
    print(f'\n[{reverb_tag}] {len(noisy_files)} files')

    for noisy_path in noisy_files:
        # Match clean file via fileid
        match = re.search(r'fileid_(\d+)', noisy_path.stem)
        if match is None:
            print(f'  WARNING: fileid not found in {noisy_path.name}, skipping.')
            continue
        fileid = match.group(1)
        clean_path = Path(clean_dir) / f'clean_fileid_{fileid}.wav'

        if not clean_path.exists():
            print(f'  WARNING: clean file not found: {clean_path}, skipping.')
            continue

        stem = noisy_path.stem  # e.g. clnsp102_traffic_..._fileid_268

        # Load and enhance noisy
        sr, audio = wavfile.read(str(noisy_path))
        input_wav = torch.from_numpy(audio.astype(np.float32) / (2**15)).to(device)

        spec = to_spec(input_wav).permute(1, 0).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            in_mag = torch.pow(torch.abs(spec), 0.3)
            mask_preds = model(in_mag)
            out_mag = torch.pow(in_mag * mask_preds, 10/3)
            out_spec = out_mag * torch.exp(1j * torch.angle(spec))
            enhanced_wav = from_spec(out_spec.squeeze(0).permute(0, 2, 1)).squeeze()

        enhanced_wav = (enhanced_wav.cpu().numpy() * (2**15)).astype(np.int16)
        wavfile.write(os.path.join(out_dir, f'{stem}_enh.wav'), sr, enhanced_wav)

        # Save clean reference
        sr_clean, clean_audio = wavfile.read(str(clean_path))
        wavfile.write(os.path.join(out_dir, f'{stem}_clean.wav'), sr_clean, clean_audio)

        print(f'  {stem}')

print('\nDone.')
