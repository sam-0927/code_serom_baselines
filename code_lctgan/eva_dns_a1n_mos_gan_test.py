import os
import torch
import numpy as np
from pathlib import Path
from scipy.io import wavfile
import torchaudio.transforms as T
import soundfile as sf
from lct_la1n import Cruse

# ================= Configuration =================
MODEL_PATH = './output/checkpoint_best.pth'
FILELIST_PATH = '/workspace/DB/librispeech_se_snr-515_eval/test-clean/metadata.txt'
DIR_WRITE = './result_lctgan'

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
with open(FILELIST_PATH, 'r') as f:
    lines = f.readlines()

for k, line in enumerate(lines):
    parts = line.strip().split(' | ')
    clean_path = parts[0].strip()
    noisy_path = parts[2].strip()
    snr = parts[4].strip()

    # Extract reverb condition from noisy path (with_reverb / without_reverb)
    if 'with_reverb' in noisy_path:
        reverb_tag = 'with_reverb'
    elif 'without_reverb' in noisy_path:
        reverb_tag = 'without_reverb'
    else:
        reverb_tag = 'unknown'

    out_dir = os.path.join(DIR_WRITE, reverb_tag, f'snr_{snr}')
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    stem = Path(noisy_path).stem

    # Load and enhance noisy
    sr, audio = wavfile.read(noisy_path)
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

    # Save clean reference (supports flac and wav)
    clean_audio_float, sr_clean = sf.read(clean_path)
    clean_audio = (clean_audio_float * (2**15)).astype(np.int16)
    wavfile.write(os.path.join(out_dir, f'{stem}_clean.wav'), sr_clean, clean_audio)

    if k % 100 == 0:
        print(f'[{k}/{len(lines)}] {reverb_tag} snr={snr} | {Path(noisy_path).name}')

print('Done.')
