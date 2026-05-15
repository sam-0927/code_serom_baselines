import os
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from scipy.io import wavfile
from torchaudio.transforms import Spectrogram, InverseSpectrogram
from teacher_pred import Teacher_Pred

# ================= Configuration =================
MODEL_PATH = './output/checkpoint_best.pth'
FILELIST_PATH = '/workspace/DB/librispeech_se_snr-515_eval/test-clean/metadata.txt'
DIR_WRITE = './result_vae'

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
with open(FILELIST_PATH, 'r') as f:
    lines = f.readlines()

for k, line in enumerate(lines):
    parts = line.strip().split(' | ')
    clean_path = parts[0].strip()
    noisy_path = parts[2].strip()
    snr = parts[4].strip()

    # Extract reverb condition from noisy path
    if 'with_reverb' in noisy_path:
        reverb_tag = 'with_reverb'
    elif 'without_reverb' in noisy_path or 'no_reverb' in noisy_path:
        reverb_tag = 'without_reverb'
    else:
        reverb_tag = 'unknown'

    out_dir = os.path.join(DIR_WRITE, reverb_tag, f'snr_{snr}')
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    noisy_name = Path(noisy_path).stem.replace(f'_snr{snr}', '')

    # Load noisy wav
    sr, audio = wavfile.read(noisy_path)
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
    wavfile.write(os.path.join(out_dir, f'{noisy_name}_enh.wav'), sr, enhanced_np)

    # Save clean reference (supports both WAV and FLAC)
    clean_audio, sr_clean = sf.read(clean_path)
    clean_np = (clean_audio * (2**15)).astype(np.int16)
    clean_name = Path(clean_path).stem
    wavfile.write(os.path.join(out_dir, f'{clean_name}_clean.wav'), sr_clean, clean_np)

    if k % 100 == 0:
        print(f'[{k}/{len(lines)}] {reverb_tag} snr={snr} | {Path(noisy_path).name}')

print('Done.')
