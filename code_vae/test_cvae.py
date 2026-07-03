import os
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from scipy.io import wavfile
from torchaudio.transforms import Spectrogram, InverseSpectrogram
from large_cpx import Cruse
from large_vae_cpx import VAE

# ================= Configuration =================
MODEL_PATH = './output_cvae_model/checkpoint_best.pth'
FILELIST_PATH = '/workspace/DB/librispeech_se_snr-515_eval/test-clean/metadata.txt'
DIR_WRITE = './result_cvae'

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
    wavfile.write(os.path.join(out_dir, f'{noisy_name}_enh.wav'), sr, enhanced_np)

    # Save clean reference (supports both WAV and FLAC)
    clean_audio, sr_clean = sf.read(clean_path)
    clean_np = (clean_audio * (2**15)).astype(np.int16)
    clean_name = Path(clean_path).stem
    wavfile.write(os.path.join(out_dir, f'{clean_name}_clean.wav'), sr_clean, clean_np)

    if k % 100 == 0:
        print(f'[{k}/{len(lines)}] {reverb_tag} snr={snr} | {Path(noisy_path).name}')

print('Done.')
