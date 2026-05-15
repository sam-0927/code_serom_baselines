# -*- coding: utf-8 -*-
import numpy as np
import os
import torch
import torchaudio
import speechmetrics 
from scipy.io import wavfile
from os import listdir
from pystoi import stoi
from dnsmos_local import dnsmos
from torchaudio.transforms import Spectrogram, InverseSpectrogram

# Import your specific model class
from small_vae import small_VAE

# --- Environment Setup ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dir_input = '/project_ghent/ds/datasets/noisy_test_5s/'
model_path = '.ckps/checkpoint_epoch1486.pth'

# --- Load Model (Strictly keeping original variable names) ---
model = small_VAE()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# --- STFT Config ---
framelen = 512
win = lambda x: torch.sqrt(torch.hann_window(x)).to(device)
to_spec = Spectrogram(n_fft=framelen, hop_length=256, power=None, window_fn=win) 
from_spec = InverseSpectrogram(n_fft=framelen, hop_length=256, window_fn=win) 

# --- Metrics Containers ---
window_length=10
metrics = speechmetrics.load(['pesq','stoi','sisdr'], window_length)
sc_pesq, sc_stoi, sc_sisdr, sc_estoi = [], [], [], []
sc_ovl, sc_bak, sc_sig = [], [], []
mur_all, mui_all, r_all, sr_all, si_all = [], [], [], [], []

file_list = [file for file in listdir(dir_input)]

# --- Inference Loop ---
for i, name in enumerate(file_list):
    noisy_file = os.path.join(dir_input, name, 'noisy.wav')
    gt_file = os.path.join(dir_input, name, 'clean.wav')

    # Load audio and normalize
    sr, noisy_sig = wavfile.read(noisy_file)
    input_t = torch.tensor(noisy_sig.astype(np.float32) / (2 ** 15)).to(device)
    
    sr, audio_gt_raw = wavfile.read(gt_file)
    audio_gt = torch.tensor(audio_gt_raw.astype(np.float32) / (2 ** 15)).to(device)

    # Signal Processing (Power 0.3 Compression)
    spec = to_spec(input_t).cfloat().permute(1, 0).unsqueeze(0).unsqueeze(0)
    inputs_mri_cpx = torch.pow(torch.abs(spec), 0.3) * torch.exp(1j * torch.angle(spec))
    inputs_ri = torch.cat((torch.real(inputs_mri_cpx), torch.imag(inputs_mri_cpx)), dim=1)

    with torch.no_grad():
        # Keep exact return variables for VAE latent stats
        outputs, re_mu, im_mu, r, re_s, im_s = model(inputs_ri)
        
        # Masking and Power 10/3 Reconstruction
        outputs = torch.complex(outputs[:,0,:,:].unsqueeze(1), outputs[:,1,:,:].unsqueeze(1))
        outputs = outputs * inputs_mri_cpx
        outputs = torch.pow(torch.abs(outputs), (10/3)) * torch.exp(1j * torch.angle(outputs))

        # Back to waveform
        s_est = outputs.squeeze().permute(1,0)
        xtilde = from_spec(s_est).cpu().detach().numpy()
    
    # Post-processing
    audio_gt_np = (audio_gt.cpu().detach().numpy() * (2 ** 15)).astype(np.int16)[:79872]
    xtilde_final = (xtilde * (2 ** 15)).astype(np.int16)

    # Write temp files for speechmetrics
    path_est = f'./eva/distill_64stu5161refine/distill_64sturefine_{i}.wav'
    path_gt = f'./eva/gt/gt_{i}.wav'
    wavfile.write(path_est, 16000, xtilde_final)
    wavfile.write(path_gt, 16000, audio_gt_np)

    # Calculate Scores
    scores = metrics(path_est, path_gt)
    scores_dns = dnsmos(path_est)
    
    sc_pesq.append(scores['pesq'])
    sc_stoi.append(scores['stoi'])
    sc_sisdr.append(scores['sisdr'])
    sc_estoi.append(stoi(audio_gt_np, xtilde_final, 16000, extended=True))
    sc_ovl.append(scores_dns['OVRL'])
    sc_bak.append(scores_dns['BAK'])
    sc_sig.append(scores_dns['SIG'])

    # Collect Latent Distributions
    mur_all.append(re_mu.cpu().detach().numpy().squeeze().transpose(1, 0, 2))
    mui_all.append(im_mu.cpu().detach().numpy().squeeze().transpose(1, 0, 2))
    r_all.append(r.cpu().detach().numpy().squeeze().transpose(1, 0, 2))
    sr_all.append(re_s.cpu().detach().numpy().squeeze().transpose(1, 0, 2))
    si_all.append(im_s.cpu().detach().numpy().squeeze().transpose(1, 0, 2))
    
    # print(f"File {i}: PESQ={scores['pesq']:.4f}")
    print(f"File {i}: PESQ={np.mean(scores['pesq']):.4f}")

# --- Final Save Statistics ---
metrics_map = {
    'pesq': sc_pesq, 'stoi': sc_stoi, 'estoi': sc_estoi, 
    'sisdr': sc_sisdr, 'ovl': sc_ovl, 'bak': sc_bak, 'sig': sc_sig
}

for key, val_list in metrics_map.items():
    arr = np.array(val_list)
    print(f"Mean {key.upper()}: {np.mean(arr):.4f}")
    np.save(f'./metric/{key}_distill_stu_5161refine.npy', arr)

# --- Save Latent Arrays ---
latent_map = {
    'mur': mur_all, 'mui': mui_all, 'r': r_all, 'sr': sr_all, 'si': si_all
}

for key, data_list in latent_map.items():
    merged = np.concatenate(data_list, axis=0)
    np.save(f'./bt_v/sturefine_{key}.npy', merged)