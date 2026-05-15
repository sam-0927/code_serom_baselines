# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

import os
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
from librosa.util import find_files
from omegaconf import OmegaConf
from models.wavlm.feature_extractor import WavLM_feat as Encoder
from models.vocoder.wavlmdec_dual import WavLMDec as Model
import re

def load_filelist(filelist_path):
    """
    Parse filelist with format: clean | noise | noisy | text | snr
    Returns list of (noisy_path, clean_path, snr) tuples.
    """
    entries = []
    with open(filelist_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split('|')]
            clean_path = parts[0]
            noisy_path = parts[2]
            snr = parts[4]
            entries.append((noisy_path, clean_path, snr))
    return entries


@torch.inference_mode()
def infer(args):
    cfg_infer = OmegaConf.load(args.config)
    cfg_network = OmegaConf.load(cfg_infer.network.config)

    save_folder = args.output_dir if args.output_dir else cfg_infer.network.enh_folder
    os.makedirs(save_folder, exist_ok=True)

    # Build (noisy_path, clean_path, snr) tuples from filelist or folder
    if args.filelist:
        entries = load_filelist(args.filelist)
        print(f"Inference on filelist: {args.filelist}, {len(entries)} files")
    else:
        noisy_folder = cfg_infer.test_dataset.noisy_dir
        clean_folder = cfg_infer.test_dataset.clean_dir
        ext = cfg_infer.test_dataset.extension
        wavs = sorted(find_files(noisy_folder, ext=ext))
        print(f"Inference on folder: {noisy_folder}, {len(wavs)} files")
        entries = [
            (w, os.path.join(clean_folder, os.path.basename(w)), None) for w in wavs
        ]

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    encoder = Encoder(**cfg_network['encoder_config']).to(device)
    model = Model(**cfg_network['vocoder_config']).to(device).eval()

    model.load_state_dict(
        torch.load(cfg_infer['network']['checkpoint'], map_location=device)['generator']
    )

    inf_scp_list = []
    ref_scp_list = []

    for wav_path, clean_path, snr in tqdm(entries):
        true_wav, fs = sf.read(wav_path, dtype='float32')

        input = torch.FloatTensor(true_wav)[None, None].to(device)

        feat_a, feat_p = encoder(input)
        output = model(feat_p, feat_a)

        esti_wav = output.cpu().detach().numpy().squeeze()
        esti_wav = esti_wav / np.max(np.abs(esti_wav)) * 0.9

        if esti_wav.shape[-1] < true_wav.shape[-1]:
            esti_wav = np.pad(esti_wav, (0, true_wav.shape[-1] - esti_wav.shape[-1]))
        else:
            esti_wav = esti_wav[..., :true_wav.shape[-1]]

        # Directory structure: save_folder/snr_{snr}/{with_reverb|without_reverb}/
        subdir = os.path.basename(os.path.dirname(wav_path))
        stem = os.path.splitext(os.path.basename(wav_path))[0]
        stem = re.sub(r'_snr[^_]*', '', stem)

        if snr is not None:
            out_subdir = os.path.join(save_folder, subdir, f"snr_{snr}")
            uid = f"{subdir}_snr_{snr}_{stem}"
        else:
            out_subdir = os.path.join(save_folder, subdir)
            uid = f"{subdir}_{stem}"

        os.makedirs(out_subdir, exist_ok=True)

        enh_path = os.path.join(out_subdir, f"{stem}_enh.wav")
        clean_save_path = os.path.join(out_subdir, f"{stem}_clean.wav")

        sf.write(enh_path, esti_wav, fs)

        # Save clean reference wav (read original and write as wav)
        clean_wav, clean_fs = sf.read(clean_path, dtype='float32')
        sf.write(clean_save_path, clean_wav, clean_fs)

        inf_scp_list.append([uid, enh_path])
        ref_scp_list.append([uid, clean_save_path])

    # Save paths into scp file for evaluation
    with open(os.path.join(save_folder, "inf.scp"), "w") as f:
        for uid, audio_path in inf_scp_list:
            f.write(f"{uid} {audio_path}\n")

    with open(os.path.join(save_folder, "ref.scp"), "w") as f:
        for uid, audio_path in ref_scp_list:
            f.write(f"{uid} {audio_path}\n")

            

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--config', default='configs/cfg_infer.yaml')
    parser.add_argument('-D', '--device', default='0', help='Index of the gpu device')
    parser.add_argument('--filelist', type=str, default='/workspace/DB/librispeech_se_snr-515_eval/test-clean/metadata.txt',
                        help='Filelist with format: clean | noise | noisy | text | snr')
    parser.add_argument('--output_dir', type=str, default='result',
                        help='Output directory for enhanced files (overrides config)')

    args = parser.parse_args()
    infer(args)
