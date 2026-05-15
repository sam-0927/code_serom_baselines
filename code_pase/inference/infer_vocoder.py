# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

import os
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
from omegaconf import OmegaConf
from models.wavlm.feature_extractor import WavLM_feat
from models.vocoder.wavlmdec import WavLMDec as Model
from losses import MelSpectrogramLoss


@torch.inference_mode()
def infer(args):
    cfg_infer = OmegaConf.load(args.config)
    cfg_network = OmegaConf.load(cfg_infer.network.config)

    save_folder = cfg_infer.network.enh_folder
    os.makedirs(save_folder, exist_ok=True)

    # load unique clean paths from filelist (clean | noise | noisy | text | snr)
    seen = set()
    wavs = []
    with open(cfg_infer.test_dataset.filelist) as f:
        for line in f:
            clean_path = line.strip().split(' | ')[0]
            if clean_path not in seen:
                seen.add(clean_path)
                wavs.append(clean_path)
    print(f"Inference on {len(wavs)} unique clean files")

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    encoder = WavLM_feat(**cfg_network['encoder_config']).to(device).eval()
    model = Model(**cfg_network['vocoder_config']).to(device).eval()

    model.load_state_dict(
        torch.load(cfg_infer['network']['checkpoint'], map_location=device)['generator']
    )

    mel_loss_fn = MelSpectrogramLoss().to(device)

    inf_scp_list = []
    ref_scp_list = []
    total_mel_loss = 0.0

    for wav_path in tqdm(wavs):
        true_wav, fs = sf.read(wav_path, dtype='float32')

        input = torch.FloatTensor(true_wav)[None,None].to(device)

        feat = encoder(input)
        output  = model(feat)

        esti_wav = output.cpu().detach().numpy().squeeze()
        esti_wav = esti_wav / np.max(np.abs(esti_wav)) * 0.9

        if esti_wav.shape[-1] < true_wav.shape[-1]:
            esti_wav = np.pad(esti_wav, (0, true_wav.shape[-1]-esti_wav.shape[-1]))
        else:
            esti_wav = esti_wav[..., :true_wav.shape[-1]]

        esti_tensor = torch.FloatTensor(esti_wav)[None].to(device)
        true_tensor = torch.FloatTensor(true_wav)[None].to(device)
        mel_loss = mel_loss_fn(esti_tensor, true_tensor).item()
        total_mel_loss += mel_loss

        uid = os.path.splitext(os.path.basename(wav_path))[0]
        true_path = wav_path
        esti_path = os.path.join(save_folder, f'{uid}.wav')

        sf.write(esti_path, esti_wav, fs)

        inf_scp_list.append([uid, esti_path])
        ref_scp_list.append([uid, true_path])

    avg_mel_loss = total_mel_loss / len(wavs)
    print(f"\nMelSpectrogramLoss (avg over {len(wavs)} files): {avg_mel_loss:.4f}")
        
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

    args = parser.parse_args()
    infer(args)
