# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""
infer_vocoder_dual_dns.py — DNS-Challenge-2020 synthetic test set 전용 inference

DNS 데이터 구조:
  {dns_root}/no_reverb/noisy/clnsp..._snrN_..._fileid_XXX.wav
  {dns_root}/no_reverb/clean/clean_fileid_XXX.wav
  {dns_root}/with_reverb/noisy/clnsp..._snrN_..._fileid_XXX.wav
  {dns_root}/with_reverb/clean/clean_fileid_XXX.wav

  - SNR   : noisy 파일명에서 snrN 패턴으로 추출 (e.g. snr0 → 0)
  - fileid: noisy 파일명 끝의 fileid_XXX → clean_fileid_XXX.wav 와 매칭

저장 구조 (--output_dir 아래):
  {output_dir}/{reverb}/{stem}_enh.wav
  {output_dir}/{reverb}/{stem}_clean.wav
  {output_dir}/inf.scp
  {output_dir}/ref.scp
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from models.wavlm.feature_extractor import WavLM_feat as Encoder
from models.vocoder.wavlmdec_dual import WavLMDec as Model


# ─────────────────────────────────────────────────────────────────────────────
# DNS dataset scanning
# ─────────────────────────────────────────────────────────────────────────────

_SNR_RE    = re.compile(r'snr(-?\d+)', re.IGNORECASE)
_FILEID_RE = re.compile(r'fileid_(\d+)', re.IGNORECASE)


def scan_dns(dns_root: Path) -> list:
    """
    dns_root 아래의 no_reverb / with_reverb 두 폴더를 스캔해
    (noisy_path, clean_path, snr, reverb, stem) 목록을 반환합니다.

    noisy 파일명 예: clnsp102_traffic_248091_3_snr0_tl-21_fileid_268.wav
    clean 파일명 예: clean_fileid_268.wav
    """
    entries = []
    for reverb_dir in ("no_reverb", "with_reverb"):
        noisy_dir = dns_root / reverb_dir / "noisy"
        clean_dir = dns_root / reverb_dir / "clean"

        if not noisy_dir.exists():
            print(f"[warn] Not found, skipping: {noisy_dir}")
            continue

        # build fileid → clean_path index
        clean_index = {}
        for p in clean_dir.glob("*.wav"):
            m = _FILEID_RE.search(p.stem)
            if m:
                clean_index[m.group(1)] = p

        for noisy_path in sorted(noisy_dir.glob("*.wav")):
            m_snr    = _SNR_RE.search(noisy_path.stem)
            m_fileid = _FILEID_RE.search(noisy_path.stem)

            if not m_fileid:
                print(f"[warn] Cannot parse fileid from: {noisy_path.name}")
                continue

            fileid     = m_fileid.group(1)
            clean_path = clean_index.get(fileid)
            if clean_path is None:
                print(f"[warn] No clean file for fileid={fileid}: {noisy_path.name}")
                continue

            snr = int(m_snr.group(1)) if m_snr else None

            entries.append({
                "stem":   noisy_path.stem,
                "noisy":  str(noisy_path),
                "clean":  str(clean_path),
                "snr":    snr,
                "reverb": reverb_dir,
            })

    print(f"[DNS] Found {len(entries)} utterances in {dns_root}")
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def infer(args):
    cfg_infer   = OmegaConf.load(args.config)
    cfg_network = OmegaConf.load(cfg_infer.network.config)

    save_folder = args.output_dir if args.output_dir else cfg_infer.network.enh_folder
    os.makedirs(save_folder, exist_ok=True)

    dns_root = Path(args.dns_root)
    entries  = scan_dns(dns_root)
    print(f"Processing {len(entries)} utterances  →  {save_folder}")

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    encoder = Encoder(**cfg_network['encoder_config']).to(device)
    model   = Model(**cfg_network['vocoder_config']).to(device).eval()

    model.load_state_dict(
        torch.load(cfg_infer['network']['checkpoint'], map_location=device)['generator']
    )

    inf_scp_list = []
    ref_scp_list = []

    for entry in tqdm(entries):
        noisy_path = entry["noisy"]
        clean_path = entry["clean"]
        stem       = entry["stem"]
        reverb     = entry["reverb"]

        true_wav, fs = sf.read(noisy_path, dtype='float32')

        inp = torch.FloatTensor(true_wav)[None, None].to(device)

        feat_a, feat_p = encoder(inp)
        output = model(feat_p, feat_a)

        esti_wav = output.cpu().detach().numpy().squeeze()
        esti_wav = esti_wav / np.max(np.abs(esti_wav)) * 0.9

        if esti_wav.shape[-1] < true_wav.shape[-1]:
            esti_wav = np.pad(esti_wav, (0, true_wav.shape[-1] - esti_wav.shape[-1]))
        else:
            esti_wav = esti_wav[..., :true_wav.shape[-1]]

        # Save under {save_folder}/{reverb}/
        out_subdir = os.path.join(save_folder, reverb)
        os.makedirs(out_subdir, exist_ok=True)

        enh_path        = os.path.join(out_subdir, f"{stem}_enh.wav")
        clean_save_path = os.path.join(out_subdir, f"{stem}_clean.wav")

        sf.write(enh_path, esti_wav, fs)

        clean_wav, clean_fs = sf.read(clean_path, dtype='float32')
        sf.write(clean_save_path, clean_wav, clean_fs)

        uid = f"{reverb}_{stem}"
        inf_scp_list.append((uid, enh_path))
        ref_scp_list.append((uid, clean_save_path))

    # Save scp files for evaluation
    with open(os.path.join(save_folder, "inf.scp"), "w") as f:
        for uid, audio_path in inf_scp_list:
            f.write(f"{uid} {audio_path}\n")

    with open(os.path.join(save_folder, "ref.scp"), "w") as f:
        for uid, audio_path in ref_scp_list:
            f.write(f"{uid} {audio_path}\n")

    print(f"Saved {len(entries)} × 2 files under: {save_folder}")
    print(f"SCP files: {save_folder}/inf.scp, ref.scp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DNS-Challenge-2020 synthetic test set inference (infer_vocoder_dual 기반)"
    )
    parser.add_argument('-C', '--config', default='configs/cfg_infer.yaml',
                        help='Inference config yaml')
    parser.add_argument('-D', '--device', default='0',
                        help='Index of the gpu device')
    parser.add_argument('--dns_root',
                        default='/workspace/DB/DNS-Challenge-2020/datasets/test_set/synthetic',
                        help='DNS synthetic test root (no_reverb/ and with_reverb/ 포함)')
    parser.add_argument('--output_dir', type=str, default='result_dns',
                        help='Output directory for enhanced files')

    args = parser.parse_args()
    infer(args)
