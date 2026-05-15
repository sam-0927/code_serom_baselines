import os
import argparse
import numpy as np
import torch
from scipy.io import wavfile
from torchaudio.transforms import Spectrogram
from tqdm import tqdm


def preprocess(filelist_path, output_dir):
    win = lambda x: torch.sqrt(torch.hann_window(x))
    spec_transform = Spectrogram(n_fft=512, hop_length=256, power=None, window_fn=win)

    clean_dir = os.path.join(output_dir, 'clean')
    noisy_dir = os.path.join(output_dir, 'noisy')
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)

    with open(filelist_path, "r") as f:
        lines = f.readlines()

    out_paths = []
    for i, line in enumerate(tqdm(lines, desc=f"Preprocessing {filelist_path}")):
        parts = line.strip().split(" | ")
        clean_path = parts[0].strip()
        noisy_path = parts[2].strip()

        sr, audio_gt = wavfile.read(clean_path)
        audio_gt = torch.tensor(audio_gt.astype(np.float32) / (2 ** 15))

        sr, audio_noisy = wavfile.read(noisy_path)
        audio_noisy = torch.tensor(audio_noisy.astype(np.float32) / (2 ** 15))

        gt = spec_transform(audio_gt).cfloat().permute(1, 0).unsqueeze(0)
        noisy = spec_transform(audio_noisy).cfloat().permute(1, 0).unsqueeze(0)

        pt_name = os.path.splitext(os.path.basename(noisy_path))[0] + '.pt'
        clean_pt_path = os.path.join(clean_dir, pt_name)
        noisy_pt_path = os.path.join(noisy_dir, pt_name)

        torch.save(gt, clean_pt_path)
        torch.save(noisy, noisy_pt_path)
        out_paths.append((clean_pt_path, noisy_pt_path))

    new_filelist_path = filelist_path.replace('.txt', '_pt.txt')
    with open(new_filelist_path, 'w') as f:
        for clean_pt, noisy_pt in out_paths:
            f.write(f"{clean_pt} | {noisy_pt}\n")

    print(f"Saved {len(out_paths)} files to {output_dir}")
    print(f"New filelist: {new_filelist_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', type=str, required=True, help='원본 filelist txt 경로')
    parser.add_argument('--output_dir', type=str, required=True, help='.pt 파일 저장 디렉토리')
    args = parser.parse_args()

    preprocess(args.filelist, args.output_dir)
