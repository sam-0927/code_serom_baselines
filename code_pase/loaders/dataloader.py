# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""
Dataset for speech enhancement (WavLM fine-tuning and dual-stream vocoder).
Reads from a txt filelist with format:
    clean | noise | noisy | text [| snr]
Loads pre-mixed noisy and clean audio directly — no on-the-fly noise simulation.
"""
import random
from torch.utils import data
import numpy as np
from utils import simulate_utils


class URGENT2Dataset(data.Dataset):
    def __init__(
        self,
        filelist: str,
        wav_len=4,
        num_per_epoch=10000,
        random_start=False,
        default_fs=16000,
        mode='train'
    ):
        super().__init__()
        assert mode in ['train', 'validation']
        self.wav_len = wav_len
        self.num_per_epoch = num_per_epoch
        self.random_start = random_start
        self.default_fs = default_fs
        self.mode = mode

        self.meta = []
        with open(filelist) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(' | ')
                clean_path = parts[0]
                noisy_path = parts[2]
                self.meta.append({"id": f"fileid_{i}", "clean": clean_path, "noisy": noisy_path})

        print(f"Number of {mode} samples: {len(self.meta)}")
        self.sample_data_per_epoch(mode)

    def sample_data_per_epoch(self, mode='train'):
        if mode == 'train':
            self.meta_selected = random.sample(self.meta, min(self.num_per_epoch, len(self.meta)))
        else:
            self.meta_selected = self.meta[:self.num_per_epoch]

    def __getitem__(self, idx):
        info = self.meta_selected[idx]
        uid = info["id"]
        fs = self.default_fs
        rng = np.random.default_rng(int(uid.split("_")[-1]))

        clean_sample = simulate_utils.read_audio(info["clean"], force_1ch=True, fs=fs)[0]
        noisy_sample = simulate_utils.read_audio(info["noisy"], force_1ch=True, fs=fs)[0]

        orig_len = clean_sample.shape[1]

        if self.wav_len != 0:
            seg_len = int(self.wav_len * fs)
            # align noisy to same length as clean before slicing
            noisy_len = noisy_sample.shape[1]
            if noisy_len < orig_len:
                noisy_sample = np.pad(noisy_sample, ((0, 0), (0, orig_len - noisy_len)), constant_values=0)
            elif noisy_len > orig_len:
                noisy_sample = noisy_sample[:, :orig_len]

            if seg_len < orig_len:
                start_point = rng.integers(0, orig_len - seg_len) if self.random_start else 0
                clean_sample = clean_sample[:, start_point:start_point + seg_len]
                noisy_sample = noisy_sample[:, start_point:start_point + seg_len]
            elif seg_len > orig_len:
                pad_points = seg_len - orig_len
                clean_sample = np.pad(clean_sample, ((0, 0), (0, pad_points)), constant_values=0)
                noisy_sample = np.pad(noisy_sample, ((0, 0), (0, pad_points)), constant_values=0)

        scale = 0.9 / (max(
            np.max(np.abs(noisy_sample)),
            np.max(np.abs(clean_sample)),
        ) + 1e-12)
        noisy_sample = noisy_sample * scale
        clean_sample = clean_sample * scale

        info = {'id': uid, 'fs': fs, 'length': orig_len}
        return noisy_sample.astype(np.float32).squeeze(), clean_sample.astype(np.float32).squeeze(), info

    def __len__(self):
        return len(self.meta_selected)
