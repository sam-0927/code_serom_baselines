# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""
Dataset for (single-stream) vocoder training and validation.
Reads from a txt filelist with format:
    clean | noise | noisy | text [| snr]
Only the clean column is used.
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
                self.meta.append({"id": f"fileid_{i}", "clean": clean_path})

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

        speech_sample = simulate_utils.read_audio(info["clean"], force_1ch=True, fs=fs)[0]

        orig_len = speech_sample.shape[1]

        if self.wav_len != 0:
            seg_len = int(self.wav_len * fs)
            if seg_len < orig_len:
                start_point = rng.integers(0, orig_len - seg_len) if self.random_start else 0
                speech_sample = speech_sample[:, start_point:start_point + seg_len]
            elif seg_len > orig_len:
                pad_points = seg_len - orig_len
                speech_sample = np.pad(speech_sample, ((0, 0), (0, pad_points)), constant_values=0)

        scale = 0.9 / (np.max(np.abs(speech_sample)) + 1e-12)
        speech_sample = speech_sample * scale

        info = {'id': uid, 'fs': fs, 'length': orig_len}
        return speech_sample.astype(np.float32), info

    def __len__(self):
        return len(self.meta_selected)
