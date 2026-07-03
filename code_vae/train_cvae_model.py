import argparse
import logging
import os
import random
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from pathlib import Path
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchaudio.transforms import Spectrogram
from cru_loss_mres2 import cruse_loss

from large_cpx import Cruse
from large_vae_cpx import VAE
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from kld_cpx import kld_cpx_tosn


# Directories
dir_input = '/workspace/DB/librispeech_se_snr-515/metadata.txt'
dir_input_val = '/workspace/DB/librispeech_se_snr-515_eval/dev-clean/metadata.txt'
dir_checkpoint = './output_cvae_model'
dir_lossplots = os.path.join(dir_checkpoint,'loss_plot')
os.makedirs(dir_checkpoint, exist_ok=True)
os.makedirs(dir_lossplots, exist_ok=True)

N_FFT = 512
HOP_LENGTH = 256
TARGET_FRAMES = 313  # 5초 * 16kHz / hop_length(256) + 1
CROP_SAMPLES = (TARGET_FRAMES - 1) * HOP_LENGTH  # 80000 samples = 5s @ 16kHz

_win = lambda x: torch.sqrt(torch.hann_window(x))
_to_spec = Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH, power=None, window_fn=_win)

def _crop_and_spec(path, start):
    wav, _ = sf.read(path, start=start, stop=start + CROP_SAMPLES, dtype='float32')
    t = torch.from_numpy(wav).unsqueeze(0)   # [1, T_audio]
    return _to_spec(t).permute(0, 2, 1)      # [1, T_frames, F]

def load_valid_files(filelist_path):
    """filelist에서 5초 이상인 wav 파일만 걸러 반환. 학습 시작 시 1회만 호출."""
    all_files = []
    with open(filelist_path, "r") as f:
        for line in f:
            parts = line.strip().split(" | ")
            all_files.append((parts[0].strip(), parts[2].strip()))

    valid = []
    skipped_corrupt = 0
    for clean_path, noisy_path in all_files:
        try:
            info = sf.info(noisy_path)
            if info.frames < CROP_SAMPLES:
                continue
            # sf.info()는 헤더만 검증하므로 실제 데이터도 짧게 읽어 손상 여부 확인
            sf.read(noisy_path, start=0, stop=min(1024, info.frames), dtype='float32')
            sf.read(clean_path, start=0, stop=min(1024, sf.info(clean_path).frames), dtype='float32')
        except Exception:
            skipped_corrupt += 1
            continue
        valid.append((clean_path, noisy_path))
    if skipped_corrupt > 0:
        logging.warning(f'Skipped {skipped_corrupt} corrupt files in {filelist_path}')
    logging.info(f'{filelist_path}: {len(all_files)} total, {len(valid)} kept (>= 5s)')
    return valid

class MyDataset(Dataset):
    def __init__(self, pt_files, Train=True, data_num=9000):
        n = min(data_num, len(pt_files))
        self.pt_files = random.sample(pt_files, n) if Train else pt_files
        self.train = Train

    def __getitem__(self, idx):
        clean_path, noisy_path = self.pt_files[idx]
        try:
            info = sf.info(noisy_path)
            max_start = info.frames - CROP_SAMPLES
            start = random.randint(0, max_start) if self.train else 0
            gt = _crop_and_spec(clean_path, start)
            noisy = _crop_and_spec(noisy_path, start)
            return {'input': noisy, 'gt': gt}
        except Exception:
            # 파일 중간 구간 손상 시 다른 샘플로 대체
            logging.warning(f'Skipping corrupt audio at idx={idx}: {noisy_path}')
            return self.__getitem__(random.randint(0, len(self.pt_files) - 1))

    def __len__(self):
        return len(self.pt_files)

def train_model(model, model_pred, device, epochs=5, batch_size=8, learning_rate=1e-5, save_checkpoint=True, start_epoch=1):
    # 유효 파일 목록 1회 로드
    valid_train_files = load_valid_files(dir_input)
    valid_val_files = load_valid_files(dir_input_val)

    dataset_val = MyDataset(valid_val_files, Train=False)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-08)

    if args.load and os.path.isfile(args.load):
        checkpoint = torch.load(args.load, map_location=device)
        if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logging.info(f'Optimizer state restored from {args.load}')

    writer = SummaryWriter(log_dir=os.path.join(dir_checkpoint, 'tensorboard'))
    train_loss_list, val_loss_list, x_axis = [], [], []
    best_val_loss = float('inf')

    for epoch in range(start_epoch, epochs + 1):
        dataset_train = MyDataset(valid_train_files, Train=True, data_num=9000)
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

        model.train()
        epoch_loss = 0
        epoch_cruse_loss = 0
        epoch_kld_loss = 0
        iteration = 1
        with tqdm(total=len(dataset_train), desc=f'Epoch {epoch}/{epochs}', unit='aud') as pbar:
            for batch in train_loader:
                optimizer.zero_grad()

                inputs = batch['input'].to(device, dtype=torch.cfloat)
                gts = batch['gt'].to(device, dtype=torch.cfloat)

                # Forward pass with power compression and RI concatenation
                inputs_mri_cpx = torch.pow(torch.abs(inputs), 0.3) * torch.exp(1j * torch.angle(inputs))
                model_in = torch.cat((torch.real(inputs_mri_cpx), torch.imag(inputs_mri_cpx)), dim=1)
                with torch.no_grad():
                    # Teacher/Pred model forward
                    outputs_p = model_pred(model_in)
                    outputs_p = torch.complex(outputs_p[:,0,:,:], outputs_p[:,1,:,:]).unsqueeze(1) * inputs_mri_cpx
                outputs_p = torch.cat((torch.real(outputs_p), torch.imag(outputs_p)), dim=1)
                vae_in = torch.cat((outputs_p, model_in), dim=1)
                outputs, re_mu, im_mu, r, re_s, im_s = model(vae_in)
                outputs = torch.complex(outputs[:,0,:,:].unsqueeze(1), outputs[:,1,:,:].unsqueeze(1)) * inputs_mri_cpx
                outputs = torch.pow(torch.abs(outputs), (10/3)) * torch.exp(1j * torch.angle(outputs))

                cruse = cruse_loss(outputs, gts)
                kl = torch.mean(kld_cpx_tosn(re_mu, im_mu, r, re_s, im_s))
                loss = cruse + kl
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_cruse_loss += cruse.item()
                epoch_kld_loss += kl.item()

                # logging.info(f'Epoch {epoch}: Train Loss: {loss:.6f}')
                writer.add_scalar('Train/loss', loss.item(), iteration)
                pbar.update(inputs.shape[0])
                iteration += 1

        # Validation phase
        model.eval()
        val_loss = 0
        val_kld_loss = 0
        with torch.no_grad():
            with tqdm(total=len(dataset_val), desc=f'Epoch {epoch}/{epochs}', unit='aud') as pbar:
                for batch in val_loader:
                    inputs = batch['input'].to(device, dtype=torch.cfloat)
                    gts = batch['gt'].to(device, dtype=torch.cfloat)

                    inputs_mri_cpx = torch.pow(torch.abs(inputs), 0.3) * torch.exp(1j * torch.angle(inputs))
                    model_in = torch.cat((torch.real(inputs_mri_cpx), torch.imag(inputs_mri_cpx)), dim=1)
                    outputs_p = model_pred(model_in)
                    outputs_p = torch.complex(outputs_p[:,0,:,:], outputs_p[:,1,:,:]).unsqueeze(1) * inputs_mri_cpx
                    outputs_p = torch.cat((torch.real(outputs_p), torch.imag(outputs_p)), dim=1)
                    vae_in = torch.cat((outputs_p, model_in), dim=1)
                    outputs, re_mu, im_mu, r, re_s, im_s = model(vae_in)
                    outputs = torch.complex(outputs[:,0,:,:].unsqueeze(1), outputs[:,1,:,:].unsqueeze(1)) * inputs_mri_cpx
                    outputs = torch.pow(torch.abs(outputs), (10/3)) * torch.exp(1j * torch.angle(outputs))

                    val_loss += cruse_loss(outputs, gts).item()
                    val_kld_loss += torch.mean(kld_cpx_tosn(re_mu, im_mu, r, re_s, im_s)).item()
                    pbar.update(inputs.shape[0])

        model.train()
        # Logging and Plotting
        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_cruse = epoch_cruse_loss / len(train_loader)
        avg_train_kld = epoch_kld_loss / len(train_loader)
        avg_val_cruse = val_loss / len(val_loader)
        avg_val_kld = val_kld_loss / len(val_loader)
        avg_val_loss = avg_val_cruse + avg_val_kld
        logging.info(
            f'Epoch {epoch}: '
            f'Train Loss: {avg_train_loss:.6f} (CRUSE: {avg_train_cruse:.6f}, KLD: {avg_train_kld:.6f}) | '
            f'Val Loss: {avg_val_loss:.6f} (CRUSE: {avg_val_cruse:.6f}, KLD: {avg_val_kld:.6f})'
        )
        writer.add_scalar('Train/Loss', avg_train_loss, epoch)
        writer.add_scalar('Train/CRUSE', avg_train_cruse, epoch)
        writer.add_scalar('Train/KLD', avg_train_kld, epoch)
        writer.add_scalar('Val/Loss', avg_val_loss, epoch)
        writer.add_scalar('Val/CRUSE', avg_val_cruse, epoch)
        writer.add_scalar('Val/KLD', avg_val_kld, epoch)

        train_loss_list.append(avg_train_loss)
        val_loss_list.append(avg_val_loss)

        x_axis.append(epoch)

        if epoch % 5 == 0:
            plt.figure()
            plt.plot(x_axis, train_loss_list, 'r', label='train_loss')
            plt.plot(x_axis, val_loss_list, 'b', label='val_loss')
            plt.legend()
            plt.savefig(os.path.join(dir_lossplots, f'loss_epoch{epoch}.png'))
            plt.close()

        if save_checkpoint:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(dir_checkpoint, f'checkpoint_epoch{epoch}.pth'))

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                }, os.path.join(dir_checkpoint, 'checkpoint_best.pth'))
                logging.info(f'Best checkpoint saved at epoch {epoch} (val_loss={best_val_loss:.6f})')

    writer.close()

def get_args():
    parser = argparse.ArgumentParser(description='Train Teacher Predictive Model')
    parser.add_argument('--epochs', '-e', type=int, default=2000)
    parser.add_argument('--batch-size', '-b', type=int, default=8)
    parser.add_argument('--learning-rate', '-lr', type=float, default=2e-4, dest='lr')
    parser.add_argument('--load', '-f', type=str, default=None)
    parser.add_argument('--start-epoch', type=int, default=None, help='이어서 학습할 시작 epoch (미지정 시 체크포인트에서 자동 감지)')
    parser.add_argument('--amp', action='store_true', default=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VAE().to(device)
    model_pred = Cruse().to(device)
    model_pred.eval()
    # Load pre-trained weights
    model_pred_checkpoint = torch.load('output_predictive_model/checkpoint_epoch200.pth', map_location=device)
    model_pred.load_state_dict(model_pred_checkpoint['model_state_dict'])

    start_epoch = 1
    if args.load:
        checkpoint = torch.load(args.load, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            saved_epoch = checkpoint.get('epoch', 0)
            start_epoch = (args.start_epoch if args.start_epoch is not None else saved_epoch + 1)
        else:
            # 구형 체크포인트 (state_dict만 저장된 경우) 호환
            model.load_state_dict(checkpoint, strict=False)
            if args.start_epoch is not None:
                start_epoch = args.start_epoch
        logging.info(f'Model loaded from {args.load}, resuming from epoch {start_epoch}')

    model.to(device)
    try:
        train_model(model=model, model_pred=model_pred, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, device=device, start_epoch=start_epoch)
    except torch.cuda.OutOfMemoryError:
        logging.error('OOM detected. Cleaning cache.')
        torch.cuda.empty_cache()
