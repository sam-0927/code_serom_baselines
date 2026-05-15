import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from os import listdir
from scipy.io import wavfile
from torchaudio.transforms import Spectrogram
import matplotlib.pyplot as plt

# Custom modules
from cru_loss_mres2 import cruse_loss
from kld_cpx import kl_divergence_complexn
from small_vae import small_VAE
from large_vae_cpx import VAE
from large_cpx import Cruse

# Config paths
dir_input = '/project_ghent/ds/datasets/noisy_train_5s/'
dir_input_val = '/project_ghent/ds/datasets/noisy_test_5s/'
dir_checkpoint = './ckp/ckp_vae_cpx_stu_t5161_64'
dir_lossplots = './loss/cpx_vae_cpx_stu_t5161_64'

class MyDataset(Dataset):
    def __init__(self, dir_input, Train=True, data_num=9000):
        file_list = [file for file in listdir(dir_input)]
        sub_list = random.sample(file_list, data_num) if Train else file_list
        
        self.noisy_files = [os.path.join(dir_input, f, 'noisy.wav') for f in sub_list]
        self.gt_files = [os.path.join(dir_input, f, 'clean.wav') for f in sub_list]
        # Pre-define window function for efficiency
        self.win_fn = lambda x: torch.sqrt(torch.hann_window(x))

    def __getitem__(self, idx):
        # Load and normalize audio
        sr, audio_gt = wavfile.read(self.gt_files[idx])
        audio_gt = torch.tensor(audio_gt.astype(np.float32) / (2 ** 15))
        
        sr, audio_noisy = wavfile.read(self.noisy_files[idx])
        audio_noisy = torch.tensor(audio_noisy.astype(np.float32) / (2 ** 15))

        # STFT transformation
        spec_cfg = dict(n_fft=512, hop_length=256, power=None, window_fn=self.win_fn)
        gt = Spectrogram(**spec_cfg)(audio_gt).cfloat().permute(1, 0).unsqueeze(0)
        noisy = Spectrogram(**spec_cfg)(audio_noisy).cfloat().permute(1, 0).unsqueeze(0)

        return {'input': noisy, 'gt': gt}

    def __len__(self):
        return len(self.gt_files)

def train_model(model, device, epochs=5, batch_size=8, learning_rate=1e-5, save_checkpoint=True, amp=False):
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    
    # Validation loader
    dataset_val = MyDataset(dir_input_val, Train=False)
    val_loader = DataLoader(dataset_val, shuffle=True, drop_last=True, **loader_args)
    n_val = len(dataset_val)

    logging.info(f'Starting training on {device.type}')
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-08)

    train_loss_list, val_loss_list, x_axis = [], [], []

    for epoch in range(1, epochs + 1):
        model.train()
        dataset_train = MyDataset(dir_input, Train=True, data_num=9000)
        train_loader = DataLoader(dataset_train, shuffle=True, drop_last=True, **loader_args)
        
        epoch_loss = 0
        with tqdm(total=len(dataset_train), desc=f'Epoch {epoch}/{epochs}', unit='aud') as pbar:
            for batch in train_loader:
                optimizer.zero_grad()
                inputs = batch['input'].to(device=device, dtype=torch.cfloat)
                gts = batch['gt'].to(device=device, dtype=torch.cfloat)

                # Feature compression and RI concatenation
                inputs_mri_cpx = torch.pow(torch.abs(inputs), 0.3) * torch.exp(1j * torch.angle(inputs))
                inputs_ri = torch.cat((torch.real(inputs_mri_cpx), torch.imag(inputs_mri_cpx)), dim=1)

                with torch.no_grad():
                    # Teacher/Pred model forward
                    outputs_p = model_pred(inputs_ri)
                    outputs_p = torch.complex(outputs_p[:,0,:,:], outputs_p[:,1,:,:]).unsqueeze(1) * inputs_mri_cpx
                    outputs_p = torch.cat((torch.real(outputs_p), torch.imag(outputs_p)), dim=1)
                    
                    vae_in = torch.cat((outputs_p, inputs_ri), dim=1)
                    _, gre_mu, gim_mu, gr, gre_s, gim_s = model_vae(vae_in)

                # Student model forward
                outputs, re_mu, im_mu, r, re_s, im_s = model(inputs_ri)
                
                # Complex reconstruction and power law decompression
                outputs = torch.complex(outputs[:,0,:,:].unsqueeze(1), outputs[:,1,:,:].unsqueeze(1)) * inputs_mri_cpx
                outputs = torch.pow(torch.abs(outputs), (10/3)) * torch.exp(1j * torch.angle(outputs))

                # Loss calculation
                c_loss = cruse_loss(outputs, gts)
                rep_loss = torch.mean(kl_divergence_complexn(re_mu, im_mu, r, re_s, im_s, gre_mu, gim_mu, gr, gre_s, gim_s))
                train_loss = c_loss + rep_loss
                
                train_loss.backward()
                optimizer.step()
                
                pbar.update(inputs.shape[0])
                epoch_loss += train_loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        for batch in val_loader:
            with torch.no_grad():
                inputs, gts = batch['input'].to(device), batch['gt'].to(device)
                inputs_mri_cpx = torch.pow(torch.abs(inputs), 0.3) * torch.exp(1j * torch.angle(inputs))
                inputs_ri = torch.cat((torch.real(inputs_mri_cpx), torch.imag(inputs_mri_cpx)), dim=1)

                # Get teacher targets for validation
                outputs_p = model_pred(inputs_ri)
                outputs_p = torch.complex(outputs_p[:,0,:,:], outputs_p[:,1,:,:]).unsqueeze(1) * inputs_mri_cpx
                outputs_p = torch.cat((torch.real(outputs_p), torch.imag(outputs_p)), dim=1)
                _, gre_mu, gim_mu, gr, gre_s, gim_s = model_vae(torch.cat((outputs_p, inputs_ri), dim=1))

                # Student prediction
                outputs, re_mu, im_mu, r, re_s, im_s = model(inputs_ri)
                outputs = torch.complex(outputs[:,0,:,:].unsqueeze(1), outputs[:,1,:,:].unsqueeze(1)) * inputs_mri_cpx
                outputs = torch.pow(torch.abs(outputs), (10/3)) * torch.exp(1j * torch.angle(outputs))

                val_loss += (cruse_loss(outputs, gts) + torch.mean(kl_divergence_complexn(re_mu, im_mu, r, re_s, im_s, gre_mu, gim_mu, gr, gre_s, gim_s))).item()

        # Logging and Plotting
        avg_train, avg_val = epoch_loss/len(dataset_train), val_loss/n_val
        logging.info(f'Epoch {epoch}: Train Loss: {avg_train:.6f}, Val Loss: {avg_val:.6f}')
        
        train_loss_list.append(avg_train)
        val_loss_list.append(avg_val)
        x_axis.append(epoch)

        if epoch % 5 == 0:
            Path(dir_lossplots).mkdir(parents=True, exist_ok=True)
            plt.figure()
            plt.plot(x_axis, train_loss_list, 'r', label='train_loss')
            plt.plot(x_axis, val_loss_list, 'b', label='val_loss')
            plt.legend()
            plt.savefig(os.path.join(dir_lossplots, f'loss_epoch{epoch}.png'))
            plt.close()

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(dir_checkpoint, f'checkpoint_epoch{epoch}.pth'))

def get_args():
    parser = argparse.ArgumentParser(description='Train VAE for Speech Enhancement')
    parser.add_argument('--epochs', '-e', type=int, default=2000)
    parser.add_argument('--batch-size', '-b', type=int, default=8)
    parser.add_argument('--learning-rate', '-lr', type=float, default=5e-4, dest='lr')
    parser.add_argument('--amp', action='store_true', default=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model initialization
    model_vae = VAE().to(device)
    model_pred = Cruse().to(device)
    model_small_vae = small_VAE().to(device)

    # Load pre-trained weights
    model_pred.load_state_dict(torch.load('.ckps/checkpoint_epoch1515.pth', map_location=device))
    model_vae.load_state_dict(torch.load('.ckps/checkpoint_epoch341.pth', map_location=device))

    try:
        train_model(model=model_small_vae, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, device=device, amp=args.amp)
    except torch.cuda.OutOfMemoryError:
        logging.error('OOM detected, clearing cache...')
        torch.cuda.empty_cache()