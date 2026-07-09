import argparse
import logging
import os
import random
import numpy as np
import itertools
import torch
from pathlib import Path
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from cru_loss_mres2 import cruse_loss
from discriminator import *
import soundfile as sf
from torchaudio.transforms import InverseSpectrogram
from model import Cruse
from torch.utils.tensorboard import SummaryWriter

# Path Configurations
dir_input = '/workspace/DB/librispeech_se_snr-515/metadata.txt'
dir_input_val = '/workspace/DB/librispeech_se_snr-515_eval/dev-clean/metadata.txt'

# Helper functions for signal processing
device = torch.device("cuda:0")
win = lambda x: torch.sqrt(torch.hann_window(x)).to(device)
from_spec = InverseSpectrogram(n_fft=512, hop_length=256, window_fn=win)

TARGET_SAMPLES = 5 * 16000  # 5 seconds at 16 kHz
_STFT_WINDOW = torch.sqrt(torch.hann_window(512))  # shared, CPU

def _stft(waveform: torch.Tensor) -> torch.Tensor:
    # Returns [channels, time, freq] to match the original .pt layout:
    # the training loop does permute(0,1,3,2) -> [ch, freq, time] before InverseSpectrogram
    out = []
    for ch in waveform:
        spec = torch.stft(ch, n_fft=512, hop_length=256,
                          window=_STFT_WINDOW, return_complex=True)  # [freq, time]
        out.append(spec.T)  # [time, freq]
    return torch.stack(out)  # [channels, time, freq]

class MyDataset(Dataset):
    def __init__(self, filelist_path, data_num=None, random_crop=True):
        all_files = []
        skipped = 0
        with open(filelist_path, "r") as f:
            for line in f:
                parts = line.strip().split(" | ")
                clean_path, noisy_path = parts[0].strip(), parts[2].strip()
                info = sf.info(noisy_path)
                if info.frames < TARGET_SAMPLES:
                    skipped += 1
                    continue
                all_files.append((clean_path, noisy_path))
        if skipped > 0:
            logging.info(f'Skipped {skipped} files shorter than 5 seconds.')
        if data_num is not None:
            self.pt_files = random.sample(all_files, min(data_num, len(all_files)))
        else:
            self.pt_files = all_files
        self.random_crop = random_crop

    def __getitem__(self, idx):
        clean_path, noisy_path = self.pt_files[idx]
        gt_np, _ = sf.read(clean_path, always_2d=True)   # [samples, channels]
        noisy_np, _ = sf.read(noisy_path, always_2d=True)
        if self.random_crop:
            start = random.randint(0, noisy_np.shape[0] - TARGET_SAMPLES)
        else:
            start = 0
        end = start + TARGET_SAMPLES
        gt = torch.from_numpy(gt_np.T).float()[:, start:end]    # [channels, samples]
        noisy = torch.from_numpy(noisy_np.T).float()[:, start:end]
        return {'input': _stft(noisy), 'gt': _stft(gt)}

    def __len__(self):
        return len(self.pt_files)

def data_generator(filelist_path, data_num=None, random_crop=True):
    dataset = MyDataset(filelist_path, data_num=data_num, random_crop=random_crop)
    datasize = len(dataset)
    return dataset, datasize

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 8,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        resume_epoch: int = 0,
):
    # Initialize Multi-Scale and Multi-Period Discriminators
    msd = MultiScaleDiscriminator().to(device)
    mpd = MultiPeriodDiscriminator().to(device)

    # Load discriminator checkpoints before creating optimizer (so param refs are correct)
    if resume_epoch > 0:
        msd_ckp = str(dir_checkpoint_msd + '/checkpoint_epoch{}.pth'.format(resume_epoch))
        mpd_ckp = str(dir_checkpoint_mpd + '/checkpoint_epoch{}.pth'.format(resume_epoch))
        if os.path.exists(msd_ckp):
            msd.load_state_dict(torch.load(msd_ckp, map_location=device))
            logging.info(f'MSD loaded from {msd_ckp}')
        else:
            logging.warning(f'MSD checkpoint not found: {msd_ckp}')
        if os.path.exists(mpd_ckp):
            mpd.load_state_dict(torch.load(mpd_ckp, map_location=device))
            logging.info(f'MPD loaded from {mpd_ckp}')
        else:
            logging.warning(f'MPD checkpoint not found: {mpd_ckp}')

    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()), lr=args.lrd,betas=(0.8, 0.99))

    msd.to(device);
    msd.train();
    mpd.to(device);
    mpd.train();

    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    dataset_val, datasize_val = data_generator(dir_input_val, random_crop=False)
    n_val = datasize_val
    val_loader = DataLoader(dataset_val, shuffle=False, drop_last=True, **loader_args)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    optim_g = optim.AdamW(model.parameters(), lr=args.lr,betas=(0.9, 0.99), eps=1e-08)

    writer = SummaryWriter(log_dir=os.path.join(dir_checkpoint, 'tensorboard'))
    global_step = 0
    train_loss_list = []
    val_loss_list = []
    x = []
    best_val_loss = float('inf')

    if resume_epoch > 0:
        optim_ckp = str(dir_checkpoint + '/optim_epoch{}.pth'.format(resume_epoch))
        if os.path.exists(optim_ckp):
            optim_state = torch.load(optim_ckp, map_location=device)
            optim_g.load_state_dict(optim_state['optim_g'])
            optim_d.load_state_dict(optim_state['optim_d'])
            global_step = optim_state.get('global_step', 0)
            train_loss_list = optim_state.get('train_loss_list', [])
            val_loss_list = optim_state.get('val_loss_list', [])
            x = optim_state.get('x', [])
            logging.info(f'Optimizer state loaded from {optim_ckp}')
        else:
            logging.warning(f'Optimizer checkpoint not found: {optim_ckp} — optimizer state not restored')
        logging.info(f'Resuming training from epoch {resume_epoch + 1}')

    for epoch in range(resume_epoch + 1, epochs + 1):
        model.train()
        dataset_train, n_train = data_generator(dir_input, data_num=9000)
        train_loader = DataLoader(dataset_train, shuffle=True, drop_last=True, **loader_args)
        num_train_batches = len(train_loader)
        epoch_loss = 0
        epoch_lossd = 0
        epoch_lossc = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='aud') as pbar:
            for batch in train_loader:
                inputs, gts = batch['input'], batch['gt']
                inputs = inputs.to(device=device, dtype=torch.cfloat)
                gts = gts.to(device=device, dtype=torch.cfloat)

                # Generator Forward Pass (Magnitude processing)
                inputs_mag = torch.pow(torch.abs(inputs), 0.3)
                mask_preds = model(inputs_mag)
                outputs = inputs_mag * mask_preds
                outputs = torch.pow(outputs, (10/3))
                outputs = outputs * torch.exp(1j * torch.angle(inputs))
                
                # Transform back to time domain
                output = from_spec(outputs.permute(0, 1, 3, 2))
                s = from_spec(gts.permute(0, 1, 3, 2))

                # Discriminator Update
                optim_d.zero_grad()
                loss_disc_all = 0
                s_ds_hat_r, s_ds_hat_g, _, _ = msd(s, output.detach())
                loss_disc_s, _, _ = discriminator_loss(s_ds_hat_r, s_ds_hat_g)
                loss_disc_all += loss_disc_s *0.01

                s_dp_hat_r, s_dp_hat_g, _, _ = mpd(s, output.detach())
                loss_disc_p, _, _ = discriminator_loss(s_dp_hat_r, s_dp_hat_g)
                loss_disc_all += loss_disc_p *0.01

                loss_disc_all.backward()
                optim_d.step()
                
                del s_ds_hat_r, s_ds_hat_g, loss_disc_s
                del s_dp_hat_r, s_dp_hat_g, loss_disc_p

                # Generator Update
                optim_g.zero_grad()
                cru_loss = cruse_loss(outputs, gts)
                loss_gen_all = 0 + cru_loss

                _, s_ds_hat_g, fmap_s_r, fmap_s_g = msd(s, output)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_s, _ = generator_loss(s_ds_hat_g)
                loss_gen_all += (loss_gen_s + loss_fm_s) *0.01

                _, s_dp_hat_g, fmap_p_r, fmap_p_g = mpd(s, output)
                loss_fm_p = feature_loss(fmap_p_r, fmap_p_g)
                loss_gen_p, _ = generator_loss(s_dp_hat_g)
                loss_gen_all += (loss_gen_p + loss_fm_p) *0.01
                    
                loss_gen_all.backward()
                optim_g.step()

                # logging.info(f'''disc_loss: {loss_disc_all}''')
                # logging.info(f'''gen_loss: {loss_gen_all}''')

                writer.add_scalar('Train/disc', loss_disc_all.item(), global_step)
                writer.add_scalar('Train/gen', loss_gen_all.item(), global_step)
                writer.add_scalar('Train/cru', cru_loss.item(), global_step)

                epoch_lossd += loss_disc_all.item()
                epoch_lossc += cru_loss.item()
                pbar.update(inputs.shape[0])
                global_step += 1
                epoch_loss += loss_gen_all.item()

        # Validation Loop
        model.eval()
        num_val_batches = len(val_loader)
        val_loss = 0
        val_lossd = 0
        val_lossc = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, gts = batch['input'], batch['gt']
                inputs = inputs.to(device=device, dtype=torch.cfloat)
                gts = gts.to(device=device, dtype=torch.cfloat)
                inputs_mag = torch.pow(torch.abs(inputs), 0.3)
                mask_preds = model(inputs_mag)
                outputs = inputs_mag * mask_preds
                outputs = torch.pow(outputs, (10/3))
                outputs = outputs * torch.exp(1j * torch.angle(inputs))
                output = from_spec(outputs.permute(0, 1, 3, 2))
                s = from_spec(gts.permute(0, 1, 3, 2))

                loss_disc_all = 0
                s_ds_hat_r, s_ds_hat_g, _, _ = msd(s, output.detach())
                loss_disc_s, _, _ = discriminator_loss(s_ds_hat_r, s_ds_hat_g)
                loss_disc_all += loss_disc_s * 0.01
                s_dp_hat_r, s_dp_hat_g, _, _ = mpd(s, output.detach())
                loss_disc_p, _, _ = discriminator_loss(s_dp_hat_r, s_dp_hat_g)
                loss_disc_all += loss_disc_p * 0.01

                del s_ds_hat_r, s_ds_hat_g, loss_disc_s
                del s_dp_hat_r, s_dp_hat_g, loss_disc_p

                val_cru_loss = cruse_loss(outputs, gts)
                loss_gen_all = 0 + val_cru_loss

                _, s_ds_hat_g, fmap_s_r, fmap_s_g = msd(s, output)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_s, _ = generator_loss(s_ds_hat_g)
                loss_gen_all += (loss_gen_s + loss_fm_s) * 0.01

                _, s_dp_hat_g, fmap_p_r, fmap_p_g = mpd(s, output)
                loss_fm_p = feature_loss(fmap_p_r, fmap_p_g)
                loss_gen_p, _ = generator_loss(s_dp_hat_g)
                loss_gen_all += (loss_gen_p + loss_fm_p) * 0.01

                val_lossd += loss_disc_all.item()
                val_lossc += val_cru_loss.item()
                val_loss += loss_gen_all.item()
        model.train()

        # Logging metrics
        logging.info(f'''train_loss: {epoch_loss/num_train_batches}''')
        logging.info(f'''train_lossd: {epoch_lossd/num_train_batches}''')
        logging.info(f'''train_lossc: {epoch_lossc/num_train_batches}''')
        logging.info(f'''val_loss: {val_loss/num_val_batches}''')
        logging.info(f'''val_lossd: {val_lossd/num_val_batches}''')
        logging.info(f'''val_lossc: {val_lossc/num_val_batches}''')
        writer.add_scalar('Train/gen', epoch_loss/num_train_batches, epoch)
        writer.add_scalar('Train/disc', epoch_lossd/num_train_batches, epoch)
        writer.add_scalar('Train/cruse', epoch_lossc/num_train_batches, epoch)
        writer.add_scalar('Val/gen', val_loss/num_val_batches, epoch)
        writer.add_scalar('Val/disc', val_lossd/num_val_batches, epoch)
        writer.add_scalar('Val/cruse', val_lossc/num_val_batches, epoch)
        train_loss_list.append(epoch_loss/num_train_batches)
        val_loss_list.append(val_loss/num_val_batches)
        x.append(epoch)

        # Saving model and discriminator checkpoints
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            Path(dir_checkpoint_msd).mkdir(parents=True, exist_ok=True)
            Path(dir_checkpoint_mpd).mkdir(parents=True, exist_ok=True)
            if epoch % 1 == 0:
                state_dict = model.state_dict()
                torch.save(state_dict, str(dir_checkpoint + '/checkpoint_epoch{}.pth'.format(epoch)))
                state_dict_msd = msd.state_dict()
                torch.save(state_dict_msd, str(dir_checkpoint_msd + '/checkpoint_epoch{}.pth'.format(epoch)))
                state_dict_mpd = mpd.state_dict()
                torch.save(state_dict_mpd, str(dir_checkpoint_mpd + '/checkpoint_epoch{}.pth'.format(epoch)))
                torch.save({
                    'optim_g': optim_g.state_dict(),
                    'optim_d': optim_d.state_dict(),
                    'global_step': global_step,
                    'train_loss_list': train_loss_list,
                    'val_loss_list': val_loss_list,
                    'x': x,
                }, str(dir_checkpoint + '/optim_epoch{}.pth'.format(epoch)))
                logging.info(f'Checkpoint {epoch} saved!')

            avg_val_lossc = val_lossc / num_val_batches
            if avg_val_lossc < best_val_loss:
                best_val_loss = avg_val_lossc
                torch.save(model.state_dict(), str(dir_checkpoint + '/checkpoint_best.pth'))
                torch.save(msd.state_dict(), str(dir_checkpoint_msd + '/checkpoint_best.pth'))
                torch.save(mpd.state_dict(), str(dir_checkpoint_mpd + '/checkpoint_best.pth'))
                logging.info(f'Best checkpoint saved at epoch {epoch} (val_lossc={best_val_loss:.6f})')

    writer.close()

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on encoded and enhanced audios')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=2000, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-lr', metavar='LR', type=float, default=5e-4, help='Learning rate', dest='lr')
    parser.add_argument('--learning-rate-disc', '-lrd', metavar='LRD', type=float, default=1e-7, help='Learning rate discriminator', dest='lrd')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0, help='Percent of data for validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--resume-epoch', '-r', type=int, default=0, dest='resume_epoch',
                        help='Resume training from this epoch (loads model/discriminator/optimizer checkpoints)')
    parser.add_argument('--outdir', '-o', type=str, default='./output', dest='outdir',
                        help='Directory to save checkpoints (model, msd, mpd, optimizer, tensorboard logs)')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    dir_checkpoint = args.outdir
    dir_checkpoint_msd = os.path.join(dir_checkpoint, 'msd')
    dir_checkpoint_mpd = os.path.join(dir_checkpoint, 'mpd')

    model = Cruse()
    logging.info(f'Network:\n')

    if args.resume_epoch > 0:
        model_ckp = str(dir_checkpoint + '/checkpoint_epoch{}.pth'.format(args.resume_epoch))
        state_dict = torch.load(model_ckp, map_location=device)
        state_dict.pop('mask_values', None)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {model_ckp}')
    elif args.load:
        state_dict = torch.load(args.load, map_location=device)
        state_dict.pop('mask_values', None)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            val_percent=args.val / 100,
            amp=args.amp,
            resume_epoch=args.resume_epoch,
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! Cleaning cache...')
        torch.cuda.empty_cache()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            val_percent=args.val / 100,
            amp=args.amp,
            resume_epoch=args.resume_epoch,
        )
