import torch
import torch.nn.functional as F
from torch import nn
import math

class GRUblockf(nn.Module):
    def __init__(self, feat_size=64):
        super().__init__()
        self.gru = nn.GRU(input_size=feat_size, hidden_size=feat_size, num_layers=1, bidirectional=True, batch_first=True)  
        self.feat_size = feat_size
        self.attn = nn.MultiheadAttention(feat_size, 4, batch_first=True)
        self.activationtrans = nn.LeakyReLU(negative_slope=0.03)
        self.layernorm1 = nn.LayerNorm(feat_size)
        self.layernorm2 = nn.LayerNorm(feat_size)
        self.lin = nn.Linear(feat_size * 2, feat_size)
        
    def forward(self, x):
        b, c, t, f = x.size()
        # Reshape for frequency-wise processing
        x = x.permute(0, 2, 3, 1).contiguous().view(b*t, f, c)

        gru_out, _ = self.gru(x)
        x = x + self.lin(self.activationtrans(gru_out))
        x = self.layernorm2(x)
        
        attn_out, _ = self.attn(x, x, x)
        x = self.layernorm1(x + attn_out)
        
        return x.view(b, t, f, c).permute(0, 3, 1, 2)

class GRUblockt(nn.Module):
    def __init__(self, feat_size=64):
        super().__init__()
        self.gru = nn.GRU(input_size=feat_size, hidden_size=feat_size, num_layers=1, bidirectional=True, batch_first=True) 
        self.feat_size = feat_size
        self.attn = nn.MultiheadAttention(feat_size, 4, batch_first=True)
        self.activationtrans = nn.LeakyReLU(negative_slope=0.03)
        self.layernorm1 = nn.LayerNorm(feat_size)
        self.layernorm2 = nn.LayerNorm(feat_size)
        self.lin = nn.Linear(feat_size*2, feat_size)
        
    def forward(self, x):
        b, c, t, f = x.size()
        # Reshape for time-wise processing
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)

        gru_out, _ = self.gru(x)
        x = x + self.lin(self.activationtrans(gru_out))
        x = self.layernorm2(x)

        attn_out, _ = self.attn(x, x, x)
        x = self.layernorm1(x + attn_out)
        
        return x.view(b, f, t, c).permute(0, 3, 2, 1)

class LCT(nn.Module):
    def __init__(self):
        super().__init__()
        k, s = [2, 3], [1, 2]
        self.conv1 = nn.Conv2d(4, 16, kernel_size=k, stride=s)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=k, stride=s)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=k, stride=s)
        
        self.activation = nn.LeakyReLU(negative_slope=0.03, inplace=True)
        self.GRUf1 = GRUblockf()
        self.GRUf2 = GRUblockf()
        self.GRUt1 = GRUblockt()
        self.layernorm = nn.LayerNorm(64)
        self.out = nn.Linear(64, 32 * 5)
        self.pad = nn.ConstantPad2d((0, 0, 3, 0), 0.0)
        
    def forward(self, x):
        x = self.pad(x) 
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))

        b, c, t, f = x.size()
        x = self.layernorm(x.permute(0, 2, 3, 1).contiguous().view(b*t, f, c))
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)

        x = self.GRUf2(self.GRUt1(self.GRUf1(x)))

        b, c, t, f = x.size()
        x = self.out(x.permute(0, 2, 3, 1).contiguous().view(b*t, f, c))
        x = x.view(b, t, f, 160).permute(0, 3, 1, 2)
        
        # Split into mean and variance components
        return x[:,:32,:,:], x[:,32:64,:,:], x[:,64:96,:,:], x[:,96:128,:,:], x[:,128:160,:,:]

class LCT_de(nn.Module):
    def __init__(self):
        super().__init__()
        k, s = [2, 3], [1, 2]
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=k, stride=s)
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=k, stride=s, output_padding=(0, 1))
        self.deconv4 = nn.ConvTranspose2d(16, 2, kernel_size=k, stride=s)
        self.activation = nn.LeakyReLU(negative_slope=0.03, inplace=True)
        
    def forward(self, x):
        x = self.activation(self.deconv2(x))
        x = self.activation(self.deconv3(x))
        return self.deconv4(x)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = LCT()
        self.decoder = LCT_de()
        self.reg = nn.ReLU()

    def sample_cmplx_Gaussian(self, mu_real, mu_imag, logvar, tau_real, tau_imag, eps=1e-8):
        # Calculate complex covariance parameters
        sigma_sq = torch.exp(logvar).clamp(min=eps)   
        tau_abs_sq = (tau_real**2 + tau_imag**2).clamp_min(eps)
        
        # Ensure pseudo-covariance is valid
        ratio = torch.ones_like(sigma_sq)
        mask = tau_abs_sq > sigma_sq**2
        ratio[mask] = sigma_sq[mask] / torch.sqrt(tau_abs_sq[mask])
         
        tau_real, tau_imag = tau_real * ratio, tau_imag * ratio

        # Cholesky decomposition for complex distribution
        l11 = torch.sqrt(self.reg(0.5 * (sigma_sq + tau_real)) + eps)
        l21 = (0.5 * tau_imag) / l11
        l22 = torch.sqrt(self.reg(0.5 * (sigma_sq - tau_real) - l21**2) + eps)

        x_eps = torch.randn_like(mu_real)
        y_eps = torch.randn_like(mu_imag)

        z_real = mu_real + l11 * x_eps
        z_imag = mu_imag + l21 * x_eps + l22 * y_eps
        return z_real, z_imag

    def forward(self, x):
        mu_r, mu_i, lv, tr, ti = self.encoder(x)
        z_r, z_i = self.sample_cmplx_Gaussian(mu_r, mu_i, lv, tr, ti)

        recon_x = self.decoder(torch.cat((z_r, z_i), dim=1))
        return recon_x[:, :, :-3, :], mu_r, mu_i, lv, tr, ti