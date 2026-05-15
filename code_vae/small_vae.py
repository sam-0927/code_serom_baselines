import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math


class GRUblockf(nn.Module):
    def __init__(self, feat_size=64):
        # initilization
        # symmetric U-Net
        super().__init__()

        self.gru1 = nn.GRU(input_size=feat_size//4, hidden_size=feat_size//4, num_layers=1, bidirectional=True, batch_first=True) 
        self.gru2 = nn.GRU(input_size=feat_size//4, hidden_size=feat_size//4, num_layers=1, bidirectional=True, batch_first=True) 
        self.gru3 = nn.GRU(input_size=feat_size//4, hidden_size=feat_size//4, num_layers=1, bidirectional=True, batch_first=True) 
        self.gru4 = nn.GRU(input_size=feat_size//4, hidden_size=feat_size//4, num_layers=1, bidirectional=True, batch_first=True) 
        self.feat_size = feat_size
        self.attn = nn.MultiheadAttention(feat_size, 4, batch_first=True)

        self.activationtrans = nn.LeakyReLU(negative_slope=0.03) # , inplace=True)
        self.layernorm1 = nn.LayerNorm(feat_size)
        self.layernorm2 = nn.LayerNorm(feat_size)
        self.lin = nn.Linear(feat_size * 2, feat_size)
        
    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 2, 3, 1).contiguous().view(b*t, f, c)

        gru_in = x
        feat_size = self.feat_size
        gru_out1,_ = self.gru1(gru_in[:,:,:feat_size//4])
        gru_out2,_ = self.gru2(gru_in[:,:,feat_size//4:feat_size//4*2])
        gru_out3,_ = self.gru3(gru_in[:,:,feat_size//4*2:feat_size//4*3])
        gru_out4,_ = self.gru4(gru_in[:,:,feat_size//4*3:feat_size//4*4])
        x_temp = torch.concat((gru_out1, gru_out2, gru_out3, gru_out4), 2)
        x_temp = self.lin(self.activationtrans(x_temp))
        x = x + x_temp
        x = self.layernorm2(x)
        
        x_temp, _ = self.attn(x, x, x)
        x = x + x_temp
        x = self.layernorm1(x)
        
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x

class GRUblockt(nn.Module):
    def __init__(self, feat_size=64):
        # initilization
        # symmetric U-Net
        super().__init__()

        self.gru1 = nn.GRU(input_size=feat_size//4, hidden_size=feat_size//4, num_layers=1, batch_first=True) 
        self.gru2 = nn.GRU(input_size=feat_size//4, hidden_size=feat_size//4, num_layers=1, batch_first=True) 
        self.gru3 = nn.GRU(input_size=feat_size//4, hidden_size=feat_size//4, num_layers=1, batch_first=True) 
        self.gru4 = nn.GRU(input_size=feat_size//4, hidden_size=feat_size//4, num_layers=1, batch_first=True) 
        self.feat_size = feat_size

        self.attn = nn.MultiheadAttention(feat_size, 4, batch_first=True)
        self.activationtrans = nn.LeakyReLU(negative_slope=0.03) # , inplace=True)
        self.layernorm1 = nn.LayerNorm(feat_size)
        self.layernorm2 = nn.LayerNorm(feat_size)
        self.lin = nn.Linear(feat_size, feat_size)
        
    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        
        gru_in = x
        feat_size = self.feat_size
        gru_out1,_ = self.gru1(gru_in[:,:,:feat_size//4])
        gru_out2,_ = self.gru2(gru_in[:,:,feat_size//4:feat_size//4*2])
        gru_out3,_ = self.gru3(gru_in[:,:,feat_size//4*2:feat_size//4*3])
        gru_out4,_ = self.gru4(gru_in[:,:,feat_size//4*3:feat_size//4*4])
        x_temp = torch.concat((gru_out1, gru_out2, gru_out3, gru_out4), 2)
        x_temp = self.lin(self.activationtrans(x_temp))
        x = x + x_temp
        x = self.layernorm2(x)

        mask = torch.triu(torch.full((t, t), float('1'),device = torch.device('cuda')), diagonal=-64) - torch.triu(torch.full((t, t), float('1'),device = torch.device('cuda')), diagonal=1)
        mask = (mask - torch.full((t, t), float('1'),device = torch.device('cuda'))) * 1e9
        x_temp, _ = self.attn(x, x, x, attn_mask=mask, is_causal=True)
        x = x + x_temp
        x = self.layernorm1(x)
        x = x.view(b, f, t, c).permute(0, 3, 2, 1)
            
        return x
        


class LCT(nn.Module):

    def __init__(self):
        # initilization
        # symmetric U-Net
        super().__init__()
        
        kernel_size = [2,3]
        stride = [1,2]


        self.conv1 = nn.Conv2d(2, 16, kernel_size=kernel_size, stride=stride)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride)
        


        self.activation = nn.LeakyReLU(negative_slope=0.03, inplace=True)
        self.activations = nn.Sigmoid()
        

        self.GRUf1 = GRUblockf()
        self.GRUf2 = GRUblockf()
        self.GRUt1 = GRUblockt()
        # self.GRUt2 = GRUblockt()
        self.layernorm = nn.LayerNorm(64)

        self.act_final = nn.ReLU()

        self.out = nn.Linear(64,32 * 5)
        self.pad = nn.ConstantPad2d((0, 0, 3, 0), 0.0)
        
        
    def forward(self, x):
        x = self.pad(x) 
        # encoded = []
        x = self.activation(self.conv1(x))
        # encoded.append(x)
        x = self.activation(self.conv2(x))
        # encoded.append(x)
        x = self.activation(self.conv3(x))
        # encoded.append(x)

        b, c, t, f = x.size()
        x = x.permute(0, 2, 3, 1).contiguous().view(b*t, f, c)
        x = self.layernorm(x)
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)

        x = self.GRUf1(x)
        x = self.GRUt1(x)
        x = self.GRUf2(x)

        b, c, t, f = x.size()
        x = x.permute(0, 2, 3, 1).contiguous().view(b*t, f, c)
        x = self.out(x)
        x = x.view(b, t, f, 32*5).permute(0, 3, 1, 2)
        
        return x[:,:32,:,:], x[:,32*1:32*2,:,:], x[:,32*2:32*3,:,:], x[:,32*3:32*4,:,:], x[:,32*4:32*5,:,:]

class LCT_de(nn.Module):

    def __init__(self):
        # initilization
        # symmetric U-Net
        super().__init__()
        
        kernel_size = [2,3]
        stride = [1,2]



        
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=kernel_size, stride=stride)
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=kernel_size, stride=stride, output_padding=(0, 1))
        self.deconv4 = nn.ConvTranspose2d(16, 2, kernel_size=kernel_size, stride=stride)
        # self.deconv4p = nn.ConvTranspose2d(16, 2, kernel_size=kernel_size, stride=stride)

        self.activation = nn.LeakyReLU(negative_slope=0.03, inplace=True)
        self.activations = nn.Sigmoid()
        
        # define the layers of conv skip connections
        # self.skip2 = nn.Conv2d(64, 64, kernel_size=1)
        # self.skip3 = nn.Conv2d(32, 32, kernel_size=1)
        # self.skip4 = nn.Conv2d(16, 16, kernel_size=1)


        # self.pad = nn.ConstantPad2d((0, 0, 2, 0), 0.0)
        self.act_final = nn.ReLU()
        self.act_finalp = nn.Tanh()
        
        
    def forward(self, x):
        
        # temp = self.skip2(encoded[-1])
        # x = x + temp
        x = self.activation(self.deconv2(x))
        
        # temp = self.skip3(encoded[-2])
        # x = x + temp
        x = self.activation(self.deconv3(x))

        # temp = self.skip4(encoded[-3])
        # x = x + temp
        
        x = self.deconv4(x)
        # xp = self.act_finalp(self.deconv4p(x))
        # x = torch.cat((xm, xp), dim=1)
        # x = x[:,:,:-3,:]
        
        return x



class small_VAE(nn.Module):
    def __init__(self):
        super(small_VAE, self).__init__()

        self.encoder = LCT()
        self.decoder =  LCT_de()
        self.reg = nn.ReLU()

    def sample_cmplx_Gaussian(self, mu_real, mu_imag, logvar, tau_real, tau_imag, eps=1e-8):
    # 数值稳定性检查（可选调试模式）
        debug_mode = False
        if debug_mode:
            for name, tensor in [('mu_real', mu_real), ('mu_imag', mu_imag), 
                               ('logvar', logvar), ('tau_real', tau_real), 
                               ('tau_imag', tau_imag)]:
                if not torch.all(torch.isfinite(tensor)):
                    print(f"Warning: {name} contains NaN/Inf")

        dtype = mu_real.dtype


        sigma_sq = torch.exp(logvar).clamp(min=eps)   
        sigma_sq_sq = sigma_sq**2                   

        # tau_abs_sq = tau_real**2 + tau_imag**2    
        tau_abs_sq = (tau_real**2 + tau_imag**2).clamp_min(eps)
        tau_abs_sq = torch.minimum(tau_abs_sq, sigma_sq_sq)

        curr_len = torch.sqrt(tau_abs_sq)             
        max_len = sigma_sq           


        ratio = torch.ones_like(curr_len)
        mask = curr_len > 0
        ratio[mask] = (max_len[mask] / curr_len[mask]).clamp(max=1.0)

      
        tau_real = tau_real * ratio
        tau_imag = tau_imag * ratio



    # 协方差矩阵分解
        A = 0.5 * (sigma_sq + tau_real)
        C = 0.5 * (sigma_sq - tau_real)
        B = 0.5 * tau_imag

        A = self.reg(A) + eps
        l11 = torch.sqrt(A)
        safe_l11 = torch.where(l11 < eps, eps, l11)
        l21 = B / safe_l11

        C_reg = self.reg(C - l21**2) + eps
        l22 = torch.sqrt(C_reg)

    # 生成噪声
        xepsilon = torch.randn_like(mu_real, dtype=dtype)
        yepsilon = torch.randn_like(mu_imag, dtype=dtype)

        z_real = mu_real + l11 * xepsilon
        z_imag = mu_imag + l21 * xepsilon + l22 * yepsilon

        return z_real, z_imag



    def forward(self, x):
        # org_size = x.size()
        # batch = org_size[0]
        # x = x.view(batch, -1)

        re_mu, im_mu, r, re_s, im_s = self.encoder(x)
        re_z, im_z = self.sample_cmplx_Gaussian(re_mu, im_mu, r, re_s, im_s)

        z = torch.cat((re_z, im_z), dim=1)
        recon_x = self.decoder(z)

        recon_x = recon_x[:,:,:-3,:]

        return recon_x, re_mu, im_mu, r, re_s, im_s