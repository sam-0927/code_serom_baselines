import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math


class GRUblockf(nn.Module):
    def __init__(self, feat_size=64):
        # initilization
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

        # mask = torch.triu(torch.full((t, t), float('-inf'),device = torch.device('cuda')), diagonal=1)
        mask = torch.triu(torch.full((t, t), float('1'),device = torch.device('cuda')), diagonal=-64) - torch.triu(torch.full((t, t), float('1'),device = torch.device('cuda')), diagonal=1)
        mask = (mask - torch.full((t, t), float('1'),device = torch.device('cuda'))) * 1e9
        x_temp, _ = self.attn(x, x, x, attn_mask=mask, is_causal=True)
        x = x + x_temp
        x = self.layernorm1(x)
        
        x = x.view(b, f, t, c).permute(0, 3, 2, 1)
            
        return x
        



class Cruse(nn.Module):

    def __init__(self):
        # initilization
        # symmetric U-Net
        super().__init__()
        
        kernel_size = [2,3]
        stride = [1,2]


        self.conv1 = nn.Conv2d(1, 16, kernel_size=kernel_size, stride=stride)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride)
        
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=kernel_size, stride=stride)
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=kernel_size, stride=stride, output_padding=(0, 1))
        self.deconv4 = nn.ConvTranspose2d(16, 1, kernel_size=kernel_size, stride=stride)

        self.activation = nn.LeakyReLU(negative_slope=0.03, inplace=True)
        self.activations = nn.Sigmoid()
        
        # define the layers of conv skip connections
        self.skip2 = nn.Conv2d(64, 64, kernel_size=1, groups=64)
        self.skip3 = nn.Conv2d(32, 32, kernel_size=1, groups=32)
        self.skip4 = nn.Conv2d(16, 16, kernel_size=1, groups=16)

        self.GRUf1 = GRUblockf()
        self.GRUf2 = GRUblockf()
        self.GRUt1 = GRUblockt()
        self.layernorm = nn.LayerNorm(64)

        self.pad = nn.ConstantPad2d((0, 0, 2, 0), 0.0)
        self.act_final = nn.ReLU()
        
        
    def forward(self, x):
        x = self.pad(x) 
        encoded = []
        x = self.activation(self.conv1(x))
        encoded.append(x)
        x = self.activation(self.conv2(x))
        encoded.append(x)
        x = self.activation(self.conv3(x))
        encoded.append(x)

        b, c, t, f = x.size()
        x = x.permute(0, 2, 3, 1).contiguous().view(b*t, f, c)
        x = self.layernorm(x)
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)

        
        x = self.GRUf1(x)
        x = self.GRUt1(x)
        x = self.GRUf2(x)
        
        temp = self.skip2(encoded[-1])
        x = x + temp
        x = self.activation(self.deconv2(x))
        
        temp = self.skip3(encoded[-2])
        x = x + temp
        x = self.activation(self.deconv3(x))

        temp = self.skip4(encoded[-3])
        x = x + temp
        x = self.act_final(self.deconv4(x))
        
        x = x[:,:,:-2,:]
        
        return x