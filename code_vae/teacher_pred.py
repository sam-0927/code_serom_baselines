import torch
import torch.nn as nn

class GRUblockf(nn.Module):
    def __init__(self, feat_size=256):
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
        x = x.permute(0, 2, 3, 1).contiguous().view(b*t, f, c)
        
        gru_out, _ = self.gru(x)
        x_temp = self.lin(self.activationtrans(gru_out))
        x = self.layernorm2(x + x_temp)
        
        x_temp, _ = self.attn(x, x, x)
        x = self.layernorm1(x + x_temp)
        
        return x.view(b, t, f, c).permute(0, 3, 1, 2)

class GRUblockt(nn.Module):
    def __init__(self, feat_size=256):
        super().__init__()
        self.gru = nn.GRU(input_size=feat_size, hidden_size=feat_size, num_layers=1, batch_first=True) 
        self.feat_size = feat_size
        self.attn = nn.MultiheadAttention(feat_size, 4, batch_first=True)
        self.activationtrans = nn.LeakyReLU(negative_slope=0.03)
        self.layernorm1 = nn.LayerNorm(feat_size)
        self.layernorm2 = nn.LayerNorm(feat_size)
        self.lin = nn.Linear(feat_size, feat_size)
        
    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)

        gru_out, _ = self.gru(x)
        x_temp = self.lin(self.activationtrans(gru_out))
        x = self.layernorm2(x + x_temp)

        # Fixed: Use x.device instead of hardcoded 'cuda'
        mask = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()
        x_temp, _ = self.attn(x, x, x, attn_mask=mask, is_causal=True)
        x = self.layernorm1(x + x_temp)
        
        return x.view(b, f, t, c).permute(0, 3, 2, 1)

class Teacher_Pred(nn.Module):
    def __init__(self):
        super().__init__()
        
        kernel_size = [3, 7]
        stride = [1, 2]

        # Keep original naming to ensure weight loading
        self.conv1 = nn.Conv2d(2, 64, kernel_size=kernel_size, stride=stride)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=kernel_size, stride=stride)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=kernel_size, stride=stride)

        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=kernel_size, stride=stride, output_padding=(0, 1))
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=kernel_size, stride=stride, output_padding=(0, 1))
        self.deconv4 = nn.ConvTranspose2d(64, 2, kernel_size=kernel_size, stride=stride)

        self.activation = nn.LeakyReLU(negative_slope=0.03, inplace=True)
        
        self.skip2 = nn.Conv2d(256, 256, kernel_size=1, groups=256)
        self.skip3 = nn.Conv2d(128, 128, kernel_size=1, groups=128)
        self.skip4 = nn.Conv2d(64, 64, kernel_size=1, groups=64)

        # Restore original variable names for GRU blocks
        self.GRUf1 = GRUblockf()
        self.GRUf2 = GRUblockf()
        self.GRUt1 = GRUblockt()
        self.GRUt2 = GRUblockt()
        self.GRUf3 = GRUblockf()
        self.GRUf4 = GRUblockf()
        self.GRUt3 = GRUblockt()
        self.GRUt4 = GRUblockt()
        
        self.layernorm = nn.LayerNorm(256)
        self.pad = nn.ConstantPad2d((0, 0, 3, 0), 0.0)
        
    def forward(self, x):
        x = self.pad(x) 
        encoded = []
        x = self.activation(self.conv1(x)); encoded.append(x)
        x = self.activation(self.conv2(x)); encoded.append(x)
        x = self.activation(self.conv3(x)); encoded.append(x)

        b, c, t, f = x.size()
        x = x.permute(0, 2, 3, 1).contiguous().view(b*t, f, c)
        x = self.layernorm(x)
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)

        # Original processing order
        x = self.GRUf1(x)
        x = self.GRUt1(x)
        x = self.GRUf2(x)
        x = self.GRUt2(x)
        x = self.GRUf3(x)
        x = self.GRUt3(x)
        x = self.GRUf4(x)
        x = self.GRUt4(x)
        
        temp = self.skip2(encoded[-1])
        x = x + temp
        x = self.activation(self.deconv2(x))
        
        temp = self.skip3(encoded[-2])
        x = x + temp
        x = self.activation(self.deconv3(x))

        temp = self.skip4(encoded[-3])
        x = x + temp
        
        x = self.deconv4(x)
        return x[:, :, :-3, :]