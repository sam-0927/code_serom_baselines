import torch
from torch import nn

class GRUblockf(nn.Module):
    def __init__(self, feat_size=64):
        super().__init__()
        # Bidirectional GRU doubles the feature size, requiring the linear layer to map it back
        self.gru = nn.GRU(input_size=feat_size, hidden_size=feat_size, num_layers=1, bidirectional=True, batch_first=True)  
        self.feat_size = feat_size
        self.attn = nn.MultiheadAttention(feat_size, 4, batch_first=True)
        self.activationtrans = nn.LeakyReLU(negative_slope=0.03)
        self.layernorm1 = nn.LayerNorm(feat_size)
        self.layernorm2 = nn.LayerNorm(feat_size)
        self.lin = nn.Linear(feat_size * 2, feat_size)
        
    def forward(self, x):
        b, c, t, f = x.size()
        # Reshape for frequency-domain processing
        x = x.permute(0, 2, 3, 1).contiguous().view(b*t, f, c)

        gru_out, _ = self.gru(x)
        x_temp = self.lin(self.activationtrans(gru_out))
        x = self.layernorm2(x + x_temp)
        
        x_temp, _ = self.attn(x, x, x)
        x = self.layernorm1(x + x_temp)
        
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
        # Reshape for time-domain processing
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        
        gru_out, _ = self.gru(x)
        x_temp = self.lin(self.activationtrans(gru_out))
        x = self.layernorm2(x + x_temp)

        x_temp, _ = self.attn(x, x, x)
        x = self.layernorm1(x + x_temp)
        
        return x.view(b, f, t, c).permute(0, 3, 2, 1)

class Cruse(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = [3, 7]
        stride = [1, 2]

        # Encoder layers
        self.conv1 = nn.Conv2d(2, 16, kernel_size=kernel_size, stride=stride)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride)
        
        # Decoder layers with output padding for dimension matching
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=kernel_size, stride=stride, output_padding=(0, 1))
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=kernel_size, stride=stride, output_padding=(0, 1))
        self.deconv4 = nn.ConvTranspose2d(16, 2, kernel_size=kernel_size, stride=stride)

        self.activation = nn.LeakyReLU(negative_slope=0.03, inplace=True)
        self.activations = nn.Sigmoid()
        
        # Skip connections
        self.skip2 = nn.Conv2d(64, 64, kernel_size=1, groups=64)
        self.skip3 = nn.Conv2d(32, 32, kernel_size=1, groups=32)
        self.skip4 = nn.Conv2d(16, 16, kernel_size=1, groups=16)

        # Bottleneck GRU blocks
        self.GRUf1, self.GRUf2, self.GRUf3, self.GRUf4 = [GRUblockf() for _ in range(4)]
        self.GRUt1, self.GRUt2, self.GRUt3, self.GRUt4 = [GRUblockt() for _ in range(4)]
        
        self.layernorm = nn.LayerNorm(64)
        self.act_final = nn.ReLU()
        
    def forward(self, x):
        # Encoder stage
        encoded = []
        x = self.activation(self.conv1(x))
        encoded.append(x)
        x = self.activation(self.conv2(x))
        encoded.append(x)
        x = self.activation(self.conv3(x))
        encoded.append(x) 

        # Feature normalization
        b, c, t, f = x.size()
        x = x.permute(0, 2, 3, 1).contiguous().view(b*t, f, c)
        x = self.layernorm(x)
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)

        # Bottleneck processing
        x = self.GRUf1(x)
        x = self.GRUt1(x)
        x = self.GRUf2(x)
        x = self.GRUt2(x)
        x = self.GRUf3(x)
        x = self.GRUt3(x)
        x = self.GRUf4(x)
        x = self.GRUt4(x)
        
        # Decoder stage with skip connections
        x = self.activation(self.deconv2(x + self.skip2(encoded[-1])))
        x = self.activation(self.deconv3(x + self.skip3(encoded[-2])))
        x = self.deconv4(x + self.skip4(encoded[-3]))
        
        return x