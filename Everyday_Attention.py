import torch
import torch.nn as nn
import torch.nn.functional as F


class Squeeze_n_Excitation(nn.Module):
    def __init__(self, 
                 in_channel, 
                 reduction=16, 
                 activation=nn.ReLU
                 ):
        super(Squeeze_n_Excitation, self).__init__()
        self.fc1 = nn.Linear(in_channel, in_channel // reduction, bias=False)
        self.activation = activation()
        self.fc2 = nn.Linear(in_channel // reduction, in_channel, bias=False)

    def forward(self, x:torch.tensor, mask:torch.tensor=None):
        '''`x`: [`B`, `L`, `C`], `mask`: [`B`, `L`], \\
        `B` is batch size, `L` is sequence length, `C` is channel size.'''
        input = x
        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = torch.sum(x*mask, dim=1, keepdim=True) / torch.sum(mask, dim=1, keepdim=True)
        else:
            x = torch.mean(x, dim=1, keepdim=True)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return input * x


class Gated_Addition(nn.Module):
    def __init__(self, 
                 channel, 
                 ) -> None:
        super().__init__()
        '''channel gate'''
        self.fc1 = nn.Linear(channel*2, channel//2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(channel//2, channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x = torch.concat([x1,x2], dim=-1)
        x = self.fc1(x)
        x = x.mean(dim=1, keepdim=True)
        x = self.act(x)
        x = self.fc2(x)
        gate = self.sigmoid(x)
        x = x1*gate + x2*(1-gate)
        return x


class Channel_Attention(nn.Module):
    def __init__(self, 
                 in_channel, 
                 reduction=16, 
                 activation=nn.ReLU
                 ):
        super(Channel_Attention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv1d(in_channel, in_channel // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            activation(),
            nn.Conv1d(in_channel // reduction, in_channel, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, x:torch.tensor):
        '''`x`: [`B`, `L`, `C`], \\
        `B` is batch size, `L` is sequence length, `C` is channel size.'''
        input = x
        x = x.permute(0,2,1)
        x_max = self.max_pool(x)
        x_max = self.shared_mlp(x_max)
        x_avg = self.avg_pool(x)
        x_avg = self.shared_mlp(x_avg)
        x = torch.sigmoid(x_max + x_avg)
        x = x.permute(0,2,1)
        return input * x


class Spatial_Attention(nn.Module):
    def __init__(self, 
                 ):
        super(Spatial_Attention, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, x):
        '''`x`: [`B`, `L`, `C`], \\
        `B` is batch size, `L` is sequence length, `C` is channel size.'''
        input = x
        x_max = torch.max(x, dim=-1, keepdim=True)[0]
        x_avg = torch.mean(x, dim=-1, keepdim=True)
        x = torch.cat([x_max, x_avg], dim=-1)
        x = self.conv(x.permute(0,2,1)).permute(0,2,1)
        x = torch.sigmoid(x)
        return input * x
        

class CBAM(nn.Module):
    def __init__(self, 
                 in_channel, 
                 reduction=16, 
                 activation=nn.ReLU
                 ):
        super(CBAM, self).__init__()
        self.ca = Channel_Attention(in_channel, reduction, activation)
        self.sa = Spatial_Attention()

    def forward(self, x:torch.tensor):
        '''`x`: [`B`, `L`, `C`], \\
        `B` is batch size, `L` is sequence length, `C` is channel size.'''
        x = self.ca(x)
        x = self.sa(x)
        return x


class attention_pooling(nn.Module):
    def __init__(self, 
                 in_dim, 
                 h_num = 16, 
                 h_dim = 32,
                 ) -> None:
        super().__init__()
        self.h_num = h_num
        self.h_dim = h_dim
        self.qkv = nn.Conv1d(in_dim, h_num*h_dim*3, 3, 1, 1, bias=False)

    def forward(self, x:torch.tensor, mask:torch.tensor):
        '''
        Attention pooling \n
        `x`: tensor of shape (B,T,D) \n
        `mask`: tensor of shape (B,T) \n
        '''
        B,T,D = x.shape
        
        q, k, v = self.qkv(x.permute(0,2,1)).reshape(B,self.h_num,self.h_dim,3,T).permute(0,1,4,2,3).chunk(3, dim=-1)
        q = q.squeeze(-1) ; k = k.squeeze(-1) ; v = v.squeeze(-1) # (B,h_num,T,h_dim)
        
        mask = mask.reshape(B,1,T,1)
        q = torch.sum(q*mask, dim=2, keepdim=True) / torch.sum(mask, dim=2, keepdim=True)
        att = torch.softmax(
            q @ k.transpose(-2,-1) / (self.h_dim**0.5) - (1.-mask).transpose(-2,-1)*1e6, 
            dim=-1) # (B,h_num,1,T)
        x = att @ v # (B,h_num,1,h_dim)
        x = x.reshape(B, -1) # (B,h_num*h_dim)
        return x


if __name__ == '__main__':
    B=10 ; L=100 ; C=64
    x = torch.randn(B, L, C)
    print(x.shape)

    model = Squeeze_n_Excitation(C)
    print(model(x).shape)

    model = Channel_Attention(C)
    print(model(x).shape)
