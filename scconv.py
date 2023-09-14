import torch 
import torch.nn.functional as F  
import torch.nn as nn  


class GroupNorm2d(nn.Module):

    def __init__(self, n_groups: int = 16, n_channels: int = 16, eps: float = 1e-10):
        super(GroupNorm2d, self).__init__()  
        assert n_channels % n_groups == 0 
        self.n_groups = n_groups  
        self.gamma = nn.Parameter(torch.randn(n_channels, 1, 1))  # learnable gamma
        self.beta = nn.Parameter(torch.zeros(n_channels, 1, 1))  # learnable beta
        self.eps = eps 

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.reshape(N, self.n_groups, -1) 
        mean = x.mean(dim=2, keepdim=True)  
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps) 
        x = x.reshape(N, C, H, W)  
        return x * self.gamma + self.beta  


# Spatial and Reconstruct Unit
class SRU(nn.Module):

    def __init__(
            self,
            n_channels: int,  # in_channels
            n_groups: int = 16,  # 16
            gate_treshold: float = 0.5,  # 0.5
    ):
        super().__init__()  

        # initialize GroupNorm2d
        self.gn = GroupNorm2d(n_groups=n_groups, n_channels=n_channels)
        self.gate_treshold = gate_treshold  
        self.sigomid = nn.Sigmoid()  

    def forward(self, x):
        gn_x = self.gn(x) 
        w_gamma = self.gn.gamma / sum(self.gn.gamma)  # cal gamma weight
        reweights = self.sigomid(gn_x * w_gamma)  # importance

        info_mask = reweights >= self.gate_treshold
        noninfo_mask = reweights < self.gate_treshold
        x_1 = info_mask * x  
        x_2 = noninfo_mask * x  
        x = self.reconstruct(x_1, x_2) 
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


# Channel Reduction Unit
class CRU(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, alpha: float = 1 / 2, squeeze_radio: int = 2, groups: int = 2):
        super().__init__()

        self.up_channel = up_channel = int(alpha * in_channels)
        self.low_channel = low_channel = in_channels - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)

        self.GWC = nn.Conv2d(up_channel // squeeze_radio, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups) 
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, out_channels, kernel_size=1, bias=False)

        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, out_channels - low_channel // squeeze_radio, kernel_size=1, bias=False) 
        self.pool = nn.AdaptiveAvgPool2d(1)  
        
    def forward(self, x):

        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)

        y1 = self.GWC(up) + self.PWC1(up)

        y2 = torch.cat([self.PWC2(low), low], dim=1)

        s1 = self.pool(y1)
        s2 = self.pool(y2)
        s = torch.cat([s1, s2], dim=1)
        beta = F.softmax(s, dim=1)
        beta1, beta2 = torch.split(beta, beta.size(1) // 2, dim=1)
        y = beta1 * y1 + beta2 * y2
        return y


# Squeeze and Channel Reduction Convolution
class ScConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, n_groups: int = 16, gate_treshold: float = 0.5, alpha: float = 1 / 2, squeeze_radio: int = 2, groups: int = 2):
        super().__init__()

        self.SRU = SRU(in_channels, n_groups=n_groups, gate_treshold=gate_treshold, torch_gn=False) 
        self.CRU = CRU(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, alpha=alpha, squeeze_radio=squeeze_radio, groups=groups)

    def forward(self, x):
        x = self.SRU(x)  
        x = self.CRU(x) 
        return x


if __name__ == '__main__':
    from thop import profile

    x1 = torch.randn(16, 96, 224, 224)
    x2 = torch.randn(16, 96, 224, 224) 
    conv2d_model = nn.Sequential(nn.Conv2d(96, 64, kernel_size=3))
    flops1, params1 = profile(conv2d_model, (x1, ))
    scconv_model = ScConv(96, 64, kernel_size=3, alpha=1 / 2, squeeze_radio=2)  # out_channels > in_channels * (1-alpha) / squeeze_radio
    flops2, params2 = profile(scconv_model, (x2, ))
    print(f'model:["Conv2d"], FLOPS: [{flops1}], Params: [{params1}]') 
    print(f'model:["scConv"], FLOPS: [{flops2}], Params: [{params2}]')
