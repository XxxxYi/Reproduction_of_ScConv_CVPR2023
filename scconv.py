import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 的函数库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块


# 自定义 GroupBatchnorm2d 类，实现分组批量归一化
class GroupNorm2d(nn.Module):

    def __init__(self, n_groups: int = 16, n_channels: int = 16, eps: float = 1e-10):
        super(GroupNorm2d, self).__init__()  # 调用父类构造函数
        assert n_channels % n_groups == 0  # 断言 n_channels 能整除 n_groups
        self.n_groups = n_groups  # 设置分组数量
        self.gamma = nn.Parameter(torch.randn(n_channels, 1, 1))  # 创建可训练参数 gamma
        self.beta = nn.Parameter(torch.zeros(n_channels, 1, 1))  # 创建可训练参数 beta
        self.eps = eps  # 设置小的常数 eps 用于稳定计算

    def forward(self, x):
        N, C, H, W = x.size()  # 获取输入张量的尺寸
        x = x.reshape(N, self.n_groups, -1)  # 将输入张量重新排列为指定的形状
        mean = x.mean(dim=2, keepdim=True)  # 计算每个组的均值
        std = x.std(dim=2, keepdim=True)  # 计算每个组的标准差
        x = (x - mean) / (std + self.eps)  # 应用批量归一化
        x = x.reshape(N, C, H, W)  # 恢复原始形状
        return x * self.gamma + self.beta  # 返回归一化后的张量


# 自定义 SRU（Spatial and Reconstruct Unit）类
class SRU(nn.Module):

    def __init__(
            self,
            n_channels: int,  # 输入通道数
            n_groups: int = 16,  # 分组数，默认为16
            gate_treshold: float = 0.5,  # 门控阈值，默认为0.5
            torch_gn: bool = False  # 是否使用PyTorch内置的GroupNorm，默认为False
    ):
        super().__init__()  # 调用父类构造函数

        # 初始化 GroupNorm 层或自定义 GroupNorm2d 层
        self.gn = nn.GroupNorm(num_channels=n_channels, num_groups=n_groups) if torch_gn else GroupNorm2d(n_groups=n_groups, n_channels=n_channels)
        self.gate_treshold = gate_treshold  # 设置门控阈值
        self.sigomid = nn.Sigmoid()  # 创建 sigmoid 激活函数

    def forward(self, x):
        gn_x = self.gn(x)  # 应用分组批量归一化
        w_gamma = self.gn.gamma / sum(self.gn.gamma)  # 计算 gamma 权重
        reweights = self.sigomid(gn_x * w_gamma)  # 计算重要性权重

        # 门控机制
        info_mask = reweights >= self.gate_treshold  # 计算信息门控掩码
        noninfo_mask = reweights < self.gate_treshold  # 计算非信息门控掩码
        x_1 = info_mask * x  # 使用信息门控掩码
        x_2 = noninfo_mask * x  # 使用非信息门控掩码
        x = self.reconstruct(x_1, x_2)  # 重构特征
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)  # 拆分特征为两部分
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)  # 拆分特征为两部分
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)  # 重构特征并连接


# 自定义 CRU（Channel Reduction Unit）类
class CRU(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, alpha: float = 1 / 2, squeeze_radio: int = 2, groups: int = 2):
        super().__init__()  # 调用父类构造函数

        self.up_channel = up_channel = int(alpha * in_channels)  # 计算上层通道数
        self.low_channel = low_channel = in_channels - up_channel  # 计算下层通道数
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)  # 创建卷积层
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)  # 创建卷积层

        # 上层特征转换
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups)  # 创建卷积层
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, out_channels, kernel_size=1, bias=False)  # 创建卷积层

        # 下层特征转换
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, out_channels - low_channel // squeeze_radio, kernel_size=1, bias=False)  # 创建卷积层
        self.pool = nn.AdaptiveAvgPool2d(1)  # 创建自适应平均池化层

    def forward(self, x):
        # 分割输入特征
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)

        # 上层特征转换
        y1 = self.GWC(up) + self.PWC1(up)

        # 下层特征转换
        y2 = torch.cat([self.PWC2(low), low], dim=1)

        # 特征融合
        s1 = self.pool(y1)
        s2 = self.pool(y2)
        s = torch.cat([s1, s2], dim=1)
        beta = F.softmax(s, dim=1)
        beta1, beta2 = torch.split(beta, beta.size(1) // 2, dim=1)
        y = beta1 * y1 + beta2 * y2
        return y


# 自定义 ScConv（Squeeze and Channel Reduction Convolution）模型
class ScConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, n_groups: int = 16, gate_treshold: float = 0.5, alpha: float = 1 / 2, squeeze_radio: int = 2, groups: int = 2):
        super().__init__()  # 调用父类构造函数

        self.SRU = SRU(in_channels, n_groups=n_groups, gate_treshold=gate_treshold, torch_gn=False)  # 创建 SRU 层
        self.CRU = CRU(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, alpha=alpha, squeeze_radio=squeeze_radio, groups=groups)  # 创建 CRU 层

    def forward(self, x):
        x = self.SRU(x)  # 应用 SRU 层
        x = self.CRU(x)  # 应用 CRU 层
        return x


if __name__ == '__main__':
    from thop import profile

    x1 = torch.randn(16, 96, 224, 224)  # 创建随机输入张量
    x2 = torch.randn(16, 96, 224, 224)  # 创建随机输入张量
    conv2d_model = nn.Sequential(nn.Conv2d(96, 64, kernel_size=3))
    flops1, params1 = profile(conv2d_model, (x1, ))
    scconv_model = ScConv(96, 64, kernel_size=3, alpha=1 / 2, squeeze_radio=2)  # out_channels > in_channels * (1-alpha) / squeeze_radio
    flops2, params2 = profile(scconv_model, (x2, ))
    print(f'model:["Conv2d"], FLOPS: [{flops1}], Params: [{params1}]')  # 打印模型输出的形状
    print(f'model:["scConv"], FLOPS: [{flops2}], Params: [{params2}]')  # 打印模型输出的形状