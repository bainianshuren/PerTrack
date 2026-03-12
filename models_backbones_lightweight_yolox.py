import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    """基础卷积模块：Conv + BN + ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class CSPBlock(nn.Module):
    """轻量级CSP模块（减少参数）"""
    def __init__(self, in_channels, out_channels, num_blocks=2):
        super().__init__()
        self.split_conv = ConvBNReLU(in_channels, out_channels//2, 1, 1, 0)
        self.blocks = nn.Sequential(
            *[ConvBNReLU(out_channels//2, out_channels//2) for _ in range(num_blocks)]
        )
        self.concat_conv = ConvBNReLU(out_channels, out_channels, 1, 1, 0)
    
    def forward(self, x):
        x1, x2 = torch.chunk(self.split_conv(x), 2, dim=1)
        x2 = self.blocks(x2)
        return self.concat_conv(torch.cat([x1, x2], dim=1))

class LightweightYOLOX(nn.Module):
    """轻量级YOLOX骨干网络
    减少通道数和模块数，适配小目标跟踪
    """
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        
        # 主干网络（缩小通道数）
        self.stem = ConvBNReLU(in_channels, 16, 3, 2, 1)  # 1/2
        self.stage1 = nn.Sequential(
            ConvBNReLU(16, 32, 3, 2, 1),  # 1/4
            CSPBlock(32, 32, num_blocks=1)
        )
        self.stage2 = nn.Sequential(
            ConvBNReLU(32, 64, 3, 2, 1),  # 1/8
            CSPBlock(64, 64, num_blocks=2)
        )
        self.stage3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, 2, 1), # 1/16
            CSPBlock(128, 128, num_blocks=2)
        )
        self.stage4 = nn.Sequential(
            ConvBNReLU(128, 256, 3, 2, 1),# 1/32
            CSPBlock(256, 256, num_blocks=1)
        )
        
        # 检测头（轻量级）
        self.head = nn.Sequential(
            ConvBNReLU(256, 128, 1, 1, 0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes + 4)  # 分类 + 回归
        )
    
    def forward(self, x):
        # 特征提取
        x1 = self.stem(x)
        x2 = self.stage1(x1)
        x3 = self.stage2(x2)
        x4 = self.stage3(x3)
        x5 = self.stage4(x4)
        
        # 检测输出
        pred = self.head(x5)
        
        return {
            "features": [x2, x3, x4, x5],  # 多尺度特征
            "pred": pred  # 检测结果
        }
    
    def compute_params(self):
        """计算参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
