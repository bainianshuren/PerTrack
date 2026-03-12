import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightReID(nn.Module):
    """轻量级ReID网络
    减少参数，适配小目标跟踪
    """
    def __init__(self, in_channels=3, feat_dim=128):
        super().__init__()
        self.feat_dim = feat_dim
        
        # 特征提取
        self.backbone = nn.Sequential(
            # 浅层特征
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # 中层特征
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # 深层特征
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 特征投影
        self.projection = nn.Sequential(
            nn.Linear(128, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        
        # 归一化
        self.l2_norm = lambda x: F.normalize(x, p=2, dim=1)

    def forward(self, x):
        """
        Args:
            x: 行人裁剪区域 [B, 3, H, W]
        Returns:
            feat: 归一化的ReID特征 [B, feat_dim]
        """
        # 特征提取
        feat = self.backbone(x).flatten(1)  # [B, 128]
        
        # 特征投影
        feat = self.projection(feat)  # [B, feat_dim]
        
        # L2归一化
        feat = self.l2_norm(feat)
        
        return feat
