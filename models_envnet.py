import torch
import torch.nn as nn
import torch.nn.functional as F

class EnvNet(nn.Module):
    """场景感知子网络EnvNet
    提取场景特征：拥挤度、光照、遮挡程度
    """
    def __init__(self, in_channels=3, feat_dim=64):
        super().__init__()
        self.feat_dim = feat_dim
        
        # 场景特征提取
        self.scene_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, feat_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 场景特征分类（拥挤度：低/中/高；光照：暗/中/亮；遮挡：无/轻/重）
        self.crowd_head = nn.Linear(feat_dim, 3)
        self.light_head = nn.Linear(feat_dim, 3)
        self.occlusion_head = nn.Linear(feat_dim, 3)
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * 3, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        """
        Args:
            x: 输入图片 [B, 3, H, W]
        Returns:
            scene_feat: 场景特征 [B, feat_dim]
            scene_attr: 场景属性（拥挤度、光照、遮挡）
        """
        # 提取场景特征
        feat = self.scene_encoder(x).flatten(1)  # [B, feat_dim]
        
        # 场景属性预测
        crowd_logits = self.crowd_head(feat)
        light_logits = self.light_head(feat)
        occlusion_logits = self.occlusion_head(feat)
        
        # 融合特征
        scene_feat = self.fusion(torch.cat([
            F.softmax(crowd_logits, dim=1),
            F.softmax(light_logits, dim=1),
            F.softmax(occlusion_logits, dim=1)
        ], dim=1))
        
        # 场景属性（类别）
        scene_attr = {
            "crowd": torch.argmax(crowd_logits, dim=1),
            "light": torch.argmax(light_logits, dim=1),
            "occlusion": torch.argmax(occlusion_logits, dim=1)
        }
        
        return scene_feat, scene_attr
