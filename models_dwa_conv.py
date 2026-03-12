import torch
import torch.nn as nn
import torch.nn.functional as F

class DWAConv(nn.Module):
    """动态加权空洞卷积模块(DWA-Conv)
    膨胀率r=1,2,3，动态权重生成，通道级加权融合
    """
    def __init__(self, in_channels, out_channels=None, dilation_rates=[1,2,3]):
        super(DWAConv, self).__init__()
        self.dilation_rates = dilation_rates
        out_channels = in_channels if out_channels is None else out_channels
        
        # 并行空洞卷积分支
        self.conv_branches = nn.ModuleList()
        for r in dilation_rates:
            self.conv_branches.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=r, dilation=r, bias=False)
            )
        
        # 动态权重生成网络（GAP + 两层FC）
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels * len(dilation_rates), in_channels * len(dilation_rates) // 4)
        self.fc2 = nn.Linear(in_channels * len(dilation_rates) // 4, len(dilation_rates))
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)
        
        # 输出卷积（通道融合）
        self.out_conv = nn.Conv2d(in_channels * len(dilation_rates), out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # 1. 多尺度特征提取
        branch_outs = []
        for conv in self.conv_branches:
            branch_outs.append(conv(x))  # [B, C, H, W] * 3
        
        # 2. 动态权重生成
        gap_outs = []
        for feat in branch_outs:
            gap = self.gap(feat).flatten(1)  # [B, C]
            gap_outs.append(gap)
        gap_concat = torch.cat(gap_outs, dim=1)  # [B, 3C]
        w = self.relu(self.fc1(gap_concat))  # [B, 3C/4]
        w = self.softmax(self.fc2(w))  # [B, 3] 权重分布，和为1
        
        # 3. 通道级加权
        weighted_outs = []
        for i, feat in enumerate(branch_outs):
            weight = w[:, i].unsqueeze(1).unsqueeze(2).unsqueeze(3)  # [B,1,1,1]
            weighted_outs.append(feat * weight)
        
        # 4. 特征融合与输出
        concat_feat = torch.cat(weighted_outs, dim=1)  # [B, 3C, H, W]
        out = self.bn(self.out_conv(concat_feat))  # [B, C, H, W]
        return out
    
    def compute_params(self):
        """计算参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
