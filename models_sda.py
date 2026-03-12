import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .envnet import EnvNet
from .reid import LightweightReID

class SDA(nn.Module):
    """场景感知动态关联模块(SDA)
        """
    def __init__(self, reid_feat_dim=128, scene_feat_dim=64):
        super().__init__()
        self.reid_feat_dim = reid_feat_dim
        self.scene_feat_dim = scene_feat_dim
        
        # 子网络
        self.env_net = EnvNet(feat_dim=scene_feat_dim)
        self.reid_net = LightweightReID(feat_dim=reid_feat_dim)
        
        # 动态权重网络
        self.weight_net = nn.Sequential(
            nn.Linear(scene_feat_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3),  # 运动/外观/场景权重
            nn.Softmax(dim=1)
        )

    def _compute_motion_similarity(self, pred_bboxes, det_bboxes):
        """计算运动相似度（IoU）"""
        ious = self._batch_iou(pred_bboxes, det_bboxes)
        return ious

    def _compute_appearance_similarity(self, pred_feats, det_feats):
        """计算外观相似度（余弦相似度）"""
        sim = torch.mm(pred_feats, det_feats.t())
        return sim

    def _compute_scene_similarity(self, scene_feat, pred_ids, det_ids):
        """计算场景相似度（基于场景特征的动态调整）"""
        # 简化版：场景特征越相似，相似度越高
        batch_size = len(pred_ids)
        sim = torch.ones((batch_size, len(det_ids)), device=scene_feat.device)
        return sim * scene_feat.mean()

    def _batch_iou(self, bboxes1, bboxes2):
        """批量计算IoU"""
        # bboxes1: [N, 4], bboxes2: [M, 4]
        N = bboxes1.shape[0]
        M = bboxes2.shape[0]
        
        # 扩展维度
        bboxes1 = bboxes1.unsqueeze(1).expand(N, M, 4)
        bboxes2 = bboxes2.unsqueeze(0).expand(N, M, 4)
        
        # 计算交集
        x1 = torch.max(bboxes1[..., 0], bboxes2[..., 0])
        y1 = torch.max(bboxes1[..., 1], bboxes2[..., 1])
        x2 = torch.min(bboxes1[..., 2], bboxes2[..., 2])
        y2 = torch.min(bboxes1[..., 3], bboxes2[..., 3])
        
        inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # 计算并集
        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        union_area = area1 + area2 - inter_area
        
        # 计算IoU
        iou = inter_area / (union_area + 1e-6)
        
        return iou

    def forward(self, img, pred_bboxes, pred_feats, det_bboxes, det_imgs):
        """
        Args:
            img: 整张图片 [3, H, W]
            pred_bboxes: 预测框 [N, 4]
            pred_feats: 预测框ReID特征 [N, feat_dim]
            det_bboxes: 检测框 [M, 4]
            det_imgs: 检测框裁剪图片 [M, 3, h, w]
        Returns:
            final_sim: 最终相似度矩阵 [N, M]
            weights: 动态权重 [3]
        """
        # 1. 提取场景特征
        scene_feat, scene_attr = self.env_net(img.unsqueeze(0))
        scene_feat = scene_feat.squeeze(0)
        
        # 2. 提取检测框外观特征
        det_feats = self.reid_net(det_imgs)
        
        # 3. 计算各模态相似度
        motion_sim = self._compute_motion_similarity(pred_bboxes, det_bboxes)  # [N, M]
        app_sim = self._compute_appearance_similarity(pred_feats, det_feats)   # [N, M]
        scene_sim = self._compute_scene_similarity(scene_feat, 
                                                  torch.arange(len(pred_bboxes)), 
                                                  torch.arange(len(det_bboxes)))  # [N, M]
        
        # 4. 动态权重计算
        weights = self.weight_net(scene_feat.unsqueeze(0)).squeeze(0)  # [3]
        motion_w, app_w, scene_w = weights
        
        # 5. 加权融合
        final_sim = motion_w * motion_sim + app_w * app_sim + scene_w * scene_sim
        
        return final_sim, weights
