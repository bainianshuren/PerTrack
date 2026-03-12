import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones.lightweight_yolox import LightweightYOLOX
from .dwa_conv import DWAConv
from .akf import AdaptiveKalmanFilter
from .sda import SDA

class PerTrack(nn.Module):
    """PerTrack整体模型封装
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device(config["device"])
        
        # 核心模块
        self.backbone = LightweightYOLOX(num_classes=config["model"]["num_classes"])
        self.dwa_conv = DWAConv(in_channels=256, dilation_rates=config["model"]["dwa_conv_rates"])
        self.sda = SDA(reid_feat_dim=config["model"]["reid_feat_dim"])
        
        # AKF参数
        self.akf_q_base = config["model"]["akf_q_base"]
        self.akf_r0 = config["model"]["akf_r0"]
        self.epsilon = config["model"]["epsilon"]
        
        # 跟踪器状态
        self.trackers = {}  # {track_id: AKF实例}
        self.next_id = 1

    def forward(self, x, prev_feats=None):
        """
        Args:
            x: 输入图片 [B, 3, H, W]
            prev_feats: 上一帧特征（可选）
        Returns:
            det_bboxes: 检测框 [B, N, 4]
            det_confs: 检测置信度 [B, N]
            track_results: 跟踪结果 [B, N, 5] (x1,y1,x2,y2,track_id)
        """
        # 1. 骨干网络特征提取
        backbone_out = self.backbone(x)
        features = backbone_out["features"][-1]  # 最后一层特征
        
        # 2. DWA-Conv增强特征
        enhanced_feat = self.dwa_conv(features)
        
        # 3. 检测
        det_bboxes = self._dummy_detection(enhanced_feat)
        det_confs = torch.ones(det_bboxes.shape[:2], device=self.device)
        
        # 4. 跟踪（AKF + SDA）
        track_results = self._tracking(x, det_bboxes[0], det_confs[0])
        
        return {
            "det_bboxes": det_bboxes,
            "det_confs": det_confs,
            "track_results": track_results.unsqueeze(0),
            "features": enhanced_feat
        }

    def _dummy_detection(self, feat):
        B, C, H, W = feat.shape
        # 生成随机检测框
        num_boxes = 10
        bboxes = torch.rand(B, num_boxes, 4, device=self.device)
        bboxes[..., 0] *= W * 32  # x1
        bboxes[..., 1] *= H * 32  # y1
        bboxes[..., 2] = bboxes[..., 0] + torch.rand(B, num_boxes, device=self.device) * 50  # x2
        bboxes[..., 3] = bboxes[..., 1] + torch.rand(B, num_boxes, device=self.device) * 100  # y2
        return bboxes

    def _tracking(self, img, det_bboxes, det_confs):
        """跟踪逻辑：AKF预测 + SDA匹配"""
        # 1. AKF预测所有活跃轨迹
        pred_bboxes = []
        pred_ids = []
        pred_feats = []
        
        for track_id, akf in self.trackers.items():
            pred_bbox = akf.predict()
            pred_bboxes.append(pred_bbox)
            pred_ids.append(track_id)
            pred_feats.append(torch.rand(self.config["model"]["reid_feat_dim"], device=self.device))
        
        if len(pred_bboxes) == 0:
            # 初始化轨迹
            track_results = []
            for i, bbox in enumerate(det_bboxes):
                track_results.append([*bbox.cpu().numpy(), self.next_id])
                # 初始化AKF
                self.trackers[self.next_id] = AdaptiveKalmanFilter(
                    bbox.cpu().numpy(),
                    self.akf_q_base,
                    self.akf_r0,
                    self.epsilon
                )
                self.next_id += 1
            return torch.tensor(track_results, device=self.device)
        
        # 2. SDA匹配
        pred_bboxes = torch.tensor(pred_bboxes, device=self.device)
        pred_feats = torch.stack(pred_feats)
        
        # 裁剪检测框区域（简化版）
        det_imgs = []
        for bbox in det_bboxes:
            x1, y1, x2, y2 = bbox.int()
            crop = F.interpolate(
                img[:, y1:y2, x1:x2].unsqueeze(0),
                size=(64, 32),
                mode="bilinear"
            ).squeeze(0)
            det_imgs.append(crop)
        det_imgs = torch.stack(det_imgs)
        
        # SDA计算相似度
        sim_matrix, weights = self.sda(
            img[0],  # 取第一张图片
            pred_bboxes,
            pred_feats,
            det_bboxes,
            det_imgs
        )
        
        # 3. 匹配（匈牙利算法）
        from scipy.optimize import linear_sum_assignment
        cost_matrix = 1 - sim_matrix.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 4. 更新匹配的轨迹
        matched_det = set()
        track_results = []
        
        for r, c in zip(row_ind, col_ind):
            if sim_matrix[r, c] > self.config["track"]["cosine_thresh"]:
                track_id = pred_ids[r]
                det_bbox = det_bboxes[c]
                det_conf = det_confs[c]
                
                # 计算IoU
                iou = self._compute_iou(pred_bboxes[r].cpu().numpy(), det_bbox.cpu().numpy())
                
                # 更新AKF
                self.trackers[track_id].update(det_bbox.cpu().numpy(), det_conf.item(), iou)
                
                track_results.append([*det_bbox.cpu().numpy(), track_id])
                matched_det.add(c)
        
        # 5. 新增轨迹
        for i, bbox in enumerate(det_bboxes):
            if i not in matched_det:
                track_results.append([*bbox.cpu().numpy(), self.next_id])
                self.trackers[self.next_id] = AdaptiveKalmanFilter(
                    bbox.cpu().numpy(),
                    self.akf_q_base,
                    self.akf_r0,
                    self.epsilon
                )
                self.next_id += 1
        
        return torch.tensor(track_results, device=self.device)

    def _compute_iou(self, bbox1, bbox2):
        """计算单个IoU"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        return inter / (area1 + area2 - inter + 1e-6)

    def reset(self):
        """重置跟踪器状态"""
        self.trackers = {}
        self.next_id = 1

    def compute_params(self):
        """计算总参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
