import torch
import numpy as np
from models import PerTrack

class PerTrackTracker:
    """PerTrack跟踪器封装
    高层API，方便调用
    """
    def __init__(self, config, weight_path=None):
        self.config = config
        self.device = torch.device(config["device"])
        
        # 加载模型
        self.model = PerTrack(config).to(self.device)
        
        # 加载权重
        if weight_path and weight_path.endswith(".pth"):
            self.load_weights(weight_path)
        
        # 模型设为评估模式
        self.model.eval()
        
        # 跟踪状态
        self.prev_feats = None
        self.frame_id = 0

    def load_weights(self, weight_path):
        """加载权重"""
        checkpoint = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded weights from {weight_path}")

    def track(self, img):
        """单帧跟踪
        Args:
            img: 输入图片 (H, W, 3) RGB格式
        Returns:
            track_results: 跟踪结果 list of [x1,y1,x2,y2,track_id]
        """
        self.frame_id += 1
        
        # 预处理
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # 前向传播
        with torch.no_grad():
            outputs = self.model(img_tensor, self.prev_feats)
        
        # 更新状态
        self.prev_feats = outputs["features"]
        
        # 解析结果
        track_results = outputs["track_results"][0].cpu().numpy()
        
        return track_results

    def reset(self):
        """重置跟踪器"""
        self.model.reset()
        self.prev_feats = None
        self.frame_id = 0
