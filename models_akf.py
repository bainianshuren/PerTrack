import numpy as np
import torch

class AdaptiveKalmanFilter:
    """自适应卡尔曼滤波(AKF)
    通过检测置信度调整观测噪声R，通过IoU调整过程噪声Q
    状态向量：[x, y, w, h, vx, vy, vw, vh]（中心坐标、宽高、对应速度）
    """
    def __init__(self, init_bbox, q_base=0.01, r0=0.1, epsilon=1e-6):
        self.epsilon = epsilon
        # 状态初始化（转换为中心坐标）
        x1, y1, x2, y2 = init_bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)  # 初始速度为0
        self.dim_x = 8
        self.dim_z = 4  # 观测：cx, cy, w, h
        
        # 状态转移矩阵F
        self.F = np.eye(self.dim_x, dtype=np.float32)
        for i in range(self.dim_z):
            self.F[i, self.dim_z + i] = 1.0
        
        # 观测矩阵H
        self.H = np.eye(self.dim_z, self.dim_x, dtype=np.float32)
        
        # 协方差矩阵P
        self.P = np.eye(self.dim_x, dtype=np.float32) * 10.0
        
        # 基础噪声矩阵
        self.Q_base = np.eye(self.dim_x, dtype=np.float32) * q_base
        self.Q = self.Q_base.copy()
        self.R0 = np.eye(self.dim_z, dtype=np.float32) * r0
        self.R = self.R0.copy()

    def predict(self):
        """预测步骤：状态+协方差"""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        # 转换回x1,y1,x2,y2格式
        cx, cy, w, h = self.x[:4]
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        return [x1, y1, x2, y2]

    def update(self, z, conf, iou):
        """更新步骤：自适应调整噪声+卡尔曼更新
        Args:
            z: 观测bbox [x1,y1,x2,y2]
            conf: 检测置信度（0~1）
            iou: 预测框与观测框的IoU（0~1）
        """
        # 转换为中心坐标
        x1, y1, x2, y2 = z
        z_cxcy = np.array([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], dtype=np.float32)
        
        # 1. 自适应调整噪声矩阵（论文公式1、2）
        self._adjust_noise(conf, iou)
        
        # 2. 卡尔曼标准更新
        y = z_cxcy - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        
        # 防止奇异矩阵
        try:
            S_inv = np.linalg.inv(S)
        except:
            S_inv = np.linalg.pinv(S)
        
        K = np.dot(np.dot(self.P, self.H.T), S_inv)
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.dim_x, dtype=np.float32)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

    def _adjust_noise(self, conf, iou):
        """调整观测噪声R和过程噪声Q"""
        # 观测噪声R：置信度越高，R越小
        conf = max(conf, self.epsilon)
        self.R = self.R0 / conf
        
        # 过程噪声Q：IoU越小，Q越接近Q_base
        self.Q = (1 - iou) * self.Q_base + iou * self.Q
