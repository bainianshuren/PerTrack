import torch
import numpy as np

def box_iou(boxes1, boxes2):
    """
    计算IoU
    Args:
        boxes1: (N, 4) [x1,y1,x2,y2]
        boxes2: (M, 4) [x1,y1,x2,y2]
    Returns:
        iou_matrix: (N, M)
    """
    if isinstance(boxes1, np.ndarray):
        boxes1 = torch.from_numpy(boxes1)
    if isinstance(boxes2, np.ndarray):
        boxes2 = torch.from_numpy(boxes2)
    
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    
    # 扩展维度
    boxes1 = boxes1.unsqueeze(1).expand(N, M, 4)
    boxes2 = boxes2.unsqueeze(0).expand(N, M, 4)
    
    # 计算交集
    x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
    x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
    y2 = torch.min(boxes1[..., 3], boxes2[..., 3])
    
    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # 计算并集
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union_area = area1 + area2 - inter_area
    
    # 计算IoU
    iou = inter_area / (union_area + 1e-6)
    
    return iou

def xyxy2cxcywh(boxes):
    """
    转换格式：x1y1x2y2 -> cxcywh
    """
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
    
    cx = (boxes[..., 0] + boxes[..., 2]) / 2
    cy = (boxes[..., 1] + boxes[..., 3]) / 2
    w = boxes[..., 2] - boxes[..., 0]
    h = boxes[..., 3] - boxes[..., 1]
    
    return torch.stack([cx, cy, w, h], dim=-1)

def cxcywh2xyxy(boxes):
    """
    转换格式：cxcywh -> x1y1x2y2
    """
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
    
    x1 = boxes[..., 0] - boxes[..., 2] / 2
    y1 = boxes[..., 1] - boxes[..., 3] / 2
    x2 = boxes[..., 0] + boxes[..., 2] / 2
    y2 = boxes[..., 1] + boxes[..., 3] / 2
    
    return torch.stack([x1, y1, x2, y2], dim=-1)
