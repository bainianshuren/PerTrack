import numpy as np

def compute_mot_metrics(gt_boxes, gt_ids, pred_boxes, pred_ids):
    """计算MOT评价指标（MOTA/IDF1/IDS等）
    """
    # 基本统计
    gt_num = len(gt_ids)
    pred_num = len(pred_ids)
    
    # 计算TP/FP/FN
    iou_matrix = _compute_iou_matrix(gt_boxes, pred_boxes)
    matched_gt = set()
    matched_pred = set()
    tp = 0
    
    for i in range(gt_num):
        for j in range(pred_num):
            if iou_matrix[i, j] > 0.5 and j not in matched_pred:
                tp += 1
                matched_gt.add(i)
                matched_pred.add(j)
                break
    
    fp = pred_num - len(matched_pred)
    fn = gt_num - len(matched_gt)
    
    # 计算ID切换（IDS）
    ids = _compute_id_switch(gt_ids, pred_ids, iou_matrix)
    
    # 计算MOTA（论文公式5）
    mota = 1 - (fn + fp + ids) / max(gt_num, 1)
    
    # 计算IDF1
    idtp, idfp, idfn = _compute_id_metrics(gt_ids, pred_ids, iou_matrix)
    idf1 = 2 * idtp / max(2 * idtp + idfp + idfn, 1)
    
    return {
        "MOTA": mota * 100,
        "IDF1": idf1 * 100,
        "IDS": ids,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": tp / max(tp + fp, 1) * 100,
        "Recall": tp / max(tp + fn, 1) * 100
    }

def _compute_iou_matrix(boxes1, boxes2):
    """计算IoU矩阵"""
    iou_matrix = np.zeros((len(boxes1), len(boxes2)))
    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            iou_matrix[i, j] = _compute_iou(box1, box2)
    return iou_matrix

def _compute_iou(box1, box2):
    """计算单个IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return inter / (area1 + area2 - inter + 1e-6)

def _compute_id_switch(gt_ids, pred_ids, iou_matrix):
    """计算ID切换次数"""
    # 简化版计算
    ids = 0
    matched_ids = {}  # gt_id -> pred_id
    
    for i in range(len(gt_ids)):
        gt_id = gt_ids[i]
        # 找到最佳匹配
        if len(pred_ids) == 0:
            continue
        best_j = np.argmax(iou_matrix[i])
        pred_id = pred_ids[best_j]
        
        if gt_id in matched_ids:
            if matched_ids[gt_id] != pred_id:
                ids += 1
        else:
            matched_ids[gt_id] = pred_id
    
    return ids

def _compute_id_metrics(gt_ids, pred_ids, iou_matrix):
    """计算ID相关指标"""
    idtp = 0
    idfp = 0
    idfn = 0
    
    for i in range(len(gt_ids)):
        gt_id = gt_ids[i]
        if len(pred_ids) == 0:
            idfn += 1
            continue
        
        best_j = np.argmax(iou_matrix[i])
        if iou_matrix[i, best_j] > 0.5:
            if pred_ids[best_j] == gt_id:
                idtp += 1
            else:
                idfp += 1
                idfn += 1
        else:
            idfn += 1
    
    for j in range(len(pred_ids)):
        pred_id = pred_ids[j]
        # 检查是否是FP
        max_iou = np.max(iou_matrix[:, j]) if len(gt_ids) > 0 else 0
        if max_iou < 0.5:
            idfp += 1
    
    return idtp, idfp, idfn
