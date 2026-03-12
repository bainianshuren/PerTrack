import cv2
import numpy as np
import os

def visualize_tracking(img, track_results, save_path=None):
    """
    可视化跟踪结果（与论文图8一致）
    Args:
        img: (H, W, 3) RGB图片
        track_results: list of [x1,y1,x2,y2,track_id]
        save_path: 保存路径
    Returns:
        vis_img: 可视化后的图片
    """
    # 转换为BGR（OpenCV格式）
    vis_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 为每个track_id分配唯一颜色
    color_map = {}
    for result in track_results:
        track_id = int(result[4])
        if track_id not in color_map:
            # 随机颜色
            color_map[track_id] = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            )
    
    # 绘制跟踪框和ID
    for result in track_results:
        x1, y1, x2, y2, track_id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        track_id = int(track_id)
        color = color_map[track_id]
        
        # 绘制框
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        
        # 绘制ID背景
        label = f"ID: {track_id}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
        cv2.rectangle(
            vis_img, 
            (x1, label_y - label_size[1] - 5),
            (x1 + label_size[0], label_y + 5),
            color, 
            -1
        )
        
        # 绘制ID文本
        cv2.putText(
            vis_img, 
            label, 
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    # 保存图片
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, vis_img)
    
    # 转换回RGB
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    
    return vis_img
