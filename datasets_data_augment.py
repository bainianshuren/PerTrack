import cv2
import numpy as np
import random

class MOTDataAugment:
    """MOT数据集数据增强
    包含：随机裁剪、随机翻转、颜色抖动、随机缩放
    """
    def __init__(self, config):
        self.config = config
        self.aug_params = {
            "crop_prob": 0.5,
            "flip_prob": 0.5,
            "color_jitter_prob": 0.8,
            "scale_prob": 0.7,
            "scale_range": [0.8, 1.2]
        }

    def _random_crop(self, img, bboxes, ids):
        """随机裁剪"""
        h, w = img.shape[:2]
        if random.random() < self.aug_params["crop_prob"] and len(bboxes) > 0:
            # 基于bbox裁剪，保证至少包含一个bbox
            bbox_idx = random.randint(0, len(bboxes)-1)
            crop_x1 = max(0, int(bboxes[bbox_idx][0] - random.uniform(0, 100)))
            crop_y1 = max(0, int(bboxes[bbox_idx][1] - random.uniform(0, 100)))
            crop_x2 = min(w, int(bboxes[bbox_idx][2] + random.uniform(0, 100)))
            crop_y2 = min(h, int(bboxes[bbox_idx][3] + random.uniform(0, 100)))
            
            # 裁剪图片
            img = img[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # 更新bbox
            new_bboxes = []
            new_ids = []
            for i, (x1, y1, x2, y2) in enumerate(bboxes):
                # 转换到裁剪后坐标
                nx1 = max(0, x1 - crop_x1)
                ny1 = max(0, y1 - crop_y1)
                nx2 = min(crop_x2 - crop_x1, x2 - crop_x1)
                ny2 = min(crop_y2 - crop_y1, y2 - crop_y1)
                
                # 过滤无效bbox
                if nx1 < nx2 and ny1 < ny2:
                    new_bboxes.append([nx1, ny1, nx2, ny2])
                    new_ids.append(ids[i])
            
            bboxes = np.array(new_bboxes) if new_bboxes else np.array([])
            ids = np.array(new_ids) if new_ids else np.array([])
        
        return img, bboxes, ids

    def _random_flip(self, img, bboxes, ids):
        """随机水平翻转"""
        if random.random() < self.aug_params["flip_prob"] and len(bboxes) > 0:
            img = cv2.flip(img, 1)
            h, w = img.shape[:2]
            
            # 更新bbox
            new_bboxes = []
            for x1, y1, x2, y2 in bboxes:
                nx1 = w - x2
                nx2 = w - x1
                new_bboxes.append([nx1, y1, nx2, y2])
            bboxes = np.array(new_bboxes)
        
        return img, bboxes, ids

    def _color_jitter(self, img):
        """颜色抖动：亮度、对比度、饱和度"""
        if random.random() < self.aug_params["color_jitter_prob"]:
            # 亮度
            brightness = random.uniform(0.8, 1.2)
            img = np.clip(img * brightness, 0, 255).astype(np.uint8)
            
            # 对比度
            contrast = random.uniform(0.8, 1.2)
            img = np.clip((img - 127.5) * contrast + 127.5, 0, 255).astype(np.uint8)
            
            # 饱和度
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            saturation = random.uniform(0.8, 1.2)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return img

    def _random_scale(self, img, bboxes, ids):
        """随机缩放"""
        if random.random() < self.aug_params["scale_prob"] and len(bboxes) > 0:
            scale = random.uniform(*self.aug_params["scale_range"])
            h, w = img.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            
            # 缩放图片
            img = cv2.resize(img, (new_w, new_h))
            
            # 更新bbox
            new_bboxes = []
            for x1, y1, x2, y2 in bboxes:
                nx1 = int(x1 * scale)
                ny1 = int(y1 * scale)
                nx2 = int(x2 * scale)
                ny2 = int(y2 * scale)
                new_bboxes.append([nx1, ny1, nx2, ny2])
            bboxes = np.array(new_bboxes)
        
        return img, bboxes, ids

    def __call__(self, img, bboxes, ids):
        """执行增强流程"""
        # 随机缩放
        img, bboxes, ids = self._random_scale(img, bboxes, ids)
        
        # 随机裁剪
        img, bboxes, ids = self._random_crop(img, bboxes, ids)
        
        # 随机翻转
        img, bboxes, ids = self._random_flip(img, bboxes, ids)
        
        # 颜色抖动
        img = self._color_jitter(img)
        
        return img, bboxes, ids
