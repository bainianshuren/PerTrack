import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml

class MOTDataset(Dataset):
    """MOT数据集加载类（支持MOT17/MOT20）
    """
    def __init__(self, config, is_train=True):
        self.config = config
        self.is_train = is_train
        self.data_path = config["data"]["data_path"]
        self.set_name = config["data"]["train_set"] if is_train else config["data"]["test_set"]
        self.seq_list = config["data"]["seq_list"]
        
        # 加载所有样本路径和标注
        self.samples = self._load_samples()
        
        # 数据增强
        if is_train and config["train"]["data_aug"]:
            from .data_augment import MOTDataAugment
            self.augmentor = MOTDataAugment(config)
        else:
            self.augmentor = None

    def _load_samples(self):
        """加载数据集样本（图片路径+标注）"""
        samples = []
        for seq in self.seq_list:
            seq_path = os.path.join(self.data_path, self.set_name, seq)
            img_path = os.path.join(seq_path, "img1")
            label_path = os.path.join(seq_path, "gt", "gt.txt")
            
            # 读取标注文件
            if os.path.exists(label_path):
                labels = np.loadtxt(label_path, delimiter=",")
                # 按帧分组标注
                frame_ids = np.unique(labels[:, 0]).astype(int)
                for frame_id in frame_ids:
                    frame_labels = labels[labels[:, 0] == frame_id]
                    img_file = f"{frame_id:06d}.jpg"
                    img_full_path = os.path.join(img_path, img_file)
                    if os.path.exists(img_full_path):
                        samples.append({
                            "img_path": img_full_path,
                            "seq_name": seq,
                            "frame_id": frame_id,
                            "labels": frame_labels
                        })
        return samples

    def _parse_labels(self, labels):
        """解析标注：[frame, id, x1, y1, w, h, conf, class, visibility]"""
        bboxes = []
        ids = []
        for label in labels:
            # 过滤非行人（class=1）和忽略标注（conf=-1）
            if label[7] == 1 and label[6] != -1:
                x1, y1, w, h = label[2:6]
                # 转换为绝对坐标
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = int(x1 + w)
                y2 = int(y1 + h)
                bboxes.append([x1, y1, x2, y2])
                ids.append(int(label[1]))
        return np.array(bboxes), np.array(ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 读取图片
        img = cv2.imread(sample["img_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 解析标注
        bboxes, ids = self._parse_labels(sample["labels"])
        
        # 数据增强
        if self.augmentor is not None:
            img, bboxes, ids = self.augmentor(img, bboxes, ids)
        
        # 格式转换
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        bboxes = torch.from_numpy(bboxes).float()
        ids = torch.from_numpy(ids).long()
        
        return {
            "img": img,
            "bboxes": bboxes,
            "ids": ids,
            "img_path": sample["img_path"],
            "seq_name": sample["seq_name"],
            "frame_id": sample["frame_id"]
        }

def build_dataloader(config, is_train=True):
    """构建数据加载器"""
    dataset = MOTDataset(config, is_train)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"] if is_train else 1,
        shuffle=is_train,
        num_workers=config["num_workers"],
        collate_fn=lambda x: x  # 自定义collate_fn处理变长标注
    )
    return dataloader
