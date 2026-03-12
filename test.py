import yaml
import torch
import cv2
import numpy as np
import os
from tqdm import tqdm

from models import PerTrack
from datasets import build_dataloader
from tracker import PerTrackTracker
from utils import (
    setup_logger, load_model, compute_mot_metrics,
    visualize_tracking, get_logger
)

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # 加载基础配置
    if "_base_" in config:
        base_config_path = config["_base_"]
        with open(base_config_path, "r") as f:
            base_config = yaml.safe_load(f)
        # 合并配置
        for k, v in base_config.items():
            if k not in config:
                config[k] = v
            elif isinstance(v, dict):
                config[k].update(v)
    
    return config

def test(config_path, weight_path):
    """测试主函数"""
    # 加载配置
    config = load_config(config_path)
    device = torch.device(config["device"])
    
    # 设置日志
    logger = setup_logger()
    logger.info(f"Starting PerTrack testing with config: {config_path}")
    
    # 构建数据集
    test_loader = build_dataloader(config, is_train=False)
    logger.info(f"Testing dataset size: {len(test_loader.dataset)}")
    
    # 构建跟踪器
    tracker = PerTrackTracker(config, weight_path)
    
    # 统计指标
    all_metrics = []
    seq_metrics = {}
    
    # 测试循环
    pbar = tqdm(test_loader, desc="Testing")
    current_seq = None
    seq_gt_boxes = []
    seq_gt_ids = []
    seq_pred_boxes = []
    seq_pred_ids = []
    
    for batch_data in pbar
