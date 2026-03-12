import yaml
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os

from models import PerTrack
from datasets import build_dataloader
from utils import setup_logger, save_model, count_params

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

def train(config_path):
    """训练主函数"""
    # 加载配置
    config = load_config(config_path)
    device = torch.device(config["device"])
    
    # 设置日志
    logger = setup_logger()
    logger.info(f"Starting PerTrack training with config: {config_path}")
    
    # 构建数据集
    train_loader = build_dataloader(config, is_train=True)
    logger.info(f"Training dataset size: {len(train_loader.dataset)}")
    
    # 构建模型
    model = PerTrack(config).to(device)
    logger.info(f"Model parameters: {count_params(model)} M")
    
    # 构建优化器和损失函数
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["train"]["lr_init"],
        weight_decay=config["train"]["weight_decay"]
    )
    criterion = nn.CrossEntropyLoss()
    
    # 学习率调度器
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["train"]["epochs"],
        eta_min=config["train"]["lr_30epoch"]
    )
    
    # 训练循环
    best_loss = float("inf")
    for epoch in range(config["train"]["epochs"]):
        model.train()
        total_loss = 0.0
        
        # 进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}")
        
        for batch_idx, batch_data in enumerate(pbar):
            # 前向传播
            optimizer.zero_grad()
            
            # 处理批次数据
            imgs = torch.stack([item["img"] for item in batch_data]).to(device)
            outputs = model(imgs)
            
            # 简化版损失计算（实际使用时替换为真实损失）
            loss = torch.tensor(0.1, device=device, requires_grad=True)  # 占位符
            total_loss += loss.item()
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 更新进度条
            pbar.set_postfix({"loss": loss.item(), "avg_loss": total_loss/(batch_idx+1)})
        
        # 更新学习率
        lr_scheduler.step()
        
        # 记录日志
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        # 保存模型
        save_path = os.path.join("pretrained", f"pertrack_epoch_{epoch+1}.pth")
        save_model(model, epoch+1, save_path, optimizer, avg_loss)
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, epoch+1, os.path.join("pretrained", "pertrack_best.pth"), optimizer, avg_loss)
            logger.info(f"Best model updated with loss: {best_loss:.4f}")
    
    logger.info("Training completed!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PerTrack Training")
    parser.add_argument("--config", type=str, default="configs/mot17.yaml", help="config file path")
    args = parser.parse_args()
    
    train(args.config)
