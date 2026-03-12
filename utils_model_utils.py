import torch
import os

def save_model(model, epoch, save_path, optimizer=None, loss=None):
    """保存模型"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "loss": loss
    }
    
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")

def load_model(model, weight_path, device="cuda"):
    """加载模型"""
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded weights from {weight_path}")
    return model, checkpoint.get("epoch", 0)

def count_params(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params / 1e6,  # 转换为M
        "trainable": trainable_params / 1e6
    }
