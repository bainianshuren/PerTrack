import logging
import os
import time

def setup_logger(log_dir="./results/logs"):
    """设置日志"""
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 日志文件名
    log_file = os.path.join(log_dir, f"pertrack_{time.strftime('%Y%m%d_%H%M%S')}.log")
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("PerTrack")

def get_logger():
    """获取日志实例"""
    return logging.getLogger("PerTrack")
