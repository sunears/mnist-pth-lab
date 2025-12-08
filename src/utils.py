import torch
import random
import numpy as np
import logging
import os


def set_seed(seed=42):
    """
    设置随机种子以确保结果可复现。
    
    Args:
        seed (int): 随机种子数值，默认为 42。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 保证使用确定性算法，可能会降低性能
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Info] 随机种子已设置为 {seed}")

def save_model(model, path):
    """
    保存模型的状态字典 (state dict)。
    
    Args:
        model (torch.nn.Module): 要保存的模型实例。
        path (str): 模型保存路径 (例如 'models/model.pth')。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[Info] 模型已保存至 {path}")

def load_model(model_instance, path, device):
    """
    将保存的状态字典加载到现有的模型实例中。
    
    Args:
        model_instance (torch.nn.Module): 初始化好的模型架构实例。
        path (str): 模型权重文件的路径。
        device (torch.device): 加载模型的目标设备 (CPU 或 CUDA)。
        
    Returns:
        model_instance (torch.nn.Module): 加载了权重的模型实例。
    """
    model_instance.load_state_dict(torch.load(path, map_location=device))
    model_instance.to(device)
    print(f"[Info] 模型已从 {path} 加载")
    return model_instance

def get_logger(name, level=logging.INFO):
    """
    创建一个简单的日志记录器配置。
    
    Args:
        name (str): 日志记录器的名称。
        level (int): 日志级别，默认为 logging.INFO。
        
    Returns:
        logger (logging.Logger): 配置好的日志记录器。
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

