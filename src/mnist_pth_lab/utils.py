import torch
import torch.nn as nn
import random
import numpy as np
import logging
import os


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Info] 随机种子已设置为 {seed}")


def save_model(model: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[Info] 模型已保存至 {path}")


def load_model(model_instance: nn.Module, path: str, device: torch.device) -> nn.Module:
    model_instance.load_state_dict(torch.load(path, map_location=device))
    model_instance.to(device)
    print(f"[Info] 模型已从 {path} 加载")
    return model_instance


def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger
