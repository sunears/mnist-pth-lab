"""
模块说明（中文）：
本模块实现一个用于演示的后台训练器（`training_demo`）。
功能：在独立线程中执行简化的 MNIST 训练循环，并通过线程安全的全局状态暴露实时进度和日志。
用途：用于本项目的 Web 前端（`/training_demo` 页面）轮询获取训练日志、Loss/Accuracy 曲线数据和训练进度。
运行方式：在 Flask 中通过 POST 调用 `/training_demo/start` 启动训练，GET `/training_demo/progress` 获取进度。
注意：该演示以可视化为目的，默认使用小批量和短 epochs，可能不会与完整训练完全一致。
"""

import threading
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
from collections import deque

from mnist_pth_lab.model import build_model
from mnist_pth_lab.dataset import get_dataloaders
from mnist_pth_lab.utils import set_seed, save_model, get_logger


# Global state for training progress
_training_state = {
    'is_running': False,
    'logs': deque(maxlen=500),  # Keep last 500 log lines
    'current_epoch': 0,
    'total_epochs': 0,
    'current_batch': 0,
    'total_batches': 0,
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
}
_training_lock = threading.Lock()


def _log(msg):
    """Add message to training log"""
    with _training_lock:
        _training_state['logs'].append(msg)


def get_progress():
    """Get current training progress"""
    with _training_lock:
        return {
            'is_running': _training_state['is_running'],
            'logs': list(_training_state['logs']),
            'current_epoch': _training_state['current_epoch'],
            'total_epochs': _training_state['total_epochs'],
            'current_batch': _training_state['current_batch'],
            'total_batches': _training_state['total_batches'],
            'train_loss': _training_state['train_loss'],
            'train_acc': _training_state['train_acc'],
            'val_loss': _training_state['val_loss'],
            'val_acc': _training_state['val_acc'],
        }


def _training_worker(epochs=3, batch_size=64, lr=0.001, device='cpu'):
    """Background worker to run training"""
    try:
        with _training_lock:
            _training_state['is_running'] = True
            _training_state['total_epochs'] = epochs

        _log("[INFO] 初始化模型...")
        set_seed(42)
        model = build_model().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        _log("[INFO] 加载训练数据...")
        train_loader, val_loader, _ = get_dataloaders(batch_size, num_workers=0, seed=42)
        total_batches = len(train_loader)
        _training_state['total_batches'] = total_batches

        _log(f"[INFO] 模型已初始化 | 优化器: Adam | 学习率: {lr}")
        _log(f"[INFO] 数据集加载完毕 | Train batches: {total_batches} | Batch size: {batch_size}")
        time.sleep(0.5)

        for epoch in range(1, epochs + 1):
            with _training_lock:
                _training_state['current_epoch'] = epoch

            _log(f"\n{'='*60}")
            _log(f"[EPOCH {epoch}/{epochs}] 开始前向传播与梯度更新...")
            _log(f"{'='*60}")

            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                with _training_lock:
                    _training_state['current_batch'] = batch_idx + 1

                images, labels = images.to(device), labels.to(device)

                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, labels)

                # 反向传播 + 梯度更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Log progress for demo
                if (batch_idx + 1) % max(1, len(train_loader) // 3) == 0 or batch_idx == 0:
                    batch_acc = 100.0 * correct / total
                    batch_loss = loss.item()
                    _log(f"  [Batch {batch_idx+1}/{total_batches}] Loss: {batch_loss:.4f} | Acc: {batch_acc:.2f}%")
                    time.sleep(0.1)  # Brief pause for UI update

            train_loss /= total
            train_acc = 100.0 * correct / total

            with _training_lock:
                _training_state['train_loss'].append(train_loss)
                _training_state['train_acc'].append(train_acc)

            _log(f"[Train] Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
            _log(f"[INFO] 验证集评估中...")

            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            val_loss /= total
            val_acc = 100.0 * correct / total

            with _training_lock:
                _training_state['val_loss'].append(val_loss)
                _training_state['val_acc'].append(val_acc)

            _log(f"[Val]   Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
            time.sleep(0.5)

        _log(f"\n{'='*60}")
        _log(f"[SUCCESS] 训练完成！")
        _log(f"  最佳训练集准确率: {max(_training_state['train_acc']):.2f}%")
        _log(f"  最佳验证集准确率: {max(_training_state['val_acc']):.2f}%")
        _log(f"{'='*60}")

    except Exception as e:
        _log(f"[ERROR] 训练异常: {str(e)}")
        import traceback
        _log(traceback.format_exc())

    finally:
        with _training_lock:
            _training_state['is_running'] = False


def start_training(epochs=3, batch_size=64, lr=0.001, device='cpu'):
    """Start training in background thread"""
    with _training_lock:
        if _training_state['is_running']:
            return {'error': 'Training already in progress'}
        # Reset state
        _training_state['logs'].clear()
        _training_state['current_epoch'] = 0
        _training_state['current_batch'] = 0
        _training_state['train_loss'] = []
        _training_state['train_acc'] = []
        _training_state['val_loss'] = []
        _training_state['val_acc'] = []

    thread = threading.Thread(target=_training_worker, args=(epochs, batch_size, lr, device), daemon=True)
    thread.start()
    return {'status': 'Training started'}


def reset_training():
    """Reset training state"""
    with _training_lock:
        _training_state['logs'].clear()
        _training_state['current_epoch'] = 0
        _training_state['current_batch'] = 0
        _training_state['total_epochs'] = 0
        _training_state['total_batches'] = 0
        _training_state['train_loss'] = []
        _training_state['train_acc'] = []
        _training_state['val_loss'] = []
        _training_state['val_acc'] = []
