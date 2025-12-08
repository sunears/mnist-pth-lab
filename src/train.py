import argparse
import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

from model import build_model
from dataset import get_dataloaders
from utils import set_seed, save_model, get_logger

def train(args):
    """
    执行完整的训练流程。
    
    包含:
    - 环境设置 (Seed, Device)
    - 数据加载
    - 模型初始化
    - 训练循环 (Training Loop)
    - 为了避免过拟合的验证循环 (Validation Loop)
    - 结果记录与绘图
    """
    # 1. 环境设置
    set_seed(args.seed)
    # 如果可用且指定了 cuda，则使用 GPU，否则使用 CPU
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logger = get_logger("Train")
    logger.info(f"Using device: {device}") # 记录使用的设备
    
    # 2. 目录准备
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True) # 确保模型保存目录存在
    os.makedirs("experiments", exist_ok=True) # 确保实验结果目录存在

    # 3. 数据准备
    # 获取训练集、验证集 DataLoader
    train_loader, val_loader, _ = get_dataloaders(args.batch_size, args.num_workers, seed=args.seed)
    
    # 4. 模型与优化器
    model = build_model().to(device) # 构建模型并移动到指定设备
    optimizer = optim.Adam(model.parameters(), lr=args.lr) # 使用 Adam 优化器
    criterion = nn.CrossEntropyLoss() # 使用交叉熵损失函数 (适用于多分类)
    
    # 5. 日志记录初始化
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0 # 用于记录最佳验证集准确率
    
    # 6. 训练循环
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # --- 训练阶段 (TRAIN) ---
        model.train() # 切换到训练模式 (启用 Dropout, BatchNorm 等)
        train_loss = 0.0
        correct = 0
        total = 0
        
        # 使用 tqdm 显示进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device) # 数据移动到设备
            
            optimizer.zero_grad() # 清空梯度
            outputs = model(images) # 前向传播
            loss = criterion(outputs, labels) # 计算损失
            loss.backward() # 反向传播
            optimizer.step() # 更新参数
            
            # 统计指标
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1) # 获取预测类别
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条显示的当前 loss
            pbar.set_postfix({'loss': loss.item()})
            
        train_loss /= total # 计算平均 loss
        train_acc = correct / total # 计算准确率
        
        # --- 验证阶段 (VALIDATION) ---
        model.eval() # 切换到评估模式 (关闭 Dropout, BatchNorm 等)
        val_loss = 0.0
        correct = 0
        total = 0
        
        # 禁用梯度计算，节省显存并加速
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        val_loss /= total
        val_acc = correct / total
        
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f} | Time={epoch_time:.2f}s")
        
        # 保存历史记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 保存最佳模型 (Best Model Checkpoint)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, args.save_path)
            logger.info(f"新最佳模型已保存，验证集准确率: {val_acc:.4f}")

    total_time = time.time() - start_time
    logger.info(f"训练完成，总耗时 {total_time:.2f}s。最佳验证集准确率: {best_val_acc:.4f}")

    # 7. 保存实验配置
    with open('experiments/config.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    # 8. 保存训练日志 (CSV)
    with open('experiments/training_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
        for i in range(args.epochs):
            writer.writerow([i+1, history['train_loss'][i], history['train_acc'][i], history['val_loss'][i], history['val_acc'][i]])

    # 9. 绘制训练曲线
    plt.figure(figsize=(10, 5))
    # Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.legend()
    
    # Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy Curve')
    plt.legend()
    
    plt.savefig('experiments/training_curves.png')
    logger.info("训练曲线已保存至 experiments/training_curves.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="在 MNIST 数据集上训练 CNN 模型")
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数/Epochs (数据集完整遍历次数)')
    parser.add_argument('--batch-size', type=int, default=64, help='批大小/Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率/Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='训练设备 (cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子/Random seed')
    parser.add_argument('--save-path', type=str, default='models/mnist_cnn.pth', help='模型保存路径')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader 工作进程数 (Windows 建议为 0)')
    
    args = parser.parse_args()
    train(args)
