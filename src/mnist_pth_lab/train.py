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

from mnist_pth_lab.model import build_model
from mnist_pth_lab.dataset import get_dataloaders
from mnist_pth_lab.utils import set_seed, save_model, get_logger


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logger = get_logger("Train")
    logger.info(f"Using device: {device}")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs("experiments", exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders(args.batch_size, args.num_workers, seed=args.seed)

    model = build_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix({'loss': loss.item()})

        train_loss /= total
        train_acc = correct / total

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
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

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, args.save_path)
            logger.info(f"新最佳模型已保存，验证集准确率: {val_acc:.4f}")

    total_time = time.time() - start_time
    logger.info(f"训练完成，总耗时 {total_time:.2f}s。最佳验证集准确率: {best_val_acc:.4f}")

    with open('experiments/config.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    with open('experiments/training_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
        for i in range(args.epochs):
            writer.writerow([i+1, history['train_loss'][i], history['train_acc'][i], history['val_loss'][i], history['val_acc'][i]])

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.legend()
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
