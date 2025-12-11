import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, dropout=0.5):
        super(SimpleCNN, self).__init__()
        # 两层 3x3 卷积 + ReLU；保持尺寸，再经过 2x2 池化减半
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Dropout 防止过拟合
        self.dropout = nn.Dropout(dropout)
        # 经过两次 2x2 池化后，28x28 -> 14x14 -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 卷积 + ReLU + 池化
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 展平为全连接输入
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def build_model(dropout=0.5):
    """工厂函数：方便在训练/评估/推理时统一构造模型"""
    return SimpleCNN(dropout=dropout)
