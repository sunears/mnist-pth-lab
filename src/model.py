import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    一个简单的卷积神经网络 (CNN)，用于 MNIST 数字分类。
    
    结构:
    - 2 个卷积层 (Convolutional Layers)
    - 2 个最大池化层 (Max Pooling Layers)
    - Dropout 层 (防止过拟合)
    - 2 个全连接层 (Fully Connected Layers)
    """
    def __init__(self, dropout=0.5):
        super(SimpleCNN, self).__init__()
        # 卷积层 1: 输入通道 1 (灰度图), 输出通道 32, 卷积核 3x3, 填充 1
        # padding=1 保证了卷积后尺寸不变 (28x28)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        
        # 卷积层 2: 输入通道 32, 输出通道 64, 卷积核 3x3, 填充 1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # 最大池化层: 窗口大小 2x2, 步长 2
        # 每次经过池化，特征图的长宽减半
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout 层: 随机丢弃 50% 的神经元，防止过拟合
        self.dropout = nn.Dropout(dropout)
        
        # 全连接层 (Fully Connected Layers)
        # 尺寸计算:
        # 原始输入: 28x28
        # 经过 conv1 (padding=1) -> 28x28 -> pool -> 14x14
        # 经过 conv2 (padding=1) -> 14x14 -> pool -> 7x7
        # 最终特征图大小: 64 个通道 * 7 * 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128) # 展平后连接到 128 个神经元
        self.fc2 = nn.Linear(128, 10)         # 输出层: 10 个类别 (0-9)

    def forward(self, x):
        """
        定义前向传播路径。
        
        Args:
            x (torch.Tensor): 输入图像 batch。
            
        Returns:
            x (torch.Tensor): 模型的输出 logits。
        """
        # 第一层卷积块: Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x)))
        
        # 第二层卷积块: Conv2 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))
        
        # 展平 (Flatten): 将多维特征图展平为一维向量，以便输入全连接层
        # -1 表示自动推断 batch_size 维度
        x = x.view(-1, 64 * 7 * 7)
        
        # 第一层全连接: FC1 -> ReLU -> Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # 输出层 (FC2)
        # 注意: 这里不加 Softmax，因为 CrossEntropyLoss 会在内部进行 LogSoftmax
        x = self.fc2(x)
        return x

def build_model(dropout=0.5):
    """
    构建模型的工厂函数。
    
    Args:
        dropout (float): Dropout 概率。
        
    Returns:
        model (SimpleCNN): 初始化的模型实例。
    """
    return SimpleCNN(dropout=dropout)
