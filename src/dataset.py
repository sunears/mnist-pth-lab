import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

def get_dataloaders(batch_size=64, num_workers=4, val_split=0.1, seed=42):
    """
    下载并准备 MNIST 数据集，构建 DataLoader。
    
    Args:
        batch_size (int): 每个批次使用的样本数量。
        num_workers (int): 用于数据加载的子进程数量。
        val_split (float): 验证集占训练集的比例 (0.0 ~ 1.0)。
        seed (int): 用于随机分割数据集的种子。
        
    Returns:
        tuple: 包含三个 DataLoader 对象 (train_loader, val_loader, test_loader)。
    """
    # 定义数据预处理流程
    # ToTensor: 将 PIL Image 或 numpy.ndarray 转换为 tensor (0-1 之间, 且将通道移到最前)
    # Normalize: 使用 MNIST 数据集的均值 (0.1307) 和标准差 (0.3081) 进行标准化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 确定数据存储目录 (位于项目根目录下的 data 文件夹)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)

    # 下载并加载训练数据 (Train=True)
    full_train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    # 下载并加载测试数据 (Train=False)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    # 将训练集划分为训练集和验证集
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    
    # 使用指定的种子生成器进行随机分割，保证可复现性
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=generator)

    # 创建 DataLoader
    # train_loader 开启 shuffle 用于训练过程打乱数据
    # val_loader 和 test_loader 通常不需要 shuffle
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
