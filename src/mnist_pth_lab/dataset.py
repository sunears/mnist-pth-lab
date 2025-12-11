import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os


def get_dataloaders(batch_size=64, num_workers=4, val_split=0.1, seed=42):
    """
    构建训练/验证/测试 DataLoader：
    - 统一归一化：MNIST 均值 0.1307、方差 0.3081
    - 固定随机种子，保证划分可复现
    - 验证集按比例从训练集划分
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 获取项目根目录（向上三级：dataset.py -> mnist_pth_lab -> src -> 项目根目录）
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # 下载/加载原始训练与测试数据
    full_train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    # 训练集划分出验证集
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=generator)

    # 构建 DataLoader（训练集随机打乱，验证/测试不打乱）
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
