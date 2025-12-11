import os
import torch
import torchvision
import matplotlib.pyplot as plt
from mnist_pth_lab.utils import get_logger, load_model
from mnist_pth_lab.model import build_model
from torch import nn
from torch import Tensor

logger = get_logger("Test")


def linear_layer_demo():
    """演示一个线性层的权重/偏置形状及前向输出"""
    linear = nn.Linear(3, 5)
    linear.state_dict()
    logger.info("")
    logger.info(
        "\033[1;32mLinear layer state_dict:\033[0m\n%s",
        linear.state_dict()
    )
    input = Tensor([
        [1.0, 2.0, 3.0],
    ])
    output = linear(input)
    logger.info("\033[1;32mLinear layer 3D output:\033[0m")
    logger.info(output)


def dataset_demo():
    """下载/演示 MNIST 测试集，并用单个样本做前向推理"""
    # 获取项目根目录（向上三级：test.py -> mnist_pth_lab -> src -> 项目根目录）
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(project_root, 'datasets')
    os.makedirs(data_dir, exist_ok=True)
    train_data = torchvision.datasets.MNIST(root=data_dir, train=True, download=True)
    test_data = torchvision.datasets.MNIST(root=data_dir, train=False, download=True)

    image = test_data.data[0]
    label = test_data.targets[88]
    logger.info("type(test_data): %s", type(test_data))
    logger.info("isinstance(test_data, torchvision.datasets.MNIST): %s", isinstance(test_data, torchvision.datasets.MNIST))
    logger.info("test_data.data.shape: %s", test_data.data.shape)
    # 将 (N,28,28) 展平成 (N,784)，方便做向量化演示
    flat_test_data = test_data.data.view(10000,784)
    logger.info("flat_test_data.shape: %s", flat_test_data.shape)
    logger.info("label of test_data.targets[88]: %s", label)
    # logger.info("image of test_data.data[0]:\n%s", image)
    # 简单归一化到 [0,1]
    float_flat_test_data = flat_test_data.float() / 255.0
    logger.info("float_flat_test_data.shape: %s", float_flat_test_data.shape)
    # logger.info("\nfloat_flat_test_data.data[1]: %s", float_flat_test_data.data[1])
    input_tensor = float_flat_test_data[88]
    logger.info("input_tensor.shape: %s", input_tensor.shape)
    logger.info("\ninput_tensor: %s", input_tensor)
    
    # 加载模型并进行推理
    model_path = os.path.join(project_root, 'models', 'mnist_cnn.pth')
    
    if os.path.exists(model_path):
        logger.info("\n[Info] 加载模型: %s", model_path)
        device = torch.device('cpu')
        model = build_model()
        model = load_model(model, model_path, device)
        model.eval()
        
        # 将 input_tensor 从 (784,) reshape 成 (1, 1, 28, 28) 以匹配模型输入
        input_batch = input_tensor.view(1, 1, 28, 28)
        logger.info("input_batch.shape (模型输入): %s", input_batch.shape)
        
        with torch.no_grad():
            output = model(input_batch)
            logger.info("output.shape (模型输出): %s", output.shape)
            logger.info("output (原始 logits):\n%s", output)
            
            # 获取预测类别
            _, predicted = output.max(1)
            logger.info("预测类别 (predicted class): %d", predicted.item())
            
            # 计算 softmax 概率
            probabilities = torch.nn.functional.softmax(output, dim=1)
            logger.info("概率分布 (probabilities):\n%s", probabilities)
            logger.info("最大概率值: %.4f (类别 %d)", probabilities.max().item(), predicted.item())
    else:
        logger.warning("模型文件不存在: %s", model_path)
    
    # plt.imshow(image, cmap='gray')
    # plt.title(f"Label: {label.item()}")
    # plt.show()


def print_pth():
    # 获取项目根目录（向上三级：test.py -> mnist_pth_lab -> src -> 项目根目录）
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model_path = os.path.join(project_root, 'models', 'mnist_cnn.pth')
    state_dict = torch.load(model_path, map_location='cpu')
    logger.info("Model state_dict:")
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            logger.info("%s: %s", key, tuple(value.shape))
        else:
            logger.info("%s: %s", key, str(type(value)))


def main():
    # linear_layer_demo()
    print_pth()
    # dataset_demo()
if __name__ == "__main__":
    main()
