import os
import torch
import torchvision
import matplotlib.pyplot as plt
from utils import get_logger
from torch import nn
from torch import Tensor
# module-level shared logger for this script
logger = get_logger("Test")
def linear_layer_demo():
    linear = nn.Linear(3, 5)
    linear.state_dict()   # 只是触发一下参数初始化
    
    # 先打印一个空行，让输出更清爽
    logger.info("")  # 或者 logger.info("\n")
    
    # 带颜色的关键一行（绿色 + 加粗）
    logger.info(
        "\033[1;32mLinear layer state_dict:\033[0m\n%s",
        linear.state_dict()
    )
    
    # input=Tensor([[1.0, 2.0, 3.0]])
    # output=linear(input)
    # logger.info("\033[1;32mLinear layer input:\033[0m")
    # logger.info("input: \n%s", input)
    # logger.info("\033[1;32mLinear layer output:\033[0m")
    # logger.info("output: \n%s", output)
    
    # input=Tensor([
    #     [0.5, 1.5, -2.0],
    #     [3.0, -0.5, 2.0]
    # ])
    # output=linear(input)
    # logger.info("\033[1;32mLinear layer new input:\033[0m")
    # logger.info("input: \n%s", input)
    # logger.info("\033[1;32mLinear layer new output:\033[0m")
    # logger.info("output: \n%s", output)

    input=Tensor([
        [
            [1.0, 2.0, 3.0],
            [4, 5, 6],
        ],  
        [
            [11, 22, 33],
            [44, 55, 66],
        ],
    ])
    output=linear(input)
    logger.info("\033[1;32mLinear layer 3D input shape:\033[0m")
    logger.info(input.shape)
    logger.info("\033[1;32mLinear layer 3D input:\033[0m")
    logger.info("input: \n%s", input)
    logger.info("\033[1;32mLinear layer 3D output:\033[0m")
    logger.info("output: \n%s", output)
    logger.info("\033[1;32mLinear layer 3D output shape:\033[0m")
    logger.info(output.shape)
def dataset_demo():
    # 确定数据存储目录 (位于项目根目录下的 data 文件夹)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets')
    os.makedirs(data_dir, exist_ok=True)
    train_data=torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True
    )
    test_data=torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True
    )

    image = train_data.data[0]
    label = train_data.targets[0]
    logger.info(type(train_data))
    logger.info(isinstance(train_data, torchvision.datasets.MNIST))
    logger.info(train_data.data.shape)
    logger.info(test_data.data.shape)
    logger.info("Label: %d", label.item())
    logger.info("image:\n%s",image)
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label.item()}")
    plt.show()
def print_pth():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    model_path = os.path.join(data_dir, 'mnist_cnn.pth')
    # 使用 map_location 确保在 CPU 上也能加载 GPU 保存的模型
    state_dict = torch.load(model_path, map_location='cpu')
    logger.info("Model state_dict:")
    for key, value in state_dict.items():
        # 安全处理：可能 value 不是 Tensor（例如嵌套 dict/列表）
        if isinstance(value, torch.Tensor):
            logger.info("%s: %s", key, tuple(value.shape))
            logger.info("types: %s, %s", type(key).__name__, type(value).__name__)
        else:
            logger.info("%s: %s", key, str(type(value)))

    # 如果存在 conv1.weight 并且是 Tensor，则打印前 10 个元素作为示例
    sample_key = 'conv1.weight'
    if sample_key in state_dict and isinstance(state_dict[sample_key], torch.Tensor):
        tensor = state_dict[sample_key].cpu().view(-1)
        sample = tensor[:10].tolist()
        logger.info("example weight tensor sample (first 10): %s", sample)
    else:
        logger.info("No '%s' tensor found to sample.", sample_key)
def print_pth2():
    # 假设您的 .pth 文件路径
    file_path = 'models/mnist_cnn.pth' 

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件未找到在路径: {file_path}")
    else:
        # 使用 torch.load 加载文件
        # map_location='cpu' 确保即使文件是在 GPU 上保存的，也能在 CPU 上加载
        try:
            data = torch.load(file_path, map_location='cpu')
            print(f"成功加载文件：{file_path}")
            target_dict = None
            if isinstance(data, dict):
                print("\n--- 文件是一个字典/状态字典 ---")
                
                if 'state_dict' in data:
                    target_dict = data['state_dict']
                    print("状态字典位于 'state_dict' 键下。")
                else:
                    target_dict = data
                    print("文件直接是状态字典。")
            
            # ... (处理 Tensor, list, tuple 的逻辑不变)
            
            # --- 通用遍历并打印 (修复了格式化错误) ---
            if target_dict is not None and isinstance(target_dict, dict):
                print("\n--- 模型权重/参数列表 ---")
                
                for key, content in target_dict.items():
                    if isinstance(content, torch.Tensor):
                        # *** 最终修复点 ***
                        # 1. 将形状元组转换为字符串
                        shape_str = str(tuple(content.shape)) 
                        # 2. 对字符串应用对齐格式
                        print(f"键: {key:<40} | 形状: {shape_str:<20} | 数据类型: {content.dtype}")
                    else:
                        # 非 Tensor 内容 (已修复)
                        content_str = str(content)
                        print(f"键: {key:<40} | 内容: {content_str}")
        except Exception as e:
            print(f"加载文件时发生错误：{e}")
def main():
    linear_layer_demo()
    # print_pth()
if __name__ == "__main__":
    main()