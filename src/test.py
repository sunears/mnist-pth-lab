import os
import torchvision
import matplotlib.pyplot as plt
from utils import get_logger
from torch import nn
from torch import Tensor
def linear_layer_demo():
    linear = nn.Linear(3, 5)
    linear.state_dict()   # 只是触发一下参数初始化
    
    logger = get_logger("Test")
    
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
def main():
    # linear_layer_demo()
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
    logger = get_logger("Test")
    logger.info(train_data.data.shape)
    logger.info(test_data.data.shape)
    logger.info("Label: %d", label.item())
    logger.info("\n%s",image)
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label.item()}")
    plt.show()
if __name__ == "__main__":
    main()