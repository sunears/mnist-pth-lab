import os
import torch
import torchvision
import matplotlib.pyplot as plt
from mnist_pth_lab.utils import get_logger
from torch import nn
from torch import Tensor

logger = get_logger("Test")


def linear_layer_demo():
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
    # 获取项目根目录（向上三级：test.py -> mnist_pth_lab -> src -> 项目根目录）
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(project_root, 'datasets')
    os.makedirs(data_dir, exist_ok=True)
    train_data = torchvision.datasets.MNIST(root=data_dir, train=True, download=True)
    test_data = torchvision.datasets.MNIST(root=data_dir, train=False, download=True)

    image = test_data.data[0]
    label = test_data.targets[0]
    logger.info("type(test_data): %s", type(test_data))
    logger.info("isinstance(test_data, torchvision.datasets.MNIST): %s", isinstance(test_data, torchvision.datasets.MNIST))
    logger.info("test_data.data.shape: %s", test_data.data.shape)
    flat_test_data = test_data.data.view(10000,784)
    logger.info("flat_test_data.shape: %s", flat_test_data.shape)
    # logger.info("label of test_data.targets[0]: %s", test_data.targets[0])
    # logger.info("image of test_data.data[0]:\n%s", image)
    float_flat_test_data = flat_test_data.float() / 255.0
    logger.info("float_flat_test_data.shape: %s", float_flat_test_data.shape)
    logger.info("\nfloat_flat_test_data.data[1]: %s", float_flat_test_data.data[1])
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
    # print_pth()
    dataset_demo()
if __name__ == "__main__":
    main()
