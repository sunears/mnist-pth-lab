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
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets')
    os.makedirs(data_dir, exist_ok=True)
    train_data = torchvision.datasets.MNIST(root=data_dir, train=True, download=True)
    test_data = torchvision.datasets.MNIST(root=data_dir, train=False, download=True)

    image = train_data.data[0]
    label = train_data.targets[0]
    logger.info(type(train_data))
    logger.info(isinstance(train_data, torchvision.datasets.MNIST))
    logger.info(train_data.data.shape)
    logger.info(test_data.data.shape)
    logger.info("Label: %d", label.item())
    logger.info("image:\n%s", image)
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label.item()}")
    plt.show()


def print_pth():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    model_path = os.path.join(data_dir, 'mnist_cnn.pth')
    state_dict = torch.load(model_path, map_location='cpu')
    logger.info("Model state_dict:")
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            logger.info("%s: %s", key, tuple(value.shape))
        else:
            logger.info("%s: %s", key, str(type(value)))


def main():
    linear_layer_demo()


if __name__ == "__main__":
    main()
