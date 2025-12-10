````markdown
```python
def main():
    # linear_layer_demo()
    # 确定数据存储目录 (位于项目根目录下的 data 文件夹)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets')
    os.makedirs(data_dir, exist_ok=True)
    train_data=torchvision.datasets.MNIST(
        root="./datasets",
        train=True,
        download=True
    )
    test_data=torchvision.datasets.MNIST(
        root="./datasets",
        train=False,
        download=True
    )
    logger = get_logger("Test")
    logger.info(train_data.data.shape)
    logger.info(test_data.data.shape)
    logger.info(test_data.targets[0])
    logger.info(train_data.data[0])
```

太棒了！你已经成功下载并看到了真实的 MNIST 数据！  
现在我们把你看到的这堆信息，**用最清晰、最实战的方式彻底讲透**，让你 3 分钟彻底掌握 MNIST 数据集！

### 你现在看到的到底是什么？

```text
train_data.data.shape → torch.Size([60000, 28, 28])
test_data.data.shape  → torch.Size([10000, 28, 28])
test_data.targets[0]  → tensor(7)
train_data.data[0]    → 一张 28×28 的手写数字「7」
```

...（略）

金句三连（背会你就能教别人了）：
1. “MNIST 没有颜色通道，只有 [N, 28, 28]”
2. “喂给网络前一定要 unsqueeze(1) 变成 [N, 1, 28, 28]”
3. “像素要除以 255 变成 0~1 之间的浮点数”
````
