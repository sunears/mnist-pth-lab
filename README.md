# mnist-pth-lab

本项目实现了一个简洁、可复现的 PyTorch MNIST 手写数字识别训练与评估流程，包含数据处理、模型训练、评估与实验管理脚本，便于学习与研究。

一个完整的 PyTorch MNIST 手写数字识别项目，包含规范的训练、评估和实验管理流程。

## 项目概览

本项目展示了一个生产级的 PyTorch 工作流：
- **模型**：使用带 Dropout 的简单 CNN 进行稳健分类。
- **数据**：使用 `torchvision` 自动下载和处理 MNIST 数据。
- **训练**：完整的训练循环，包含 `tqdm` 进度条、验证集记录和最佳模型保存。
- **评估**：全面的评估指标（准确率、精确率、召回率、F1 分数）、混淆矩阵以及错误样本可视化。
- **管理**：使用 `uv` 进行超快速的依赖管理和虚拟环境配置。

## 前置要求：安装 `uv`

本项目推荐使用 **uv** 进行 Python 包管理。

1.  **下载 uv**：
    -   访问 [uv releases](https://github.com/astral-sh/uv/releases) 下载 `uv-x86_64-pc-windows-msvc.zip`（适用于 Windows）。
    -   解压并将 `uv.exe` 放入本项目根目录，**或者**将其添加到系统 PATH 环境变量中。

2.  **故障排除**：
    -   如果 `run-uv.bat` 找不到 `uv`，请手动将 `uv.exe` 放在脚本同级目录下。

## 快速开始 (Windows)

我们提供了一键运行脚本 `run-uv.bat`，可自动处理环境创建和脚本执行。

### 1. 训练
训练模型 5 个 epoch（默认）并保存到 `models/mnist_cnn.pth`。

```cmd
run-uv.bat src\train.py --epochs 5 --batch-size 64 --save-path models\mnist_cnn.pth
```
*注意：首次运行会自动在 `.venv` 中创建虚拟环境并安装依赖。*

### 2. 评估
在测试集上评估训练好的模型。

```cmd
run-uv.bat src\eval.py --model models\mnist_cnn.pth --output-dir experiments\output
```
查看 `experiments\output` 目录获取评估报告和混淆矩阵。

### 3. 数据处理工具
本项目提供了两个脚本用于 MNIST 数据格式 (IDX) 与图片格式 (PNG) 之间的转换。

#### 解包 (IDX -> PNG)
将 IDX 格式的数据解包为图片文件夹和 CSV 标签。

```cmd
:: 解包训练数据
run-uv.bat src\unpack_idx.py --images-idx data\MNIST\raw\train-images-idx3-ubyte --labels-idx data\MNIST\raw\train-labels-idx1-ubyte --out-dir unpacked_data\train

:: 解包测试数据
run-uv.bat src\unpack_idx.py --images-idx data\MNIST\raw\t10k-images-idx3-ubyte --labels-idx data\MNIST\raw\t10k-labels-idx1-ubyte --out-dir unpacked_data\test
```

#### 打包 (PNG -> IDX)
将图片文件夹打包回 IDX 格式（常用于制作自己的数据集）。

```cmd
:: 打包训练数据
run-uv.bat src\pack_idx.py --images-dir unpacked_data\train\images --labels-csv unpacked_data\train\labels.csv --out-images-idx new-train-images-idx3-ubyte.gz --out-labels-idx new-train-labels-idx1-ubyte.gz

:: 打包测试数据
run-uv.bat src\pack_idx.py --images-dir unpacked_data\test\images --labels-csv unpacked_data\test\labels.csv --out-images-idx new-test-images-idx3-ubyte.gz --out-labels-idx new-test-labels-idx1-ubyte.gz
```

## 快速开始 (Linux/macOS)

我们提供了一键运行脚本 `run-uv.sh`，可自动处理环境创建和脚本执行。

### 1. 训练
训练模型 5 个 epoch（默认）并保存到 `models/mnist_cnn.pth`。

```bash
./run-uv.sh src/train.py --epochs 5 --batch-size 64 --save-path models/mnist_cnn.pth
```
*注意：首次运行会自动在 `.venv` 中创建虚拟环境并安装依赖。*

### 2. 评估
在测试集上评估训练好的模型。

```bash
./run-uv.sh src/eval.py --model models/mnist_cnn.pth --output-dir experiments/output
```
查看 `experiments/output` 目录获取评估报告和混淆矩阵。

## 手动使用方法

如果您更习惯手动运行命令，可以参考以下步骤：

1.  **创建环境并安装依赖**：
    ```bash
    # 创建虚拟环境
    uv venv .venv --python 3.10
    
    # 激活环境
    # Windows:
    .venv\Scripts\activate
    # Linux/Mac:
    source .venv/bin/activate
    
    # 安装依赖
    uv pip install -r requirements.txt
    # 或者从 pyproject.toml 安装
    uv pip install .
    ```

## 数据处理工具

本项目提供了两个脚本用于 MNIST 数据格式 (IDX) 与图片格式 (PNG) 之间的转换。

### 1. 解包 (IDX -> PNG)
将 IDX 格式的数据解包为图片文件夹和 CSV 标签。

```bash
# 解包训练数据
./run-uv.sh src/unpack_idx.py --images-idx data/MNIST/raw/train-images-idx3-ubyte --labels-idx data/MNIST/raw/train-labels-idx1-ubyte --out-dir unpacked_data/train

# 解包测试数据
./run-uv.sh src/unpack_idx.py --images-idx data/MNIST/raw/t10k-images-idx3-ubyte --labels-idx data/MNIST/raw/t10k-labels-idx1-ubyte --out-dir unpacked_data/test
```

### 2. 打包 (PNG -> IDX)
将图片文件夹打包回 IDX 格式（常用于制作自己的数据集）。

```bash
# 打包训练数据
./run-uv.sh src/pack_idx.py --images-dir unpacked_data/train/images --labels-csv unpacked_data/train/labels.csv --out-images-idx new-train-images-idx3-ubyte.gz --out-labels-idx new-train-labels-idx1-ubyte.gz

# 打包测试数据
./run-uv.sh src/pack_idx.py --images-dir unpacked_data/test/images --labels-csv unpacked_data/test/labels.csv --out-images-idx new-test-images-idx3-ubyte.gz --out-labels-idx new-test-labels-idx1-ubyte.gz
```

2.  **运行脚本**：
    ```bash
    # 训练
    python src/train.py --epochs 5 --save-path models/mnist_cnn.pth
    
    # 评估
    python src/eval.py --model models/mnist_cnn.pth
    ```

## 文件结构

```
mnist-pth-lab/
├── pyproject.toml       # 项目元数据和依赖配置
├── requirements.txt     # 依赖列表（用于兼容传统方式）
├── run-uv.bat           # Windows 下的 UV 辅助运行脚本
├── run-uv.sh            # Linux/macOS 下的 UV 辅助运行脚本
├── AGENTS.md            # AI 助手开发指南
├── CLINE.md             # CLINE 开发指南
├── gemini.md            # Gemini 开发指南
├── models/              # 存放保存的 .pth 模型
├── data/                # 下载的 MNIST 数据
├── experiments/         # 日志、图表和输出产物
└── src/
    ├── model.py         # CNN 模型架构定义
    ├── dataset.py       # DataLoaders 和数据变换
    ├── train.py         # 训练循环与日志记录
    ├── eval.py          # 评估指标与混淆矩阵
    └── utils.py         # 随机种子、日志、保存辅助函数
```

## GitHub Actions (CI/CD)

要在 GitHub 上自动化训练或测试，可以使用 `astral-sh/setup-uv` action。

`.github/workflows/test.yml` 示例片段：
```yaml
steps:
  - uses: actions/checkout@v4
  - uses: astral-sh/setup-uv@v1
  - run: uv sync
  - run: uv run python src/train.py --epochs 1  # 冒烟测试
```

## 许可证 (License)

MIT License. 详见 `LICENSE` 文件。
