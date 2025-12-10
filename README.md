# mnist-pth-lab

本项目实现了一个简洁、可复现的 PyTorch MNIST 手写数字识别训练与评估流程，包含数据处理、模型训练、评估与实验管理脚本，便于学习与研究。

## 项目概览

本项目展示了一个生产级的 PyTorch 工作流：
- **模型**：使用带 Dropout 的简单 CNN 进行稳健分类。
- **数据**：使用 `torchvision` 自动下载和处理 MNIST 数据。
- **训练**：完整的训练循环，包含 `tqdm` 进度条、验证集记录和最佳模型保存。
- **评估**：全面的评估指标（准确率、精确率、召回率、F1 分数）、混淆矩阵以及错误样本可视化。
- **管理**：使用 `uv` 进行依赖管理和虚拟环境配置。

## 前置要求：安装 `uv`

本项目推荐使用 **uv** 进行 Python 包管理。

1.  **下载 uv**：
    -   访问 [uv releases](https://github.com/astral-sh/uv/releases) 下载适合平台的 `uv` 二进制。
    -   将 `uv.exe`（Windows）或 `uv`（Linux/macOS）放入本项目根目录，或将其添加到系统 PATH。

2.  **故障排除**：
    -   如果 `run-uv.bat`/`run-uv.sh` 找不到 `uv`，请确认 `uv` 可执行文件与脚本在同一目录或已在 PATH 中。

## 快速开始 (Windows)

项目提供一键运行脚本 `run-uv.bat`，可自动处理虚拟环境创建和依赖安装，并运行指定命令。

### 训练（推荐）
训练模型 5 个 epoch（默认）并保存到 `models/mnist_cnn.pth`：

```cmd
:: 推荐（模块形式，src-layout）
run-uv.bat python -m mnist_pth_lab.train --epochs 5 --batch-size 64 --save-path models\mnist_cnn.pth

:: 向后兼容：直接运行脚本路径
run-uv.bat python src\train.py --epochs 5 --batch-size 64 --save-path models\mnist_cnn.pth
```

*注意：首次运行会自动在 `.venv` 中创建虚拟环境并安装依赖。*

### 评估
在测试集上评估训练好的模型：

```cmd
run-uv.bat python -m mnist_pth_lab.eval --model models\mnist_cnn.pth --output-dir experiments\output
```

查看 `experiments\output` 获取评估报告和混淆矩阵。

### 数据处理工具（IDX <-> PNG）
将 IDX 格式数据解包为图片和 CSV 标签，或将图片打包为 IDX。

#### 解包 (IDX -> PNG)

```cmd
:: 解包训练数据（模块形式）
run-uv.bat python -m mnist_pth_lab.unpack_idx --images-idx data\MNIST\raw\train-images-idx3-ubyte --labels-idx data\MNIST\raw\train-labels-idx1-ubyte --out-dir unpacked_data\train

:: 解包测试数据
run-uv.bat python -m mnist_pth_lab.unpack_idx --images-idx data\MNIST\raw\t10k-images-idx3-ubyte --labels-idx data\MNIST\raw\t10k-labels-idx1-ubyte --out-dir unpacked_data\test
```

#### 打包 (PNG -> IDX)

```cmd
:: 打包训练数据（模块形式）
run-uv.bat python -m mnist_pth_lab.pack_idx --images-dir unpacked_data\train\images --labels-csv unpacked_data\train\labels.csv --out-images-idx new-train-images-idx3-ubyte.gz --out-labels-idx new-train-labels-idx1-ubyte.gz

:: 打包测试数据
run-uv.bat python -m mnist_pth_lab.pack_idx --images-dir unpacked_data\test\images --labels-csv unpacked_data\test\labels.csv --out-images-idx new-test-images-idx3-ubyte.gz --out-labels-idx new-test-labels-idx1-ubyte.gz
```

## 快速开始 (Linux/macOS)

项目提供 `run-uv.sh`，功能同 `run-uv.bat`：

```bash
# 推荐（模块形式）
./run-uv.sh python -m mnist_pth_lab.train --epochs 5 --batch-size 64 --save-path models/mnist_cnn.pth

# 向后兼容
./run-uv.sh python src/train.py --epochs 5 --batch-size 64 --save-path models/mnist_cnn.pth
```

评估：

```bash
./run-uv.sh python -m mnist_pth_lab.eval --model models/mnist_cnn.pth --output-dir experiments/output
```

## 手动使用方法（进阶）

如果你更习惯手动创建虚拟环境和执行命令：

```bash
# 创建虚拟环境
uv venv .venv --python 3.10

# 激活环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 安装依赖（或用于开发的可编辑安装）
uv pip install -r requirements.txt
# 或（开发者推荐）
uv pip install -e .
```

运行（开发/调试）：

```bash
# 模块形式运行（推荐）
python -m mnist_pth_lab.train --epochs 5 --save-path models/mnist_cnn.pth
python -m mnist_pth_lab.eval --model models/mnist_cnn.pth
```

## 文件结构（概要）

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
    ├── mnist_pth_lab/   # 包（src-layout）
    └── ...
```

## 文档

- `docs/` 目录包含学习笔记与英文/中文说明（`NOTES_INDEX.md` 提供索引）。

## GitHub Actions (CI/CD)

在 GitHub Actions 中可使用 `astral-sh/setup-uv`：

```yaml
steps:
  - uses: actions/checkout@v4
  - uses: astral-sh/setup-uv@v1
  - run: uv sync
  - run: uv run python -m mnist_pth_lab.train --epochs 1  # 冒烟测试
```

## 许可证 (License)

MIT License. 详见 `LICENSE` 文件。
