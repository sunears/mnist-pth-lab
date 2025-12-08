# 项目分析说明 (Project Analysis)

## 核心功能

本项目是一个基于 PyTorch 的 MNIST 手写数字识别实验室 (`mnist-pth-lab`)。

经过对项目代码的全面分析，**当前版本并未包含任何与 Gemini 模型相关的集成或功能**。

该项目的主要功能包括：
*   **模型训练**: 使用 CNN 模型进行 MNIST 数据集训练 (`src/train.py`)。
*   **模型评估**: 对训练好的模型进行性能评估，并生成报告和混淆矩阵 (`src/eval.py`)。
*   **数据工具**: 提供将 MNIST 的 IDX 文件格式与 PNG 图片格式互相转换的工具 (`src/pack_idx.py`, `src/unpack_idx.py`)。
*   **环境管理**: 使用 `uv` 进行快速的 Python 环境和依赖管理。

更多详细信息，请参阅 `README.md` 文件。

## 注意事项

**优先中文回答**
