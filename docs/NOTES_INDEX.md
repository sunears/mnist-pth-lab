# Notes Index / 学习笔记索引

下面列出已整理到 `docs/notes/` 的学习笔记的映射、中文摘要与英文摘要：

- `docs/notes/pytorch_core_four_params.md`  
  - 中文摘要：讲解 PyTorch 训练中最核心的四个参数（batch_size、num_workers、val_split、seed），包含实践建议与 Windows 注意事项。  
  - English summary: Explains the four core PyTorch training parameters (batch_size, num_workers, val_split, seed) with practical recommendations and Windows-specific tips.  
  - 原始文件：`./docs/notes/originals/PyTorch 训练核心四参数.md`

- `docs/notes/visualize_mnist_plt.md`  
  - 中文摘要：展示如何用 matplotlib 可视化 MNIST 图像，给出最佳实践（squeeze、numpy 转换、隐藏坐标轴等）。  
  - English summary: Shows how to visualize MNIST images with matplotlib, including best practices (squeeze, convert to numpy, hide axes).  
  - 原始文件：`./docs/notes/originals/plt.md`

- `docs/notes/linear_forward_calc.md`  
  - 中文摘要：逐步手算并解释 `nn.Linear` 前向计算，展示权重、bias 与输入如何点乘得到输出的详细过程。  
  - English summary: Step-by-step manual calculation of nn.Linear forward pass, showing how weights, bias and inputs produce outputs.  
  - 原始文件：`./docs/notes/originals/linear计算过程.md`

- `docs/notes/loss_accuracy_curve.md`  
  - 中文摘要：解读训练过程中的损失与准确率曲线，分阶段分析训练行为并给出优化建议。  
  - English summary: Interprets training loss and accuracy curves, analyzes training phases and offers improvement suggestions.  
  - 原始文件：`./docs/notes/originals/LossAccuracyCurve.md`

- `docs/notes/epoch_explained.md`  
  - 中文摘要：用通俗比喻明确定义 epoch 的含义、常见误区和合适的 epoch 数量选择。  
  - English summary: Plain-language explanation of what an epoch is, common pitfalls, and guidance on choosing the number of epochs.  
  - 原始文件：`./docs/notes/originals/epoch.md`

- `docs/notes/dataset_and_tuple.md`  
  - 中文摘要：阐明 `torchvision.datasets.MNIST` 的真实类型与用法，解释 `train_data[i]` 返回 tuple（image, label）的语义与操作建议。  
  - English summary: Clarifies the torchvision.datasets.MNIST type and usage; explains that train_data[i] returns a tuple (image, label) and how to handle it.  
  - 原始文件：`./docs/notes/originals/dataset和truple.md`

- `docs/notes/targets_and_data.md`  
  - 中文摘要：深入说明 MNIST 中 `data` 与 `targets` 的结构、像素范围、如何转换为网络输入（float/unsqueeze/normalize）。  
  - English summary: Details MNIST `data` and `targets` structure, pixel ranges, and how to convert them into network-ready inputs (float, unsqueeze, normalize).  
  - 原始文件：`./docs/notes/originals/targets 和data.md`

- `docs/notes/run_with_uv.md`  
  - 中文摘要：详解 `run_with_uv.bat` 的工作流程、uv 的用法与项目环境隔离、依赖锁定与离线运行能力。  
  - English summary: Deep dive into `run_with_uv.bat`: how it uses uv to create isolated environments, lock dependencies, and support offline execution.  
  - 原始文件：`./docs/notes/originals/run_with_uv.md`

注意：`CLINE.md` 与 `gemini.md` 被保留在仓库根目录并不移动（它们被 `cline` 插件与 `gemini` CLI 使用）。

---

如果你确认这些文件和摘要无误，我可以：

- 1) 把这些新文件提交到 git 并推送（提交信息建议：`docs: 整理学习笔记到 docs/notes/ 并生成索引`），或
- 2) 仅保留工作区更改，由你手动检查并提交。请告诉我你要我执行哪项。
