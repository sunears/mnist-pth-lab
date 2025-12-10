# Release Notes

Release: v0.3.0
Date: 2025-12-10

Highlights
---------
- 新增“训练演示”网页：可实时查看训练日志、训练/验证 Loss 与 Accuracy 曲线、训练进度条（文件：`src/mnist_pth_lab/training_demo.py`、`templates/training_demo.html`）。
- 在 `webapp.py` 中集成训练演示路由：`/training_demo`、`/training_demo/start`、`/training_demo/progress` 与 `/training_demo/reset`，便于在浏览器中触发与观察训练过程。
- 完善并更新文档：在 `README.md` 与 `README_en.md` 中增加“笔记与可视化”节，提供运行示例与页面链接；新增学习笔记 `docs/notes/originals/生命循环.md`（训练“生命循环”详解）。
- 为若干演示与模块文件添加中文摘要注释，提升可读性（`training_demo.py`、`webapp.py`、`templates/training_demo.html`、`docs/notes/originals/生命循环.md`）。
- 初始的 Web 应用（手写预测 + 卷积核可视化）已集成并发布（文件：`src/mnist_pth_lab/webapp.py`、`templates/index.html`、`static/vis.js`、`static/styles.css`）。

Files Changed / Added
---------------------
- src/mnist_pth_lab/training_demo.py (new)
- src/mnist_pth_lab/templates/training_demo.html (new)
- src/mnist_pth_lab/webapp.py (updated)
- src/mnist_pth_lab/templates/index.html (existing)
- src/mnist_pth_lab/static/vis.js (existing)
- src/mnist_pth_lab/static/styles.css (existing)
- README.md (updated)
- README_en.md (updated)
- docs/notes/originals/生命循环.md (new)
- RELEASE_NOTES.md (this file)

Notes
-----
1. 该版本以教学与可视化为主，训练演示在资源受限的环境（CPU）上采取了较小的 batch 与短 epoch，旨在演示训练流程与可视化效果，而非复现完整训练性能。若需长期训练或高性能评估，请使用 `train.py` 或相应模块在 GPU 环境下运行完整训练流程。
2. 若要在浏览器中查看训练演示：启动 WebApp（推荐使用 `run-uv.bat` 或已激活 venv 下运行 `python -m mnist_pth_lab.webapp`），然后打开 `http://localhost:5000/training_demo`。

If you want, I can create a Git annotated tag `v0.3.0` and push it to the `gitee` remote so you can create a Release entry on Gitee from that tag.
# Release Notes — 2025-12-10

Release: Documentation patch (no code behavior changes)

Summary
- 更新 `README.md` 与 `README_en.md` 的示例命令，推荐使用模块执行形式（`python -m mnist_pth_lab.<cmd>`）。
- 补充了可编辑安装说明（`uv pip install -e .`），简化开发者本地调试流程。

Impact
- 无需更改代码；用户工作流更清晰，向后兼容旧命令（直接运行脚本路径仍然可用）。

Migration / Notes
- 若你使用 CI 或脚本，请考虑将 `python src/<script>.py` 替换为 `python -m mnist_pth_lab.<script>`，以避免 import path 问题并与 src-layout 一致。
- 推荐在开发环境中运行：

```powershell
uv venv .venv --python 3.10
.venv\Scripts\activate
uv pip install -e .
```

Where to look
- 文档变更位于 `README.md` 与 `README_en.md`，提交 id `94b291f`。
