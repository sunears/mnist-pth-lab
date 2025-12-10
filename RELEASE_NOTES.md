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
