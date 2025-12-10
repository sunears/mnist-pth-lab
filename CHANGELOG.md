# Changelog

所有显著变更都会记录在此。遵循简明时间轴格式（按时间倒序）。

## 2025-12-10 — 文档更新
- 更新项：修订 `README.md` 与 `README_en.md` 的示例命令。
  - 将示例命令推荐为模块调用形式：`python -m mnist_pth_lab.<cmd>`（支持 `src` layout）。
  - 在 `run-uv.bat` / `run-uv.sh` 示例中加入模块运行示例，并保留直接运行脚本的向后兼容写法。
  - 在英文 README 中补充 `uv pip install -e .`（可编辑安装）建议。
  - 格式化并清理 README，使新用户更易上手（包括 `run-uv` 使用说明）。

参考提交：`94b291f`（docs: 更新 README 示例，推荐模块运行与可编辑安装）
