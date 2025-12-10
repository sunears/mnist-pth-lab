# 仓库变动中文摘要

生成日期：2025-12-10

分支：`master`

当前工作树状态（来自 `git status --porcelain`）：
- `D run_with_uv.bat` — 文件被删除（工作区有删除改动）。
- `M src/mnist_pth_lab.egg-info/PKG-INFO` — 已修改（未提交）。
- `M src/mnist_pth_lab.egg-info/SOURCES.txt` — 已修改（未提交）。
- `M src/mnist_pth_lab.egg-info/top_level.txt` — 已修改（未提交）。

最近重要提交（按时间倒序，来自 `git log`）：
- `cf51589` 2025-12-10 — docs: add CHANGELOG/RELEASE_NOTES and README FAQ
  - 添加 `CHANGELOG.md` 与 `RELEASE_NOTES.md`，并在 `README` / `README_en` 末尾加入“常见问题（FAQ）”。
- `94b291f` 2025-12-10 — docs: 更新 README 示例，推荐模块运行与可编辑安装
  - 将 README 中的示例命令改为推荐的模块运行形式：`python -m mnist_pth_lab.<cmd>`，并在示例中展示 `run-uv` 的调用；英文版补充 `uv pip install -e .` 建议。
- `b3dbd50` 2025-12-10 — chore: adopt src-layout (package-dir=src) and add mnist_pth_lab package
  - 将项目转换为标准 `src` 布局，新增 `src/mnist_pth_lab/` 包（并把模块移动/复制到包中）。
- `2468628` 2025-12-10 — refactor: move modules into src/mnist_pth_lab package
  - 重构：将顶层 `src/*.py` 模块迁移到包内并修正导入路径。
- `a0292ed` 2025-12-10 — docs: update NOTES_INDEX to point to archived originals and tidy moved files
  - 更新笔记索引，归档原始笔记到 `docs/notes/originals/`，并创建英文友好副本。
- `a772862` 2025-12-10 — docs: update NOTES_INDEX to point to archived originals

说明与建议：
- 本次一系列改动以「文档整理」与「src-layout 重构」为主，代码行为未做算法性修改；但因目录结构改变，建议在本地环境中使用以下方式运行以避免导入路径问题：

```powershell
.\run-uv.bat python -m mnist_pth_lab.test
```

- 若你在 CI 或脚本中仍使用 `python src/<script>.py`，建议改为 `python -m mnist_pth_lab.<script>` 或在环境中执行 `uv pip install -e .`。

若需我同时把工作区的未提交改动（如 egg-info 文件的修改或 run_with_uv.bat 的删除）包含到 commit 中，请确认是否一并提交；也可以先保留这些改动由你手动检查再提交。
