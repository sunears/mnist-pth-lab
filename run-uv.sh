#!/bin/bash
# ==================================================
# VideoX-Fun General UV Runner
# ==================================================
set -e

echo "[Info] Checking environment..."

# 1. Find uv
if [ -f "$(dirname "$0")/uv" ]; then
    UV_EXE="$(dirname "$0")/uv"
    echo "[Success] Using local uv"
else
    if command -v uv >/dev/null 2>&1; then
        UV_EXE=$(command -v uv)
        echo "[Success] Using system uv"
    else
        echo "[Error] uv not found. Please place uv executable in this folder or add it to PATH."
        read -p "Press any key to continue..."
        exit 1
    fi
fi

echo "[Debug] UV Executable: \"$UV_EXE\""

# 2. Configure UV
# Install Python into ./uv/python to keep it self-contained
export UV_PYTHON_INSTALL_DIR="$(dirname "$0")/.uv_python"
export UV_MANAGED_PYTHON=true

# 3. Sync Environment
echo "[Info] Installing dependencies..."
# Create venv if needed and install from requirements.txt
"$UV_EXE" venv .venv --python 3.10 --allow-existing
"$UV_EXE" pip install -r requirements.txt

# Set PYTHONPATH to include src
export PYTHONPATH="$(dirname "$0")/src:$PYTHONPATH"

# 4. Execute
if [ -z "$1" ]; then
    echo "[Info] No script specified. Opening interactive shell..."
    echo "[Tip] You can verify environment with: python --version"
    "$UV_EXE" run bash
else
    echo "[Info] Running: $@"
    "$UV_EXE" run "$@"
fi
