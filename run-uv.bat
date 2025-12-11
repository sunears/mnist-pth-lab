@echo off
REM ==================================================
REM Mnist-pth-lab General UV Runner
REM ==================================================
chcp 65001 >nul
setlocal EnableDelayedExpansion

echo [Info] Checking environment...

REM 1. Find uv
if exist "%~dp0uv.exe" (
    set "UV_EXE=%~dp0uv.exe"
    echo [Success] Using local uv
) else (
    where uv >nul 2>&1
    if !errorlevel!==0 (
        for /f "delims=" %%i in ('where uv') do set "UV_EXE=%%i"
        echo [Success] Using system uv
    ) else (
        echo [Error] uv not found. Please place uv.exe in this folder or add it to PATH.
        pause
        exit /b 1
    )
)

echo [Debug] UV Executable: "!UV_EXE!"

REM 2. Configure UV
REM Install Python into ./uvwin/python to keep it self-contained (avoid conflict with Linux uv binary)
set "UV_PYTHON_INSTALL_DIR=%~dp0uvwin\python"
set "UV_MANAGED_PYTHON=true"

REM 3. Sync Environment
echo [Info] Installing dependencies...
REM Create venv if needed and install from requirements.txt
"!UV_EXE!" venv .venv --python 3.10 --allow-existing
"!UV_EXE!" pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [Error] Dependency installation failed.
    pause
    exit /b 1
)

REM Set PYTHONPATH to include src
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"

REM 4. Execute
if "%~1"=="" (
    echo [Info] No script specified. Opening interactive shell...
    echo [Tip] You can verify environment with: python --version
    "%UV_EXE%" run cmd /k
) else if "%~1"=="-test" (
    echo [Info] Running test script: src\mnist_pth_lab\test.py
    "%UV_EXE%" run "%~dp0src\mnist_pth_lab\test.py"
) else (
    echo [Info] Running: %*
    "%UV_EXE%" run %*
)
