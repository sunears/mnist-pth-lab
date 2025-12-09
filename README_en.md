# mnist-pth-lab

A complete PyTorch project for MNIST handwritten digit recognition, featuring a structured pipeline for training, evaluation, and experiment management.

## Project Overview

This project demonstrates a production-ready PyTorch workflow:
- **Model**: Simple CNN with Dropout for robust classification.
- **Data**: Automatic MNIST download and processing using `torchvision`.
- **Training**: Full training loop with `tqdm` progress bars, validation logging, and best model checkpointing.
- **Evaluation**: Comprehensive breakdown with Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and visualization of top mistakes.
- **Management**: Uses `uv` for ultra-fast dependency management and virtual environments.

## Prerequisite: Install `uv`

This project recommends using **uv** for Python package management.

1.  **Download uv**:
    -   Visit [uv releases](https://github.com/astral-sh/uv/releases) and download the `uv-x86_64-pc-windows-msvc.zip` (for Windows).
    -   Extract `uv.exe` and place it in this project's root directory OR add it to your system PATH.

2.  **Troubleshooting**:
    -   If `run-uv.bat` fails to find `uv`, manually place `uv.exe` in the same folder as the script.

## Quick Start (Windows)

We provide a one-click runner script `run-uv.bat` that handles environment creation and execution automatically.

### 1. Training
Train the model for 5 epochs (default) and save to `models/mnist_cnn.pth`.

```cmd
run-uv.bat src\train.py --epochs 5 --batch-size 64 --save-path models\mnist_cnn.pth
```
*Note: The first run will automatically create a virtual environment in `.venv` and install dependencies.*

### 2. Evaluation
Evaluate the trained model on the test set.

```cmd
run-uv.bat src\eval.py --model models\mnist_cnn.pth --output-dir experiments\output
```
Check `experiments\output` for the evaluation report and confusion matrix.

### 3. Data Processing Tools
This project provides scripts to convert between MNIST IDX format and image format (PNG).

#### Unpack (IDX -> PNG)
Unpack IDX data into an image folder and a CSV label file.

```cmd
:: Unpack training data
run-uv.bat src\unpack_idx.py --images-idx data\MNIST\raw\train-images-idx3-ubyte --labels-idx data\MNIST\raw\train-labels-idx1-ubyte --out-dir unpacked_data\train

:: Unpack test data
run-uv.bat src\unpack_idx.py --images-idx data\MNIST\raw\t10k-images-idx3-ubyte --labels-idx data\MNIST\raw\t10k-labels-idx1-ubyte --out-dir unpacked_data\test
```

#### Pack (PNG -> IDX)
Pack an image folder back into IDX format (useful for creating custom datasets).

```cmd
:: Pack training data
run-uv.bat src\pack_idx.py --images-dir unpacked_data\train\images --labels-csv unpacked_data\train\labels.csv --out-images-idx new-train-images-idx3-ubyte.gz --out-labels-idx new-train-labels-idx1-ubyte.gz

:: Pack test data
run-uv.bat src\pack_idx.py --images-dir unpacked_data\test\images --labels-csv unpacked_data\test\labels.csv --out-images-idx new-test-images-idx3-ubyte.gz --out-labels-idx new-test-labels-idx1-ubyte.gz
```

## Quick Start (Linux/macOS)

We provide a one-click runner script `run-uv.sh` that handles environment creation and execution automatically.

### 1. Training
Train the model for 5 epochs (default) and save to `models/mnist_cnn.pth`.

```bash
./run-uv.sh src/train.py --epochs 5 --batch-size 64 --save-path models/mnist_cnn.pth
```
*Note: The first run will automatically create a virtual environment in `.venv` and install dependencies.*

### 2. Evaluation
Evaluate the trained model on the test set.

```bash
./run-uv.sh src/eval.py --model models/mnist_cnn.pth --output-dir experiments/output
```
Check `experiments/output` for the evaluation report and confusion matrix.

## Manual Usage

If you prefer running commands manually, you can follow these steps:

1.  **Create Environment & Install**:
    ```bash
    # Create venv
    uv venv .venv --python 3.10
    
    # Activate
    # Windows:
    .venv\Scripts\activate
    # Linux/Mac:
    source .venv/bin/activate
    
    # Install dependencies
    uv pip install -r requirements.txt
    # OR install from pyproject.toml
    uv pip install .
    ```

## Data Processing Tools

This project provides scripts to convert between MNIST IDX format and image format (PNG).

### 1. Unpack (IDX -> PNG)
Unpack IDX data into an image folder and a CSV label file.

```bash
# Unpack training data
./run-uv.sh src/unpack_idx.py --images-idx data/MNIST/raw/train-images-idx3-ubyte --labels-idx data/MNIST/raw/train-labels-idx1-ubyte --out-dir unpacked_data/train

# Unpack test data
./run-uv.sh src/unpack_idx.py --images-idx data/MNIST/raw/t10k-images-idx3-ubyte --labels-idx data/MNIST/raw/t10k-labels-idx1-ubyte --out-dir unpacked_data/test
```

### 2. Pack (PNG -> IDX)
Pack an image folder back into IDX format (useful for creating custom datasets).

```bash
# Pack training data
./run-uv.sh src/pack_idx.py --images-dir unpacked_data/train/images --labels-csv unpacked_data/train/labels.csv --out-images-idx new-train-images-idx3-ubyte.gz --out-labels-idx new-train-labels-idx1-ubyte.gz

# Pack test data
./run-uv.sh src/pack_idx.py --images-dir unpacked_data/test/images --labels-csv unpacked_data/test/labels.csv --out-images-idx new-test-images-idx3-ubyte.gz --out-labels-idx new-test-labels-idx1-ubyte.gz
```

2.  **Run Scripts**:
    ```bash
    # Train
    python src/train.py --epochs 5 --save-path models/mnist_cnn.pth
    
    # Eval
    python src/eval.py --model models/mnist_cnn.pth
    ```

## File Structure

```
mnist-pth-lab/
├── pyproject.toml       # Project metadata and dependencies
├── requirements.txt     # Dependency list (legacy/compat)
├── run-uv.bat           # Windows helper script for UV
├── run-uv.sh            # Linux/macOS helper script for UV
├── AGENTS.md            # AI assistant development guide
├── CLINE.md             # CLINE development guide
├── gemini.md            # Gemini development guide
├── models/              # Saved .pth models
├── data/                # Downloaded MNIST data
├── experiments/         # Logs, charts, and artifacts
└── src/
    ├── model.py         # CNN architecture definition
    ├── dataset.py       # DataLoaders and transformations
    ├── train.py         # Training loop & logging
    ├── eval.py          # Metrics & confusion matrix
    └── utils.py         # Seeds, logging, saving helpers
    └── docs/linear计算过程.md # Linear layer computation notes (Chinese)
    └── docs/targets_and_data_zh.md # MNIST `data` and `targets` explanation (Chinese, docs folder)

## Documentation

- `docs/linear计算过程.md` — Linear layer computation notes (Chinese).
- `docs/targets_and_data_zh.md` — MNIST `data` vs `targets` explanation and examples (Chinese, in `docs/`).
- `docs/targets_and_data_en.md` — Full English guide to `targets` and `data` (in `docs/`).
```

## GitHub Actions (CI/CD)

To automate training or testing on GitHub, you can use the `astral-sh/setup-uv` action.

Example snippet for `.github/workflows/test.yml`:
```yaml
steps:
  - uses: actions/checkout@v4
  - uses: astral-sh/setup-uv@v1
  - run: uv sync
  - run: uv run python src/train.py --epochs 1  # Smoke test
```

## License

MIT License. See `LICENSE` for details.
