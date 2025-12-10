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
:: Recommended (module form, src-layout)
run-uv.bat python -m mnist_pth_lab.train --epochs 5 --batch-size 64 --save-path models\mnist_cnn.pth

:: Backwards-compatible: run script directly
run-uv.bat python src\train.py --epochs 5 --batch-size 64 --save-path models\mnist_cnn.pth
```
*Note: The first run will automatically create a virtual environment in `.venv` and install dependencies.*

### 2. Evaluation
Evaluate the trained model on the test set.

```cmd
run-uv.bat python -m mnist_pth_lab.eval --model models\mnist_cnn.pth --output-dir experiments\output
```
Check `experiments\output` for the evaluation report and confusion matrix.

### 3. Data Processing Tools
This project provides scripts to convert between MNIST IDX format and image format (PNG).

#### Unpack (IDX -> PNG)
Unpack IDX data into an image folder and a CSV label file.

```cmd
:: Unpack training data (recommended module form)
run-uv.bat python -m mnist_pth_lab.unpack_idx --images-idx data\MNIST\raw\train-images-idx3-ubyte --labels-idx data\MNIST\raw\train-labels-idx1-ubyte --out-dir unpacked_data\train

:: Unpack test data
run-uv.bat python -m mnist_pth_lab.unpack_idx --images-idx data\MNIST\raw\t10k-images-idx3-ubyte --labels-idx data\MNIST\raw\t10k-labels-idx1-ubyte --out-dir unpacked_data\test
```

#### Pack (PNG -> IDX)
Pack an image folder back into IDX format (useful for creating custom datasets).

```cmd
:: Pack training data (recommended module form)
run-uv.bat python -m mnist_pth_lab.pack_idx --images-dir unpacked_data\train\images --labels-csv unpacked_data\train\labels.csv --out-images-idx new-train-images-idx3-ubyte.gz --out-labels-idx new-train-labels-idx1-ubyte.gz

:: Pack test data
run-uv.bat python -m mnist_pth_lab.pack_idx --images-dir unpacked_data\test\images --labels-csv unpacked_data\test\labels.csv --out-images-idx new-test-images-idx3-ubyte.gz --out-labels-idx new-test-labels-idx1-ubyte.gz
```

## Quick Start (Linux/macOS)

We provide a one-click runner script `run-uv.sh` that handles environment creation and execution automatically.

### 1. Training
Train the model for 5 epochs (default) and save to `models/mnist_cnn.pth`.

```bash
# Recommended (module form)
./run-uv.sh python -m mnist_pth_lab.train --epochs 5 --batch-size 64 --save-path models/mnist_cnn.pth

# Backwards-compatible: run script directly
./run-uv.sh python src/train.py --epochs 5 --batch-size 64 --save-path models/mnist_cnn.pth
```
*Note: The first run will automatically create a virtual environment in `.venv` and install dependencies.*

### 2. Evaluation
Evaluate the trained model on the test set.

```bash
./run-uv.sh python -m mnist_pth_lab.eval --model models/mnist_cnn.pth --output-dir experiments/output
```

### Run the Web App (visualization & live prediction)

The project includes a small web application for drawing handwritten digits and visualizing convolution activations. By default it listens on `0.0.0.0:5000`.

Windows (with `run-uv.bat`):
```powershell
run-uv.bat python -m mnist_pth_lab.webapp
```

Linux/macOS (with `run-uv.sh`):
```bash
./run-uv.sh python -m mnist_pth_lab.webapp
```

Or run directly inside an activated venv:
```bash
python -m mnist_pth_lab.webapp
```

Open `http://localhost:5000` in your browser to use the drawing canvas, predict, and view kernel activations.

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
    # OR (recommended for development) install editable package
    uv pip install -e .
    ```

## Data Processing Tools

This project provides scripts to convert between MNIST IDX format and image format (PNG).

### 1. Unpack (IDX -> PNG)
Unpack IDX data into an image folder and a CSV label file.

```bash
./run-uv.sh python -m mnist_pth_lab.unpack_idx --images-idx data/MNIST/raw/train-images-idx3-ubyte --labels-idx data/MNIST/raw/train-labels-idx1-ubyte --out-dir unpacked_data/train

# Unpack test data
./run-uv.sh python -m mnist_pth_lab.unpack_idx --images-idx data/MNIST/raw/t10k-images-idx3-ubyte --labels-idx data/MNIST/raw/t10k-labels-idx1-ubyte --out-dir unpacked_data/test
```

### 2. Pack (PNG -> IDX)
Pack an image folder back into IDX format (useful for creating custom datasets).

```bash
./run-uv.sh python -m mnist_pth_lab.pack_idx --images-dir unpacked_data/train/images --labels-csv unpacked_data/train/labels.csv --out-images-idx new-train-images-idx3-ubyte.gz --out-labels-idx new-train-labels-idx1-ubyte.gz

# Pack test data
./run-uv.sh python -m mnist_pth_lab.pack_idx --images-dir unpacked_data/test/images --labels-csv unpacked_data/test/labels.csv --out-images-idx new-test-images-idx3-ubyte.gz --out-labels-idx new-test-labels-idx1-ubyte.gz
```

2.  **Run Scripts**:
    ```bash
    # Train (module form)
    python -m mnist_pth_lab.train --epochs 5 --save-path models/mnist_cnn.pth
    
    # Eval (module form)
    python -m mnist_pth_lab.eval --model models/mnist_cnn.pth
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

## FAQ

- **Q: Why does README recommend `python -m mnist_pth_lab.*`?**
    - A: The project uses a `src` layout (package in `src/mnist_pth_lab/`). Running via `python -m mnist_pth_lab.<module>` ensures imports resolve like an installed package and avoids relative import problems that can occur when running scripts directly from `src/`.

- **Q: I don't have `uv`. Can I still run the project?**
    - A: Yes. `uv` is recommended to simplify environment setup. Without `uv` create a virtualenv and run `pip install -r requirements.txt` or `pip install -e .`.

- **Q: How do I install for development?**
    - A: Inside a `uv` venv or a manually created virtualenv run `uv pip install -e .` (or `pip install -e .`) to enable editable installs.

- **Q: What's the difference between `run-uv.bat` and `run-uv.sh`?**
    - A: They provide the same helper behavior for different platforms — Windows uses `run-uv.bat`, Unix-like systems use `run-uv.sh`. Both call `uv` to prepare the environment and execute the given command.

- **Q: Will this documentation change affect my saved models or experiments?**
    - A: No. This is a documentation-only change. Existing model files (for example `models/mnist_cnn.pth`) remain untouched unless you explicitly overwrite them by re-running training with the same save path.

- **Q: Where to report issues or request features?**
    - A: Please open an Issue in the repository or contact the maintainers listed in the README.
