# MNIST `data` and `targets` — Full Guide (English)

This document is the full English translation of the Chinese `targets_and_data_zh.md`. It explains what `dataset.data`, `dataset.targets`, and `dataset[i]` contain in PyTorch's MNIST dataset, shows practical examples, and gives recommended usage patterns and preprocessing steps.

Quick summary

- `train_data.data` — raw image pixels, shape `[60000, 28, 28]`, dtype `uint8`, values range 0..255.
- `train_data.targets` — labels for each image, integers 0..9 (often `torch.tensor`).
- `train_data[i]` (or `dataset[i]`) — preferred access: returns `(image, label)` and applies any `transform` passed to the dataset (e.g., `ToTensor`, normalization).

What you saw when running `src/test.py`

Example run output (abridged):

```
torch.Size([60000, 28, 28])    # train_data.data.shape
torch.Size([10000, 28, 28])    # test_data.data.shape
tensor(7)                       # test_data.targets[0]
tensor([[  0,   0,   0, ... ]])  # raw pixel matrix for a 28x28 image
```

Plain language: the training set has 60,000 grayscale images (28×28), the test set has 10,000. Each pixel is an integer in 0..255 (0=black, 255=white). The first example label is `7` and the printed pixel array corresponds to a handwritten digit 7.

Why you see many zeros and 255s

MNIST is a grayscale dataset (single channel). Values are integer 0..255 (uint8). Background is near 0, strokes (pen ink) are large values near 255; intermediate numbers are shades of gray.

Typical preprocessing steps (practical)

```python
# 1) Convert to float and normalize to [0,1]
images = train_data.data.float() / 255.0   # shape [N, 28, 28], float32

# 2) Add channel dimension for CNN: [N, 1, 28, 28]
images = images.unsqueeze(1)

# 3) Ensure labels are long (int64)
labels = train_data.targets.long()

# Recommended: use transforms so dataset[i] yields ready tensors
from torchvision import transforms
transform = transforms.Compose([
	transforms.ToTensor(),                     # -> [C,H,W], float in [0,1]
	transforms.Normalize((0.1307,), (0.3081,)) # example MNIST normalization
])
train = torchvision.datasets.MNIST(root='datasets', train=True, transform=transform, download=True)
for img, label in train:
	# img shape: [1,28,28], dtype: torch.float32, range depends on Normalize
	break
```

Practical examples and differences

- `train_data[0]` returns `(image, label)` — the recommended, high-level API. If `transform` is set the returned `image` is a processed tensor ready for the network.
- `train_data.data[0]` returns the raw uint8 pixel array of shape `[28, 28]`.
- `train_data.targets[0]` returns the label (e.g. `tensor(7)`).

Cheat-sheet

| Expression | Returns | Shape / dtype |
|---|---:|---|
| `train_data[0]` | `(image, label)` | `image: [1,28,28]` (float32 if transform applied), `label: int` |
| `train_data.data[0]` | raw pixels | `[28,28]`, uint8 |
| `train_data.targets[0]` | label | scalar tensor or int |

Recommended workflow

1. Always use `dataset[i]` or `for image,label in dataset` in training/evaluation loops (this applies transforms automatically).
2. If you inspect raw arrays, convert dtype, add channel dimension and normalize before feeding to models.
3. Prefer `ToTensor()` and `Normalize()` transforms for clean, reproducible preprocessing.

Quick code contrast

```python
# Recommended (works with DataLoader):
image, label = train_data[0]

# Raw access (inspect only):
raw_image = train_data.data[0]        # shape [28,28], dtype uint8
raw_label = train_data.targets[0]     # label tensor

# Convert raw to model-ready:
image_tensor = raw_image.float().unsqueeze(0) / 255.0  # [1,28,28]
```

Final notes

- Use `dataset[i]` for robust code; `.data` and `.targets` are helpful for quick debugging and visualization but are not the typical training interface.
- See `docs/targets_and_data_zh.md` for the original Chinese explanation with illustrated examples.

