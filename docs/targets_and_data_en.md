# MNIST `data` and `targets` — Quick Guide (English)

This note explains the difference between `dataset.data` and `dataset.targets` in PyTorch's MNIST, and shows the recommended way to use the dataset in training code.

Summary

- `train_data.data` — raw image pixels, shape `[60000, 28, 28]`, `uint8` values in `0..255`.
- `train_data.targets` — labels for each image, integers `0..9`.
- Recommended usage: `image, label = train_data[i]` or use `for image, label in train_data:` — the dataset will provide preprocessed tensors if transforms are set.

Common preprocessing steps

```python
# convert to float and normalize to [0,1]
x = train_data.data.float() / 255.0
# add channel dimension: [N, 1, 28, 28]
x = x.unsqueeze(1)
# convert labels to long
y = train_data.targets.long()
```

Why `dataset[i]` is preferred

- `dataset[i]` returns `(image, label)` and typically applies `transform` (e.g. normalization, ToTensor) so it's ready to feed to models.
- `dataset.data` and `dataset.targets` are the underlying raw arrays and may require manual processing.

Reference

- The original Chinese explanation is included in `targets 和data.md` in the repository.

