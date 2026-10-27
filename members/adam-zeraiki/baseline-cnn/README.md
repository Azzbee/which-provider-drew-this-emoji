# Baseline CNN (30 September)

PyTorch CNN at 96 px: three Conv/ReLU/MaxPool blocks (32, 64, 128), global average pooling, dropout 0.25, linear head. Images composited on white, aspect-preserving Lanczos resize, white padding. Adam 1e-3, batch 64, 20 epochs, seed 42, grouped split on pixel hashes.

Result: 76.71% on the stored 1,979-image validation split (1,518 correct); Kaggle 0.77546 public, 0.78906 private. Training accuracy was only 75%: the model underfits (paper section 6.6).

Files: `baseline_cnn_30sep.ipynb` (notebook), `best_model.pt` (weights, 372 KB). The same weights are re-scored by `codebase/scripts/` (see `codebase/README.md`).
