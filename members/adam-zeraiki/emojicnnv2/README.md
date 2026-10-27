# EmojiCNNv2

CNN from scratch at 96 px: four stages (32, 64, 128, 256 channels), BatchNorm, global average pooling, dropout 0.3; 390,695 parameters. AdamW 2e-3, weight decay 1e-4, OneCycle, batch 64, 25 epochs, flips and integer shifts only. A second version adds 8 file-format features by late fusion (ablation).

| Version | Validation (stored split) | Kaggle public / private |
|---|---|---|
| pixels only (epoch 23 kept) | 90.96% | 0.90747 / 0.91700 |
| + file metadata (epoch 25) | 93.38% | 0.93824 / 0.94615 |

Files: training histories (`*_history.csv`), run metadata (`*_run.json`), Kaggle submissions. Code: `codebase/src/emoji_provider/models.py`, `codebase/scripts/run_cpu_experiments.py`.
