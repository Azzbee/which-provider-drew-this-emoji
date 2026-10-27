# ConvNeXt-Tiny, 5 grouped folds: the final model

timm `convnext_tiny.fb_in1k`, fully fine-tuned at 224 px. Five StratifiedGroupKFold folds (duplicates grouped), AdamW 2e-4, weight decay 0.05, stochastic depth 0.1, OneCycle over 10 epochs, batch 32, flips and integer shifts; test predictions average 5 fold models x {image, mirror}.

| Run | Checkpoint kept | Out-of-fold accuracy | Kaggle public / private |
|---|---|---|---|
| A | lowest validation loss (epochs 7, 4, 6, 5, 8) | 97.29% | |
| **B (final)** | last epoch | **97.69%** | 0.98319 / **0.98218** |
| A + B | 10 checkpoints averaged | 97.70% | 0.98279 / 0.98198 |

One RTX 4090 on RunPod, 13 minutes per run of 5 folds.

Files: per-fold histories of run B, the OOF summary, fold 0 run metadata and the final Kaggle submission. Everything else (both runs, probabilities, figures, logs) is in `codebase/outputs/convnext224/` and `codebase/outputs/convnext224_last/`; code in `codebase/scripts/train_backbone.py`.
