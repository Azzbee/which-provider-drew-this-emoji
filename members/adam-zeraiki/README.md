# Adam Zeraiki

Team leader. Built the shared codebase (`codebase/`) and the paper. One folder per method tried, each with its own README and the files that belong to it.

| Folder | Method | Validation | Kaggle private |
|---|---|---|---|
| `first-submission-keras-cnn/` | Keras CNN at 64 px, optional size-and-mode input | about 81% (91% with metadata) | |
| `baseline-cnn/` | PyTorch CNN, 3 blocks, no BatchNorm, 96 px | 76.71% | 0.78906 |
| `metadata-shortcut-diagnostic/` | Random forest / logistic regression on file metadata only | up to 87.37% | |
| `emojicnnv2/` | 4-stage BatchNorm CNN from scratch, 96 px (+ metadata ablation) | 90.96% (93.38%) | 0.91700 (0.94615) |
| `convnext-tiny-5fold-final-model/` | **ConvNeXt-Tiny, 5 grouped folds, flip TTA, last epoch: final model** | **97.69% OOF** | **0.98218** |
| `style-features-and-dinov2-convnext-ensemble/` | Hand-crafted style features, partially tuned ConvNeXt, DINOv2 + ConvNeXt average | 97.12%* | 0.96477 |
| `learned-fusion-transformer-and-mlp/` | Transformer and MLP fusion of DINOv2, ConvNeXt and style features | best 97.01%* | not submitted |
| `julia1-decision-model/` | Julia-1 decision model on the expert outputs | 93.98%* | not submitted |
| `jev-decision-model/` | Jev (jev-1.14.0) decision model, live trial | 95.65%* | not submitted |

\* On the DINOv2 branch's own 1,976-image development split, which was reused for earlier choices; not directly comparable to the codebase's 1,979-image stored split or 5-fold OOF.

Weights of the larger models, cached features, image data and API caches are not in the repository.
