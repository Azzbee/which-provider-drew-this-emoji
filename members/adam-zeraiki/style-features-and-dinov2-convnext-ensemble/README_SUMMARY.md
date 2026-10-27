# Style features and the DINOv2 + ConvNeXt ensemble (17 October)

On the DINOv2 branch's own split (7,903 train / 1,976 development, duplicates grouped):

| Model | Development accuracy |
|---|---|
| 1,460 hand-crafted style features, histogram gradient boosting | 90.03% |
| same features, extra-trees | 84.72% |
| frozen ConvNeXt-Tiny features + logistic regression | 90.64% |
| ConvNeXt-Tiny, only the last stage fine-tuned | 95.55% |
| DINOv2 (see `members/lina-oumhand/dinov2-partial-finetune/`), with flip averaging | 95.95% |
| **DINOv2 + ConvNeXt, equal probability average** | **97.12%** (Kaggle 0.96881 public, 0.96477 private) |

The development split was reused for earlier choices, so these are optimistic estimates. Original code, configurations and results are in this folder (see `README.md`, `EXPERIMENT_RESULTS.md`, `METHOD_NOTES.md`). Weights and cached arrays are not in the repository.
