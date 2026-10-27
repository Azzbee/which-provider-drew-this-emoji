# Adam Zeraiki

Team leader. Built the shared codebase (`codebase/`) and wrote the paper with AI assistance (record to be added in `codebase/AI_USAGE.md`).

## Methods tried

| # | Method | Where | Validation | Kaggle private |
|---|---|---|---|---|
| 1 | First-submission notebook: small Keras CNN at 64 px, 15% stratified hold-out; optional size-and-mode input (`USE_SIZE_AND_MODE`) | `notebooks/01_first_submission_30sep.ipynb` (drafted with Claude) | about 81%; about 91% with the flag | |
| 2 | Baseline CNN (3 conv blocks, no BatchNorm, Adam 1e-3, 20 epochs, 96 px, white compositing, grouped split) | `notebooks/02_baseline_cnn_codex_30sep.ipynb` (generated and debugged with OpenAI Codex), weights in `codebase/data/baseline_sept30/` | 76.71% | 0.78906 |
| 3 | File-metadata diagnostic (logistic regression, random forest) | `codebase/src/emoji_provider/metadata_model.py` | up to 87.37% | |
| 4 | EmojiCNNv2 (4 stages, BatchNorm, AdamW + OneCycle, exact-pixel augmentation, 96 px) | `codebase/src/emoji_provider/models.py`, `codebase/scripts/run_cpu_experiments.py` | 90.96% | 0.91700 |
| 5 | EmojiCNNv2 + 8 file-format features (late fusion, ablation) | same | 93.38% | 0.94615 |
| 6 | ConvNeXt-Tiny 224 px, 5 grouped folds, flip TTA, lowest-val-loss epoch (run A) | `codebase/scripts/train_backbone.py`, `codebase/outputs/convnext224/` | 97.29% OOF | |
| 7 | **Same, last epoch (run B): final model** | `codebase/outputs/convnext224_last/` | **97.69% OOF** | **0.98218** |
| 8 | Average of runs A and B (10 checkpoints) | `codebase/scripts/make_final_submission.py` | 97.70% OOF | 0.98198 |
| 9 | DINOv2 ViT-S/14, last 2 blocks fine-tuned (with ChatGPT); submitted from Lina's account | `chatgpt_experiments/01_*` | 95.90% (branch dev split) | 0.95607 |
| 10 | DINOv2 + partially tuned ConvNeXt, equal average | `chatgpt_experiments/02_*` | 97.12% (branch dev split) | 0.96477 |
| 11 | Style features (ExtraTrees, gradient boosting), frozen ConvNeXt + logistic regression | `chatgpt_experiments/02_*` | 84.72%, 90.03%, 90.64% | |
| 12 | Learned fusion of DINOv2, ConvNeXt and style features: transformer and MLP heads | `chatgpt_experiments/03_*` | best 97.01% (MLP) | not submitted |
| 13 | Decision models: Julia-1 (open) and Jev (hosted) on the expert outputs | `chatgpt_experiments/04_*`, `05_*` | 93.98%, 95.65% | not submitted |

## Notes

- Methods 9 to 13 were run with ChatGPT; see `chatgpt_experiments/README.md`. Their accuracies are on that branch's own 1,976-image development split, not comparable to the codebase's 1,979-image stored split or 5-fold OOF.
- Weights, cached features and the full API caches are in the 2.1 GB archive on Google Drive, not in this repository.
