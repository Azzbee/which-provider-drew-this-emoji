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
| 9 | DINOv2 + ConvNeXt ensemble | not in the codebase | not recorded | 0.96477 |

## To complete

- Method 9 and the three-backbone fusion idea (paper Section 4.5, red box): which backbones, features or probabilities, fusion transformer size, training data. Codex's session logs on this Mac contain no record of it (checked 27 October), so it was run elsewhere, perhaps in ChatGPT in the browser. If you find the notebook or chat, add it here.
