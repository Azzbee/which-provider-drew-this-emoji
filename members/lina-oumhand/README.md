# Lina Oumhand

Explored the self-supervised Vision Transformer route (DINOv2).

## Methods tried

| Kaggle entry | Method | Validation | Public | Private |
|---|---|---|---|---|
| `submission.csv` | Extra-trees, 400 randomised trees, on resized image pixels, colour histograms and transparency | 82.14% (seed 42) | 0.81838 | 0.83259 |
| `submission2.csv` | Frozen ImageNet ResNet18 features + hand-crafted colour, transparency, edge and texture features, standardised, RBF SVM (C = 10, gamma = scale) | 96.76% (seed 42; 97.42% on seed 123) | 0.96719 | **0.96983** |
| `submission_dinov2_finetuned.csv` | DINOv2 ViT-S/14, last 2 of 12 blocks fine-tuned; files in `dinov2-partial-finetune/` | 95.90% | 0.95586 | 0.95607 |

No epochs for the first two: the extra-trees are 400 trees fitted once, ResNet18 was pretrained and frozen (no training on our images), and only the SVM was trained, until its solver converged. Each classifier was first trained on 80% of the labelled images for validation, then retrained on 100% to produce the Kaggle CSV.

## Extra Trees and frozen ResNet18 + SVM records

Code, both notebooks, requirements, the four original submission CSVs and measured outputs are in [`extra-trees-and-resnet18-svm/`](extra-trees-and-resnet18-svm/). Setup and reproduction instructions are in [`method-README.md`](extra-trees-and-resnet18-svm/method-README.md); the original Codex prompts are in [`codex-AI_USAGE.md`](extra-trees-and-resnet18-svm/codex-AI_USAGE.md).

The Extra Trees baseline uses 400 randomised trees, 24 x 24 RGB thumbnails, colour histograms and opacity. Its seed-42 stratified 80/20 validation accuracy is **82.14%** ([saved metrics](extra-trees-and-resnet18-svm/artifacts/baseline_metrics.json)); Kaggle public/private scores are **0.81838 / 0.83259**.

The SVM ablation uses the same seed-42 80/20 split, with 7,903 training and 1,976 validation images. The scaler is fitted only on the training fold. All three rows below use an RBF SVM with **C=10, gamma=scale**, as recorded in [`artifacts/improvement_comparison.csv`](extra-trees-and-resnet18-svm/artifacts/improvement_comparison.csv).

| SVM inputs | Features | Seed-42 validation accuracy |
|---|---:|---:|
| Hand-crafted colour, opacity, edge, texture and spatial descriptors alone | 1,607 | **96.05%** |
| Frozen ImageNet ResNet18 channel means and standard deviations alone, extracted at 128 x 128 | 1,920 | **91.65%** |
| Both descriptor families, selected model | 3,527 | **96.76%** |

The combined model also reaches **97.42%** on the seed-123 stratified split without changing its settings ([saved metrics](extra-trees-and-resnet18-svm/artifacts/improved_metrics_seed123.json)). These splits overlap and are not independent test sets. The combined model's Kaggle public/private scores are **0.96719 / 0.96983**. The original improved CSV is `submission_improved.csv` (identical to the packaged `submission.csv`); it was uploaded as `submission2.csv`.

There are **no epochs** for these submissions: Extra Trees fits 400 trees; ResNet18 stays frozen; the SVM is fitted to convergence. Each classifier is validated on 80/20 and then refitted on **100% of the labelled images** for its submission. The 78 MB fitted `artifacts/improved_model.joblib` and all feature/pretrained-weight caches are intentionally omitted; `improve.py` recreates them from separately supplied competition data.

## Completion record

- [x] What `submission.csv` and `submission2.csv` were (submission2 is your best entry): documented above and in the method folder.
- [x] Add any other notebooks here, and send Adam the AI tools and prompts you used (he keeps the AI-use record): both notebooks and the Codex prompts are included in this pull request and `codebase/AI_USAGE.md`.
