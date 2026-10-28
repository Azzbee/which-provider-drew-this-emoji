# Lina Oumhand

Explored the self-supervised Vision Transformer route (DINOv2).

## Methods tried

| Kaggle entry | Method | Validation | Public | Private |
|---|---|---|---|---|
| `submission.csv` | Extra-trees, 400 randomised trees, on resized image pixels, colour histograms and transparency | 82.14% | 0.81838 | 0.83259 |
| `submission2.csv` | Frozen ImageNet ResNet18 features + hand-crafted colour, transparency, edge and texture features, standardised, RBF SVM (C = 10, gamma = scale) | 96.76% (97.42% on a second split) | 0.96719 | **0.96983** |
| `submission_dinov2_finetuned.csv` | DINOv2 ViT-S/14, last 2 of 12 blocks fine-tuned; files in `dinov2-partial-finetune/` | 95.90% | 0.95586 | 0.95607 |

No epochs for the first two: the extra-trees are 400 trees fitted once, ResNet18 was pretrained and frozen (no training on our images), and only the SVM was trained, until its solver converged. Each classifier was first trained on 80% of the labelled images for validation, then retrained on 100% to produce the Kaggle CSV.

## To complete

- What `submission.csv` and `submission2.csv` were (submission2 is your best entry).
- Add any other notebooks here, and send Adam the AI tools and prompts you used (he keeps the AI-use record).
