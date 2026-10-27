# Erdem Sanver

Ran the team's main experiment series and the team's best Kaggle entry (E15, 0.98238 private).

## Experiments E1a to E9 (13 October, 80/20 split)

Source: the team spreadsheet, copied into `codebase/report/reported_experiments.json`.

| Id | Model | Method | Epochs | Val. acc. | Public | Preprocessing |
|---|---|---|---|---|---|---|
| E1a | Random forest (300 trees) | file metadata only, no pixel content | | 65.84% | | width/height, PIL mode, RGB under fully transparent pixels |
| E1b | SVM (RBF, C=10) | 512-bin RGB colour histogram + HOG (9 orientations, 8x8 cells) | | 81.28% | | composited on white, 64x64 bicubic |
| E2 | simple CNN from scratch | 3 conv blocks (32-64-128), AdamW 1e-3, OneCycle, batch 64 | 15 | 92.11% | | PIL `convert('RGB')` (default), 96x96 bicubic, ImageNet normalisation |
| E3 | simple CNN from scratch | same CNN as E2, only preprocessing changes | 15 | 93.62% | 0.93642 | composited on white, 96x96 bicubic |
| E4 | ResNet18, frozen backbone | only the linear head is trained (AdamW 3e-3, OneCycle) | 10 | 77.48% | | composited on white, 128x128 bicubic |
| E5 | ResNet18, fine-tuned | full fine-tuning (AdamW 5e-4, OneCycle) | 10 | 93.72% | | as E4 |
| E6 | EfficientNet-B0, fine-tuned | full fine-tuning (AdamW 1e-3, OneCycle) | 10 | 93.12% | | as E5 |
| E6b | ConvNeXt-Tiny, fine-tuned | full fine-tuning (AdamW 2e-4, OneCycle, batch 32) | 10 | 95.90% | | as E5 (128 px) |
| E7 | ConvNeXt-Tiny + augmentation | as E6b + h-flip, ±10° rotation, 5% shift, 0.9-1.1 scale, label smoothing 0.1 | 15 | 94.38% | | as E6b + random flip/affine |
| E8 | ConvNeXt-Tiny 224 px | exactly as E6b, input 128 -> 224 | 10 | 97.57% | 0.97408 | composited on white, 224x224 bicubic |
| E9 | ensemble E6b + E8 | soft voting (equal-weight softmax average) | | 97.93% | 0.97671 | as each member |

## Later Kaggle entries (descriptions as submitted)

| Entry | Description on Kaggle | Public | Private |
|---|---|---|---|
| E10b | E8 recipe (ConvNeXt 224 px) retrained on 100% of train data | 0.97833 | 0.97773 |
| E11 | E9 recipe (ConvNeXt 128+224 ensemble) retrained on 100% of train data | 0.97914 | 0.97793 |
| E13v | soft-vote ensemble E8 + E12b + E13 | 0.97772 | 0.97955 |
| **E15** | soft-vote ensemble of 3 ConvNeXt (224 bicubic + 144/216 nearest) retrained on 100% of train data | 0.98117 | **0.98238** |

## To complete

- What E12b and E13 were (members of E13v), and the exact settings of the 144 and 216 px nearest-neighbour models in E15.
- Add the notebooks or scripts here (`members/erdem-sanver/`), and send Adam the AI tools and prompts you used (he keeps the AI-use record).
