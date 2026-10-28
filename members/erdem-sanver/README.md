# Erdem Sanver

Ran the experiment series E1a to E16 and the team's best Kaggle entry (E15, 0.98238 private).

Everything needed to rerun the series is in [`experiment-series-E1-E16/`](experiment-series-E1-E16/):
the code, the exact command line of every experiment, one JSON record per experiment (configuration,
scores, per-class confusion matrix, training curve, runtime), the saved validation and test
probabilities, the training logs and a detailed write-up (`report/detailed.pdf`). The original
13 October spreadsheet is kept as `Strategies_Computer_Vision_Project_13_October.xlsx`; the complete
one (E1a to E16, regenerated from the JSON records) is
`experiment-series-E1-E16/Strategies_Computer_Vision_Project_E1-E16.xlsx`.

## Common protocol

| | |
|---|---|
| Validation | stratified 80/20 split of `train_labels.csv`, `sklearn.model_selection.train_test_split(test_size=0.2, stratify=y, random_state=42)`: 7,903 train / 1,976 validation images. Not the grouped split of `codebase/`. |
| Full-data runs | E10a, E10b, E14a, E14b retrain a fixed recipe on all 9,879 images; they have no validation score by design (scored on Kaggle only). |
| Preprocessing | `Image.convert("RGBA")`, alpha-composited on a white background, resized to a square (bicubic unless stated), scaled to [0, 1], ImageNet mean/std. E2 is the only run with the default `convert("RGB")`. |
| Training | cross-entropy, AdamW (weight decay 1e-4), OneCycle (15% warm-up, cosine), seed 42, no augmentation unless stated, last-epoch model. Apple M4, PyTorch MPS. |
| Backbones | `timm` 1.0.30 with ImageNet weights: `resnet18`, `efficientnet_b0`, `convnext_tiny`. |
| Ensembles | soft vote: equal-weight average of the members' softmax probabilities. |
| Metrics | accuracy and macro precision on the 1,976 validation images. |

## Experiments

| Id | Model | Method | Input | Epochs, lr, batch | Val. acc. | Public | Private |
|---|---|---|---|---|---|---|---|
| E1a | random forest, 300 trees | file metadata only: width, height, PNG mode, RGB under transparent pixels | | | 65.84% | | |
| E1b | RBF SVM, C=10 | 512-bin RGB histogram + HOG (9 orientations, 8x8 cells, 2x2 blocks) | 64 | | 81.28% | | |
| E2 | simple CNN (3 blocks, 32-64-128, 0.29 M params) | default `convert("RGB")` | 96 | 15, 1e-3, 64 | 92.11% | | |
| E3 | same CNN | composited on white | 96 | 15, 1e-3, 64 | 93.62% | 0.93642 | 0.94331 |
| E4 | ResNet18, frozen backbone | linear head only (3,591 trainable params) | 128 | 10, 3e-3, 64 | 77.48% | | |
| E5 | ResNet18 | full fine-tuning | 128 | 10, 5e-4, 64 | 93.72% | | |
| E6 | EfficientNet-B0 | full fine-tuning | 128 | 10, 1e-3, 64 | 93.12% | | |
| E6b | ConvNeXt-Tiny | full fine-tuning | 128 | 10, 2e-4, 32 | 95.90% | | |
| E7 | ConvNeXt-Tiny | E6b + h-flip, rotation 10 deg, 5% shift, scale 0.9-1.1, label smoothing 0.1 | 128 | 15, 2e-4, 32 | 94.38% | | |
| E8 | ConvNeXt-Tiny | E6b at higher resolution | 224 | 10, 2e-4, 32 | 97.57% | 0.97408 | 0.97631 |
| E9 | ensemble E6b + E8 | soft vote | | | 97.93% | 0.97671 | 0.97874 |
| E10a | ConvNeXt-Tiny | E6b recipe on all data | 128 | 10, 2e-4, 32 | n/a | | |
| E10b | ConvNeXt-Tiny | E8 recipe on all data | 224 | 10, 2e-4, 32 | n/a | 0.97833 | 0.97773 |
| E11 | ensemble E10a + E10b | E9 recipe on all data | | | n/a | 0.97914 | 0.97793 |
| E12a | ConvNeXt-Tiny | control for E12b: bicubic resize | 144 | 10, 2e-4, 32 | 96.31% | | |
| E12b | ConvNeXt-Tiny | **nearest-neighbour** resize (144 = 2 x 72) | 144 | 10, 2e-4, 32 | 97.22% | | |
| E13 | ConvNeXt-Tiny | nearest-neighbour resize (216 = 3 x 72) | 216 | 10, 2e-4, 32 | 97.47% | | |
| E13v | ensemble E8 + E12b + E13 | soft vote | | | 98.13% | 0.97772 | 0.97955 |
| E14a | ConvNeXt-Tiny | E12b recipe on all data | 144 | 10, 2e-4, 32 | n/a | | |
| E14b | ConvNeXt-Tiny | E13 recipe on all data | 216 | 10, 2e-4, 32 | n/a | | |
| **E15** | ensemble **E10b + E14a + E14b** | E13v recipe on all data: 224 bicubic + 144 nearest + 216 nearest | | | n/a | 0.98117 | **0.98238** |
| E16 | ConvNeXt-Tiny | E12b + 9,046 pseudo-labelled test images (E13v confidence >= 0.99) | 144 | 10, 2e-4, 32 | 97.72% | | |

Validation accuracies are single runs on the 80/20 split above (about 0.4 points standard error).
Public scores are from `members/kaggle_submissions_ledger_2026-10-18.csv`, private scores from the
paper's Kaggle table. E15 Kaggle file: `E15_ens_full_3models.csv`; E13v: `E13v_ens_val_3models.csv`
(all seven submitted files, E3, E8, E9, E10b, E11, E13v and E15, are in `experiment-series-E1-E16/submissions/`; the written plan approved before any code is `experiment-series-E1-E16/FEATURE_PLAN.md`).

## What E12a, E12b and E13 showed (nearest-neighbour resizing)

The error analysis of E9 found that 32 of its 41 validation errors are 72 px files. E12a and E12b
are a controlled A/B test at 144 px that changes only the resize filter:

| Run | All | 72 px files (n=1,178) | 160/240 px files (n=798) |
|---|---|---|---|
| E12a, 144 px bicubic | 96.31% | 96.69% | 95.74% |
| E12b, 144 px nearest | 97.22% | 97.11% | 97.37% |
| E8, 224 px bicubic | 97.57% | 96.60% | 99.00% |
| E13, 216 px nearest | 97.47% | 96.43% | 99.00% |

Nearest-neighbour helps at 144 px (+0.9 points), but mostly on the large files: bicubic
*downscaling* from 240 to 144 px also blurs the crisp edge pixels. At 216 px it no longer matters
(E13 equals E8 within noise), and the 72 px files stay the weak spot of every model. E12b and E13
were kept as ensemble members because they look at the image differently from E8.

## Other findings recorded here

- **Pseudo-labelling (E16).** At confidence >= 0.99 the E13v ensemble covers 91.5% of the validation
  images at 99.7% accuracy; on the test set this selects 9,046 images. Added to the E12b recipe they
  raise the single model from 97.22% to 97.72%, but add nothing inside the 3-model ensemble (98.13%
  with or without E16): the labels come from that ensemble, so E16 learns to imitate it. Not submitted.
- **Metadata stacking.** Mixing the E1a model into the E9 vote (weights 0.1 to 0.5) changes validation
  accuracy by at most one image (97.93% to 97.98%).
- **Duplicates.** 349 test images are exact pixel copies of training images; E15 already predicts all of
  them correctly.
- **Weak ensemble members hurt.** E5 + E6b + E8 gives 97.32%, below E9 (97.93%).

## Reproduce

```bash
cd members/erdem-sanver/experiment-series-E1-E16
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
mkdir -p data && cp -R <kaggle archive>/{train,test,train_labels.csv} data/   # images are not in the repo
./run_experiments.sh                    # list the experiment ids
./run_experiments.sh E12a E12b          # writes experiments/, submissions/, logs/
.venv/bin/python -m src.results         # rebuild the spreadsheet from the JSON records
.venv/bin/python -m src.analysis        # every analysis number in this README (errors, size split, ensembles, leaks, pseudo-labels, figures)
.venv/bin/python -m pytest tests -q     # 15 unit tests
```

Training on Apple MPS is not bit-exact, so a rerun can differ by about 0.2 points.
Model weights were not kept; the saved probabilities in `experiments/preds/` are enough to rebuild
every ensemble and submission file (`python -m src.ensemble <id> <members...>`).

## AI use

All code, experiments and write-ups in this folder were produced with Claude Code (Anthropic,
Claude Opus) at Erdem's direction; see `codebase/AI_USAGE.md` for the record and the prompts.
