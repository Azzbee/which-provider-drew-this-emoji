# Emoji Vendor Classification: Plan

Kaggle: `2-computer-vision-2026-b-sc-aidams-final-proj` (private, class competition)
Deadline: ~19 Oct 2026 (6 days from 13 Oct). Limit: 5 submissions/day, **shared by the team**.
Grade = leaderboard rank + **written report**. Every step must be explainable and reproducible.

## Goal
7-class classification (apple, google, whatsapp, facebook, samsung, mozilla, messenger).
Metric: accuracy. Team's current best: 0.788 (CNN from scratch). Top of the leaderboard: 0.998.

## Approach: an experiment ladder
Change ONE variable at a time. Fill one Excel row per experiment. This gives the report its
"we tried X → it moved by Y, because Z" story.

| # | Experiment | What changes | Why (for the report) |
|---|------------|--------------|----------------------|
| E0 | Data analysis | — | Class distribution, image sizes, alpha channel, label format |
| E1 | Classical ML: color histogram + HOG → SVM / LogReg | — | Pre-deep-learning reference point |
| E2 | Simple CNN from scratch (RGB, default conversion) | baseline | Reproduce the team's level (~0.78) |
| E3 | E2 + correct alpha handling (white bg / 4 channels) | preprocessing | Test the "careful preprocessing" hint |
| E4 | ResNet18 pretrained, frozen backbone + linear head | transfer learning | Effect of ImageNet features alone |
| E5 | ResNet18 full fine-tune | fine-tune vs frozen | — |
| E6 | EfficientNet-B0 / ConvNeXt-Tiny fine-tune | architecture | Effect of a stronger model |
| E7 | Best + augmentation (flip, small rotation, color jitter?) | augmentation | Careful: color is a vendor signal, jitter may hurt |
| E8 | Higher resolution (e.g. 128 → 224) | input size | Fine detail / line style |
| E9 | 5-fold CV ensemble + TTA | ensemble | Final squeeze, robustness for private LB |

Validation: stratified 80/20 hold-out (seed=42) for fast work, 5-fold for the final model.
Seed fixed and reported.

## Submission budget (team-shared, max 30 total)
We work against validation; we only submit when **val improves by a meaningful margin**.
- Day 1 (13 Oct): E0–E3, 1–2 submissions (verify format + baseline)
- Day 2: E4–E6, 2 submissions
- Day 3: E7–E8, 2 submissions
- Day 4: E9 final model, 1–2 submissions
- Day 5: Choose the 2 final submissions, finish the report
- Day 6: Buffer (no surprises at the last minute)

## Excel column interpretation (needs confirmation)
- Ratio train / Ratio Test = 0.8 / 0.2 (train/validation split)
- Train Accuracy/Precision = metrics on the training set; Test Accuracy/Precision = on the **validation** set
  (the Kaggle test set is unlabeled)
- Kaggle public score: no column → proposal: add a `Kaggle Public Score` column
- Precision = macro average

## Technical
- Local: Apple M4, 16 GB, PyTorch MPS. Small images, ~10k samples → feasible locally.
- Final notebook can also run on a Kaggle GPU for reproducibility.
- Libraries to install: `torch`, `torchvision`, `timm`, `scikit-learn`, `pandas`, `pillow`, `kaggle` (data download)
- Structure: `src/` (data, models, train, predict), `experiments/` (results per experiment as JSON),
  `results.xlsx` is generated automatically from these → the Excel is filled by code, not by hand.
- git repo, a separate commit for each experiment.

---

## Plan as executed (updated 28 October 2026)

The plan above was written and approved on 13 October, before any code. This section records how it was
carried out and what was added later; the original text above is left unchanged.

### Excel columns (decided)
Ratio train / Ratio Test = 0.8 / 0.2; "Test Accuracy / Precision" = the 20% validation split (macro
precision); a `Kaggle Public Score` column was added. Full-data runs have no validation score.

### Ladder as run (13 October, E0 to E9)
| Planned | Run as | Val. acc. | Note |
|---|---|---|---|
| E0 data analysis | EDA | | sizes, PNG modes and the colour under transparency depend on the vendor |
| E1 classical ML | E1a metadata RF, E1b HOG + colour SVM | 65.84%, 81.28% | E1a added as a shortcut diagnostic |
| E2 CNN from scratch | E2 | 92.11% | |
| E3 alpha handling | E3 white background | 93.62% | first submission, confirmed format and val = public |
| E4 ResNet18 frozen | E4 | 77.48% | |
| E5 ResNet18 fine-tune | E5 | 93.72% | |
| E6 EfficientNet-B0 / ConvNeXt-Tiny | E6, E6b | 93.12%, 95.90% | ConvNeXt-Tiny kept from here on |
| E7 augmentation | E7 | 94.38% | negative result; three changes at once |
| E8 higher resolution | E8, 224 px | 97.57% | largest single gain |
| E9 ensemble | E9, soft vote E6b + E8 | 97.93% | 5-fold CV replaced by full-data retraining (laptop compute) |

### Added after the original plan
| Date | Id | What | Why |
|---|---|---|---|
| 13 Oct | E10a, E10b, E11 | E6b, E8 and E9 retrained on 100% of the data | 25% more data once the recipe was fixed |
| 13 Oct | `--full` mode | full-data runs store no validation score | avoid a fake ~1.0 validation accuracy |
| 13 to 14 Oct | E12a, E12b | 144 px, bicubic vs nearest-neighbour (single change) | error analysis: 32 of 41 E9 errors are 72 px files |
| 14 Oct | E13 | 216 px nearest-neighbour | does the filter still matter at high resolution? |
| 14 Oct | E13v, E14a, E14b, E15 | 3-model ensemble, then its full-data version | E15: 0.98117 public, 0.98238 private, team best |
| 14 Oct | leak checks | metadata stacking, duplicates, file-id order | none gave any gain |
| 14 Oct | E16 | pseudo-labelling (9,046 test images, conf >= 0.99) | +0.5 points single model, no gain in the ensemble |
| 18 Oct | report | two LaTeX write-ups (`report/`) | for the team and the paper |
| 28 Oct | repository | this folder, `src/analysis.py`, paper rows, AI-use line | team hand-in |

### Submission budget (as used)
7 submissions over two days: E3, E8, E9, E10b, E11 (13 October) and E13v, E15 (14 October). Final
selection: E15 together with the team's final model.
