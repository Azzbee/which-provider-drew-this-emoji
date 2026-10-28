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
