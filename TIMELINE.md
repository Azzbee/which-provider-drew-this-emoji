# Project timeline

How the project actually unfolded, reconstructed from dated records: Kaggle's submissions page (read 27 October 2026, which shows ages such as "14d ago", so those dates are approximate and marked ~), the run metadata saved by each notebook, and the codebase's own logs. This repository was created on 27 October 2026; its commit dates are real and do not reflect when the work below was done.

| Date | Who | What | Result | Source |
|---|---|---|---|---|
| 30 Sept | Adam | First-submission notebook: small CNN, 64 px, 15% hold-out; optional size-and-mode flag | about 81% val.; about 91% with the flag | `members/adam-zeraiki/first-submission-keras-cnn/` |
| 30 Sept | Adam | Baseline CNN (3 blocks, 96 px, Adam 1e-3, 20 epochs), white compositing, grouped split | 76.71% val. (1,518 / 1,979) | `codebase/data/baseline_sept30/`, `members/adam-zeraiki/baseline-cnn/` |
| ~4 Oct | Charles | First Kaggle entry ("Correcting output mistake") | 0.77874 private | Kaggle |
| 13 Oct | Erdem | Experiment series E1a to E9 (metadata RF, HOG SVM, simple CNN, ResNet18, EfficientNet-B0, ConvNeXt-Tiny 128/224, augmentation, ensemble) | best 97.93% val. (E9) | `codebase/report/reported_experiments.json` |
| ~13 Oct | Erdem | Kaggle: E3, E8, E9, E10b (E8 on all data), E11 (E9 on all data) | 0.94331 to 0.97874 private | Kaggle |
| ~14 Oct | Erdem | Kaggle: E13v (E8 + E12b + E13), **E15** (3 ConvNeXt: 224 bicubic + 144/216 nearest, all data) | **0.98238 private** (team best) | Kaggle |
| ~15 Oct | Charles | ResNet18 (ImageNet) fine-tuned on 160 px RGB | 0.96396 private | Kaggle |
| ~17 Oct | Lina | Extra-trees on pixels, colour, transparency; frozen ResNet18 + hand-crafted features with an RBF SVM | 82.14% and 96.76% val.; 0.83259 and 0.96983 private | Kaggle; `members/lina-oumhand/` |
| 17 Oct | Adam | Codebase: EDA, metadata diagnostic (87.4% without pixels), baseline re-score, EmojiCNNv2 (90.96%), fusion with metadata (93.38%) | 0.91700, 0.94615 private | `codebase/outputs/cpu/` |
| 17 Oct | Adam (RunPod RTX 4090) | ConvNeXt-Tiny 224 px, 5 grouped folds, flip TTA: run A (lowest val. loss) and run B (last epoch) | 97.29% and **97.69% OOF** | `codebase/outputs/convnext224*/`, `codebase/outputs/runpod_training_logs.txt` |
| 17 Oct | Lina | DINOv2 ViT-S/14, last 2 blocks fine-tuned | 95.90% dev.; 0.95607 private | `members/lina-oumhand/dinov2-partial-finetune/` |
| 17 Oct | Adam | Style features, partially tuned ConvNeXt, DINOv2 + ConvNeXt average; learned fusion (transformer, MLP); Julia-1 decision model | average 97.12% dev.; fusion best 97.01%; Julia-1 router 93.98% | `members/adam-zeraiki/` (style, fusion, Julia-1 folders) |
| 18 Oct | Adam | Live Jev (jev-1.14.0) decision trial on 112 disagreements | router 95.65% dev.; not submitted | `members/adam-zeraiki/jev-decision-model/` |
| 18 Oct | Adam | 10-checkpoint average (runs A + B) | 0.98198 private | `codebase/outputs/submissions/` |
| 18 Oct, 19:24 | Adam | DINOv2 + ConvNeXt equal average submitted | 0.96477 private | Kaggle; `members/adam-zeraiki/style-features-and-dinov2-convnext-ensemble/` |
| 18 Oct, 21:51 | Adam | Final model (run B) submitted; run B and E15 selected as the team's two entries | 0.98319 public, **0.98218 private** | `codebase/report/public_leaderboard.json` |
| 18 Oct, 22:00 | | Kaggle competition closes | team 5th of 18 (private) | Kaggle leaderboard |
| 18 Oct | Adam | First report draft (HTML/PDF) and shared report document | | `codebase/report/` |
| 27 Oct | Adam | LaTeX paper (background, journey, double descent, DINOv2, fusion), this repository | | `paper/` |
| 28 Oct | Adam | File-forensics shortcut model: 93.48% file-only, 97.95% stacked with ConvNeXt (not submitted) | | `members/adam-zeraiki/file-forensics-shortcut-model/` |
| 28 Oct | Adam (with Claude) | Three-expert fusion: DINOv2 + ConvNeXt + file-byte model; geometric mean 97.98%, stacking 98.12% on the 1,976 dev images (reference 97.12%) | | `members/adam-zeraiki/three-expert-fusion-dinov2-convnext-filebytes/` |
| 28 Oct | Adam (with Claude, RunPod RTX 4090) | Double descent: ResNet18 widths 4 to 64, 0/10/20% label noise, 2,000 epochs | | `members/adam-zeraiki/double-descent-2000-epochs/` |
| 28 Oct, 20:00 | everyone | Moodle hand-in: paper PDF + codebase zip | | |
