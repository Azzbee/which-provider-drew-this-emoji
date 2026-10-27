# Adam's experiments with ChatGPT (17 and 18 October 2026)

The DINOv2 branch of the project: a self-supervised transformer, hand-crafted style features, learned fusion of three experts, and two decision models. Paper sections 4.4 and 4.5 describe them. Each folder keeps its original code, configurations, logs and result files. Model weights, cached features, image data and API response caches are not in the repository; they are in the full 2.1 GB archive on Adam's Google Drive.

All accuracies below are on this branch's own development split: 7,903 training and 1,976 development images, exact duplicates grouped. That split was reused for several choices, so these numbers are development evidence, not untouched test estimates.

| Folder | What | Development result |
|---|---|---|
| `01_DINOv2_partial_finetune_2026-10-17/` | DINOv2 ViT-S/14, first 10 of 12 blocks frozen; last 2 blocks, final norm and a linear head on [CLS, mean patch] trained. 224 px, white compositing, no augmentation, AdamW (3e-5 backbone, 1e-3 head), batch 16, cosine, epoch 3 kept | 95.90% (1,895 / 1,976); Kaggle 0.95586 public, 0.95607 private |
| `02_style_features_partial_convnext_ensembles_2026-10-17/` | 1,460 hand-crafted style features (ExtraTrees 84.72%, histogram gradient boosting 90.03%); ConvNeXt-Tiny with only its last stage trained (95.55%); frozen ConvNeXt + logistic regression (90.64%); DINOv2 flip TTA (95.95%); DINOv2 + ConvNeXt equal average | **97.12%** (1,919 / 1,976); Kaggle 0.96881 public, 0.96477 private |
| `03_learned_fusion_transformer_mlp_2026-10-17/` | TRIBE-v2-inspired fusion: 3 expert tokens (DINOv2 768-d, ConvNeXt 768-d, style 1,460-d) + CLS, 1 transformer layer, 4 heads, 226,631 parameters; matched MLP control (224,839). Frozen public or task-tuned encoders, seeds 17/43/101 | best: tuned MLP 97.01%; tuned transformer 96.36% |
| `04_julia1_decision_model_2026-10-17/` | Open Julia-1 decision model (144 M) choosing from the 21 expert probabilities, zero-shot | 93.98% on the disagreement router; 6.63% alone |
| `05_jev_live_trial_2026-10-18/` | Hosted Jev (jev-1.14.0, TypeSafe): image expert plus final decision on the 112 DINOv2/ConvNeXt disagreements; 324 requests, $0.035 | router 95.65% (fixed 8 errors, made 37) |
| `result_notes/` | The two result write-ups produced at the time | |
| `submissions/` | The DINOv2 and DINOv2 + ConvNeXt CSVs (submitted) and the two fusion CSVs (experimental, never submitted) | |

None of the combiners beat the plain DINOv2 + ConvNeXt average, and that average itself (0.96477 private) stayed below the final fully fine-tuned ConvNeXt-Tiny ensemble (0.98218). DINOv3 was the first choice, but its weights needed an access approval, so DINOv2 was used. A JEPA branch started by mistake was cancelled and is not an experiment.
