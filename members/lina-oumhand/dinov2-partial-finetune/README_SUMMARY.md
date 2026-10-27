# DINOv2 partial fine-tuning (17 October)

DINOv2 ViT-S/14 (DINOv3 weights needed an access approval, so v2 was used). First 10 of 12 transformer blocks frozen; last 2 blocks, final LayerNorm and a linear head on [CLS, mean patch token] (768-d) trained. 224 px, white compositing, Lanczos letterbox, no augmentation. Split 7,903 / 1,976 with duplicates grouped. AdamW 3e-5 (backbone), 1e-3 (head), weight decay 0.01, batch 16, cosine schedule over 15 epochs; 7 epochs ran, epoch 3 kept.

Result: 95.90% development (1,895 / 1,976, macro F1 0.959); 95.95% with flip averaging. Kaggle: 0.95586 public, 0.95607 private (`submission_dinov2_finetuned.csv`).

Original code (`train_meta.py`), configurations and results are in this folder; weights are not in the repository.
