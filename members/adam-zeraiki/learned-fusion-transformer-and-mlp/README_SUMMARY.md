# Learned fusion: a transformer over three experts (17 October)

Inspired by TRIBE v2. Three frozen experts, DINOv2 (768-d), ConvNeXt-Tiny (768-d) and 1,460 style features, each standardised and projected to 64-d (linear, LayerNorm, GELU), become tokens with an identity embedding; a CLS token and one pre-norm transformer layer (4 heads, FF 128, dropout 0.15, 226,631 parameters) produce the class. Matched MLP control: 224,839 parameters. Encoders either public pretrained or task-tuned; seeds 17, 43, 101; epochs chosen on an inner 6,322 / 1,581 split (max 60, patience 8), refitted on 7,903 images, seeds averaged.

| Candidate | Development accuracy (1,976 images) |
|---|---|
| Reference: DINOv2 + ConvNeXt average | 97.12% |
| Tuned encoders + MLP | 97.01% |
| Tuned encoders + transformer | 96.36% |
| Public encoders + MLP | 95.90% |
| Public encoders + transformer | 95.75% |

No head beat the plain average or the threshold fixed in advance (1,927 correct). The two fusion CSVs here were never submitted to Kaggle. Full write-up: `RESULTS_fusion_and_julia1.md`; code: `heads.py`, `extract_features.py`, `reproduce_fusion.py`.
