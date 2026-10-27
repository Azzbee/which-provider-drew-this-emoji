# Computer vision improvement experiments — 17 October 2026

## Selected candidate

The equal-probability ensemble of the earlier DINOv2 checkpoint and the newly fine-tuned ConvNeXt checkpoint is the strongest candidate in this bounded local comparison. It classifies **1,919 of 1,976 validation images correctly (97.1154%)**, versus **1,895 (95.9008%)** for DINOv2 alone.

The ensemble fixes 38 DINOv2 mistakes and loses 14 previously correct predictions, for a net gain of 24 images and 1.2146 percentage points. Validation errors fall from 81 to 57: a relative reduction of 29.63%. A paired bootstrap resampling whole exact-image groups gives an exploratory 95% interval of +0.5063 to +1.9578 percentage points for the accuracy gain.

These figures describe the reused local validation set. Both earlier and current choices used this holdout, so the best observed result is subject to selection bias. The interval does not remove that bias. The candidate has no measured Kaggle score and has not been submitted.

## All executed comparisons

| Candidate | Correct / 1,976 | Accuracy | Gain vs DINOv2 |
| --- | ---: | ---: | ---: |
| Original fine-tuned DINOv2 | 1895 | 95.9008% | +0 |
| DINOv2 flipped view alone | 1883 | 95.2935% | -12 |
| DINOv2 original + flip average | 1896 | 95.9514% | +1 |
| ExtraTrees image-style classifier | 1674 | 84.7166% | -221 |
| Histogram boosting image-style classifier | 1779 | 90.0304% | -116 |
| 90% DINOv2 + 10% ExtraTrees | 1894 | 95.8502% | -1 |
| 75% DINOv2 + 25% ExtraTrees | 1897 | 96.0020% | +2 |
| 90% DINOv2 + 10% histogram boosting | 1899 | 96.1032% | +4 |
| 75% DINOv2 + 25% histogram boosting | 1900 | 96.1538% | +5 |
| Frozen ConvNeXt + linear classifier | 1791 | 90.6377% | -104 |
| Fine-tuned ConvNeXt | 1888 | 95.5466% | -7 |
| 75% DINOv2 + 25% ConvNeXt | 1900 | 96.1538% | +5 |
| 50% DINOv2 + 50% ConvNeXt — selected | 1919 | 97.1154% | +24 |

## Why the combination helps

ConvNeXt alone is slightly less accurate than the existing DINOv2 model, but it correctly classifies 47 images that DINOv2 misses. Its errors are therefore partly different. Equal averaging retains 38 of those corrections while losing 14 DINO-correct cases. The fixed 75% DINO weighting does less well. No learned stacking layer or validation-fitted combiner was used.

The lightweight style models were useful diagnostics: the best style blend corrected five errors without introducing another one on this holdout. The single-model style predictors and frozen ConvNeXt baseline were substantially weaker. The horizontal-flip average gained only one image. These results remain in the package instead of being discarded from the experiment history.

## Training and provenance

DINOv2 is Meta ViT-S/14, originally fine-tuned in the previous run. Its first ten transformer blocks remain frozen; the final two blocks, final normalization and classifier were trained. The selected checkpoint is epoch 3. Its original validation probabilities were reproduced bit-for-bit before this extension.

ConvNeXt-Tiny starts from official TorchVision ImageNet-1K weights. The stem and first three stages remain frozen. The final stage, output normalization and new seven-provider classifier were trained: 15,472,128 pretrained parameters and 5,383 classifier parameters. Nonzero gradients and changed pretrained tensors were verified. AdamW used a 1e-4 initial backbone learning rate, 1e-3 head rate, weight decay 0.01, batch size 16 and a cosine schedule. The final stage retains its original stochastic-depth regularizer. Seven epochs ran; epoch 4 was selected; three subsequent non-improving epochs triggered the declared stopping rule.

Both models use 224-pixel, white-background RGBA compositing, aspect-preserving Lanczos fitting, centered padding and ImageNet normalization. Training runs on 7,903 labeled images, with the same 1,976 images held out. The exact split, provider order, hashes, settings, histories, predictions and independent review receipts are included. No all-label refit is presented as validated.

The frozen-feature classifier chooses C from 0.1, 1 and 10 using three grouped folds exclusively within the training partition. C=0.1 wins the inner comparison; its outer validation result is 1,791/1,976. Standardization is fitted inside each inner training fold. Duplicate groups union RGB96 and actual RGB224 equality relations.

## Reproducibility and AI use

The source code, official pretrained weights, tensor-identical portable selected checkpoints and standalone inference entry point are included. Only optimizer and RNG state were removed from the portable selected checkpoints; packaging receipts record the original and portable hashes and exact tensor equality. Large deterministic feature caches and raw data are excluded. Recover the original Kaggle dataset zip separately.

See README.md for a complete inference command, METHOD_NOTES.md for evaluation conventions, AI_USAGE.json for the request and assistance disclosure, and component READMEs for training commands. These files are AI-assisted technical records and code. Students should write their own final report and disclose AI use as the teacher requested.
