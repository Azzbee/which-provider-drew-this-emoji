# Lina Oumhand

Explored the self-supervised Vision Transformer route (DINOv2).

## Methods tried

| Kaggle entry | Method | Validation | Public | Private |
|---|---|---|---|---|
| `submission.csv` | Extra-trees, 400 randomised trees, on resized image pixels, colour histograms and transparency | 82.14% (seed 42) | 0.81838 | 0.83259 |
| `submission2.csv` | Frozen ImageNet ResNet18 features + hand-crafted colour, transparency, edge and texture features, standardised, RBF SVM (C = 10, gamma = scale) | 96.76% (seed 42; 97.42% on seed 123) | 0.96719 | **0.96983** |
| `submission_dinov2_finetuned.csv` | DINOv2 ViT-S/14, last 2 of 12 blocks fine-tuned; files in `dinov2-partial-finetune/` | 95.90% | 0.95586 | 0.95607 |

No epochs for the first two: the extra-trees are 400 trees fitted once, ResNet18 was pretrained and frozen (no training on our images), and only the SVM was trained, until its solver converged. Each classifier was first trained on 80% of the labelled images for validation, then retrained on 100% to produce the Kaggle CSV.

## Extra Trees and frozen ResNet18 + SVM records

Code, both notebooks, requirements, the four original submission CSVs and measured outputs are in [`extra-trees-and-resnet18-svm/`](extra-trees-and-resnet18-svm/). Setup and reproduction instructions are in [`method-README.md`](extra-trees-and-resnet18-svm/method-README.md); the shared AI-use record is maintained in [codebase/AI_USAGE.md](../../codebase/AI_USAGE.md).

The Extra Trees baseline uses 400 randomised trees, 24 x 24 RGB thumbnails, colour histograms and opacity. Its seed-42 stratified 80/20 validation accuracy is **82.14%** ([saved metrics](extra-trees-and-resnet18-svm/artifacts/baseline_metrics.json)); Kaggle public/private scores are **0.81838 / 0.83259**.

The SVM ablation uses the same seed-42 80/20 split, with 7,903 training and 1,976 validation images. The scaler is fitted only on the training fold. All three rows below use an RBF SVM with **C=10, gamma=scale**, as recorded in [`artifacts/improvement_comparison.csv`](extra-trees-and-resnet18-svm/artifacts/improvement_comparison.csv).

| SVM inputs | Features | Seed-42 validation accuracy |
|---|---:|---:|
| Hand-crafted colour, opacity, edge, texture and spatial descriptors alone | 1,607 | **96.05%** |
| Frozen ImageNet ResNet18 channel means and standard deviations alone, extracted at 128 x 128 | 1,920 | **91.65%** |
| Both descriptor families, selected model | 3,527 | **96.76%** |

The combined model also reaches **97.42%** on the seed-123 stratified split without changing its settings ([saved metrics](extra-trees-and-resnet18-svm/artifacts/improved_metrics_seed123.json)). These splits overlap and are not independent test sets. The combined model's Kaggle public/private scores are **0.96719 / 0.96983**. The original improved CSV is `submission_improved.csv` (identical to the packaged `submission.csv`); it was uploaded as `submission2.csv`.

There are **no epochs** for these submissions: Extra Trees fits 400 trees; ResNet18 stays frozen; the SVM is fitted to convergence. Each classifier is validated on 80/20 and then refitted on **100% of the labelled images** for its submission. The 78 MB fitted `artifacts/improved_model.joblib` and all feature/pretrained-weight caches are intentionally omitted; `improve.py` recreates them from separately supplied competition data.

## DINOv2 partial fine-tuning

I explored whether a pretrained Vision Transformer could recognise the drawing style of the seven emoji providers. The experiment used Meta’s *DINOv2 ViT-S/14*, loaded through ⁠ timm ⁠ as ⁠ vit_small_patch14_dinov2.lvd142m ⁠. DINOv3 was initially considered, but access approval for its official weights was unavailable, so the experiment used the public DINOv2 checkpoint.

### Data and preprocessing

The dataset contained 9,879 labelled training images and 9,879 unlabelled test images. The recorded split used *7,903 training images and 1,976 validation images*, with exact duplicate groups kept on the same side. The pipeline also checked that identical inputs at the actual training resolution did not cross the split.

Each image was decoded as RGBA and composited onto white to remove the colour hidden under transparent pixels. It was resized with Lanczos while preserving its aspect ratio, centred on a white *224 × 224* canvas and normalised with the ImageNet channel means and standard deviations.

No training augmentation was used.

### What was trained

DINOv2 ViT-S/14 has 12 transformer blocks. The first ten blocks and the input embeddings stayed frozen. Their outputs were cached so that training did not repeatedly recompute the same frozen features, making the experiment practical on CPU.

The trainable part consisted of:

•⁠  ⁠The final two pretrained transformer blocks.
•⁠  ⁠The final LayerNorm.
•⁠  ⁠A new seven-class linear classifier.

The classifier received a *768-dimensional vector* formed by concatenating the final CLS token, which summarises the image, with the mean of its patch tokens.

The trainable pretrained layers contained *3,551,232 parameters, and the classifier added **5,383*. Saved gradient and weight-change measurements confirm that the pretrained layers actually changed, so this was partial backbone fine-tuning rather than training only a new classifier.

### Training and checkpoint selection

| Setting | Value |
|---|---|
| Optimiser | AdamW |
| Backbone learning rate | ⁠ 3e-5 ⁠ |
| Classifier learning rate | ⁠ 1e-3 ⁠ |
| Weight decay | ⁠ 0.01 ⁠ |
| Batch size | ⁠ 16 ⁠ |
| Loss | Unweighted cross-entropy, no label smoothing |
| Learning-rate schedule | Epoch-wise cosine decay |
| Gradient clipping | Maximum norm ⁠ 1.0 ⁠ |
| Seed | ⁠ 42 ⁠ |
| Maximum training budget | 15 epochs |
| Early-stopping patience | Four epochs without improvement |
| Computation | CPU, bfloat16 |

Checkpoint selection used the highest validation accuracy, with lower validation cross-entropy breaking ties. *Epoch 3 was selected.* Four later epochs did not improve the selection score, so training stopped after epoch 7.

After epoch 3, training accuracy continued rising while validation accuracy declined and validation loss increased. Keeping the earlier checkpoint avoided using the later, more overfit model.

### Results and submission

| Measurement | Result |
|---|---:|
| Selected epoch | 3 |
| Validation correct / total | 1,895 / 1,976 |
| Validation accuracy | *95.9008%* |
| Validation macro F1 | *0.9587* |
| Kaggle public accuracy | *0.95586* |
| Kaggle private accuracy | *0.95607* |

The validation set was used to select the checkpoint, so its score is a development result.

The selected epoch-3 checkpoint generated ⁠ submission_dinov2_finetuned.csv ⁠, containing one ⁠ Id,Label ⁠ prediction for each of the 9,879 test images. *No refit on all labelled images was performed.* The submitted model also used no flip averaging or train/test duplicate-label lookup.

The saved checks verify the CSV schema, unique test IDs, valid provider labels and agreement between exported labels and probability argmax predictions. The two committed copies of the submission are byte-identical and match the recorded submission hash.
