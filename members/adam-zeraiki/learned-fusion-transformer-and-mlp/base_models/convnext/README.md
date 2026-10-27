# Complementary ConvNeXt-Tiny emoji-provider experiment

This is an additional experiment after the existing DINOv2 run. Both use the
same 7,903 training and 1,976 validation images and the same deterministic
224-pixel inputs. Every actual score must be read from the generated result
files; this document makes no claim that the new experiment has improved a
Kaggle score.

## Model and preprocessing

The checkpoint is the official TorchVision ConvNeXt-Tiny ImageNet-1K V1 model:

- URL: https://download.pytorch.org/models/convnext_tiny-983f1562.pth
- SHA256: `983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d`
- Expected file size: 114,419,221 bytes.

PNG images are decoded to RGBA, composited onto white, converted to RGB, fitted
inside a 224 × 224 square with Lanczos resampling while preserving aspect ratio,
and padded with white at the center. Pixels are divided by 255 and normalized
with the ImageNet mean and standard deviation. This is exactly the previous
DINOv2 experiment's preprocessing. It deliberately uses letterboxing instead
of the checkpoint's usual ImageNet center crop.

The first six TorchVision `features` layers stay frozen and in evaluation mode.
Their outputs have shape `(384, 14, 14)` and are stored as float16 memory-mapped
arrays. All 9,879 labeled-image outputs take approximately 1.49 GB;
test outputs require a separate cache of the same size.

## Two experiment modes

1. **Last-stage fine-tuning.** Train the final downsampling layer, final three
   ConvNeXt blocks, original output LayerNorm, and a new linear 768-to-7 head.
   The loss is ordinary cross entropy. Defaults are AdamW, learning rates
   `1e-4` for pretrained layers and `1e-3` for the head, weight decay `0.01`,
   batch size 16, CPU BF16 autocast, gradient norm clipping at 1, and seed 42.
   The epoch learning-rate multiplier follows cosine decay toward 0.1 over a
   maximum budget of 10 epochs. Early stopping uses patience 3. Selection uses
   held-out accuracy with cross entropy as the tie breaker. TorchVision's
   default stochastic-depth schedule has maximum probability 0.1; only the
   final-stage probabilities apply while that stage trains. There is no image
   augmentation and no label smoothing.

2. **Frozen features plus logistic regression.** The whole pretrained backbone
   stays frozen. Average-pooled and normalized 768-dimensional features feed
   standardized multinomial logistic regression. The C values `0.1`, `1`, and
   `10` are selected by three-fold stratified grouped cross-validation using
   only the 7,903 training images. Each fold fits its own StandardScaler using
   its training portion. Duplicate groups combine the supplied RGB96 hashes
   with exact hashes of the actual RGB224 inputs, including transitive matches.
   The held-out 1,976 images are evaluated once after C selection.

Neither mode performs all-label refitting. Neither uses hidden test labels,
test-image matching, or automated Kaggle uploads. The result metadata reports
whether a candidate gains at least eight additional correct predictions over
the earlier DINOv2 model's 1,895/1,976 result. This is a practical model-choice
threshold, not a statistical significance test.

## Reproduce

The reference run uses Python 3.12, CPU PyTorch 2.8.0, and TorchVision 0.23.0.
Install CPU packages first, then the remaining pinned dependencies:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Keep the complete dataset with `train_labels.csv`, `train/*.png`, and
`test/*.png`. Keep the supplied split and image-hash manifests from the earlier
DINOv2 experiment. Download the official checkpoint from the URL above; the
script verifies its complete SHA256 before loading it.

In Bash, configure paths once:

```bash
common=(--data /path/to/dataset --split /path/to/split.csv --hashes /path/to/image_hashes.csv --checkpoint /path/to/convnext_tiny-983f1562.pth --out /path/to/convnext_run --threads 8)
python train_convnext.py "${common[@]}" --stage audit
python train_convnext.py "${common[@]}" --stage benchmark
python train_convnext.py "${common[@]}" --stage cache --cache-set train
```

The benchmark checks prefix/suffix equivalence against the full pretrained
feature pipeline on real input images, in FP32 and BF16, and separately reports
the effect of storing the prefix as float16. It also estimates feature-cache
runtime. CPU BF16 is intended for hardware with native support.

Run either or both candidates:

```bash
python train_convnext.py "${common[@]}" --stage train
python train_convnext.py "${common[@]}" --stage logreg
```

The fine-tuning run saves resumable `selection/last.pt`, selected
`selection/best.pt`, initial pretrained suffix and head states, epoch history,
validation probabilities with IDs, metrics, configuration, and evidence of
nonzero backbone gradients and actual pretrained parameter changes. Repeating
the same train command resumes any interrupted run. Changed training settings
require a fresh output directory.

The frozen mode saves `logreg/model.npz`, inner fold assignments, C-selection
scores, configuration, and held-out probabilities and metrics. Its NPZ model
contains scaling statistics, coefficients, intercepts, and class order; no
pickled estimator is needed for inference.

After assessing validation results, generate a candidate CSV:

```bash
python train_convnext.py "${common[@]}" --stage predict --method finetune
python train_convnext.py "${common[@]}" --stage predict --method logreg
```

Prediction generates the test prefix cache if needed. The two CSV names are
`submission_convnext_finetuned.csv` and
`submission_convnext_frozen_logreg.csv`. Both contain exactly `Id,Label`, one
row for each test ID 10001 through 19879 in order. Each has a receipt with
checkpoint and CSV hashes. The script only writes the files; it does not upload
them to Kaggle.

## AI disclosure

OpenAI Codex prepared this experiment, its preprocessing implementation, model
adaptation, training loop, experiment selection, and output checks. The user
requested continued experimentation after the team's five daily Kaggle
submissions were exhausted. Preserve the actual conversation prompts and all
experiment receipts with the final codebase. The students must write their own
course report and explain the approach they choose to submit.
