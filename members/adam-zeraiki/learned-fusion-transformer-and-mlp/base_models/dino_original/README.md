# Meta DINOv2 emoji-provider experiment

This is a reproducible AI-assisted computer-vision experiment. It is not a student-written report.

## Completed run

The run stopped after seven epochs and selected epoch 3. Held-out accuracy was 1,895 / 1,976 = 95.9008%; macro F1 was 95.8658%. These are local validation measurements. The delivered prediction file uses this selected checkpoint without an all-data refit. Its hidden-test Kaggle score is unknown, and this workflow has not uploaded it.

The final two pretrained blocks and finalnorm contain 3,551,232 trainable parameters; the classifier adds 5,383. A nonzero backbone gradient was measured, and 30 pretrained tensors changed in the selected model. Full numerical evidence is in selection/metrics.json.

## What was trained

- Public Meta DINOv2 ViT-S/14 pretrained checkpoint, loaded through timm.
- Seven classes in this exact order: apple, facebook, google, messenger, mozilla, samsung, whatsapp.
- Alpha compositing onto white, aspect-ratio preserving Lanczos resize, white padding to 224x224, then ImageNet normalization.
- The first ten transformer blocks are frozen. Their token outputs are cached in float16 after bfloat16 CPU computation. The last two pretrained blocks, final normalization and a new classifier are trainable.
- The classifier uses the concatenated normalized CLS token and average patch token.
- No image augmentation is applied in this run.
- The included split manifest contains 7,903 training and 1,976 validation images. Exact duplicate groups are kept together. This is a single held-out split, not five-fold cross-validation, and not the missing original Adam baseline split.
- Checkpoint selection uses validation accuracy, with validation loss as the tiebreaker. The final run metadata records whether an all-labelled-data refit was performed.
- No hidden test labels or train/test duplicate lookup are used to create predictions.

DINOv3 was the user's example model. Its official weights required approval, so the assistant disclosed and used the public Meta DINOv2 model instead.

## Setup

Tested with Python 3.12 on Linux CPU. The exact core package versions are in requirements.txt and runtime_environment.json. A machine with at least 8GB RAM and sufficient disk space for the dataset and approximately 4GB of feature caches is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_pretrained.py
```

Extract the supplied competition dataset into a folder containing train/, test/, and train_labels.csv. Set DATA to that folder. The pretrained checkpoint download is public and requires no token. Its SHA256 is checked before use.

## Input and provenance files

- train_meta.py: experiment implementation.
- download_pretrained.py: downloads and verifies the original Meta checkpoint.
- image_hashes.csv: audited image hashes used for duplicate-aware grouping.
- run_224/split.csv: exact train/validation assignments from this experiment.
- pretrained_source.json: original checkpoint source and identity.
- AI_USAGE.md: AI involvement and user request.
- LICENSE_DINOV2.txt: Meta DINOv2 license.

## Reproduce the delivered predictions

The release package retains an inference checkpoint at run_224/selection/best.pt. Its optimizer state is not required for inference. The large intermediate feature caches are omitted because the script regenerates them from the original dataset.

```bash
CV_DATA="/absolute/path/to/extracted/competition_data"
python train_meta.py --stage predict \
  --data "$CV_DATA" \
  --checkpoint ./dinov2_vits14_pretrain.pth \
  --hashes ./image_hashes.csv \
  --split ./run_224/split.csv \
  --out ./run_224 \
  --model-file ./run_224/selection/best.pt \
  --resolution 224 --last-blocks 2 \
  --batch-size 16 --cache-batch-size 16 --precision bf16 --threads 8
```

The output is run_224/submission_dinov2_finetuned.csv. It contains only Id and Label, with one prediction for each of the 9,879 test images. The script does not upload to Kaggle. Training and inference were tested on CPU; cross-platform floating-point differences can occur even with matching package versions.

## Reproduce training from the original pretrained model

Use a different output folder so the delivered results remain available:

```bash
python train_meta.py --stage all \
  --data "$CV_DATA" \
  --checkpoint ./dinov2_vits14_pretrain.pth \
  --hashes ./image_hashes.csv \
  --split ./run_224/split.csv \
  --out ./reproduced_run \
  --resolution 224 --last-blocks 2 \
  --batch-size 16 --cache-batch-size 16 --precision bf16 --threads 8 \
  --epochs 15 --patience 4 --tail-lr 0.00003 --head-lr 0.001
```

This runs fixed-feature caching, supervised fine-tuning, held-out checkpoint selection and test prediction. Keep the same settings and output directory to resume an interrupted training run with --stage train. A completed cache is reused only when its provenance matches.

## What the reported validation score means

The supplied CSV uses the best checkpoint selected on the recorded holdout. No all-labelled-data refit is performed for the delivered experiment, so the held-out score refers to the exact model used for test predictions. The implementation includes an optional refit stage for future work; running it creates a different experiment and must be documented separately. This holdout differs from the missing original Adam baseline holdout, so the old baseline score is not a paired comparison.

The recorded method optimizes unweighted cross-entropy with AdamW, no label smoothing, gradient norm clipping at 1.0, and an epoch cosine learning-rate multiplier with a floor near 0.1. Settings and the selected epoch are recorded in run_224/selection/configuration.json and metrics.json.

Keep the code and AI prompts with the project submission, and write the report yourselves as required by the teacher.

## Sources

- https://github.com/facebookresearch/dinov2
- https://huggingface.co/facebook/dinov2-small
- https://github.com/huggingface/pytorch-image-models
