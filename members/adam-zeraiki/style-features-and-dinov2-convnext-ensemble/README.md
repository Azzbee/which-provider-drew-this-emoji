# Reproduce the selected emoji-provider submission

The selected candidate is **50% fine-tuned DINOv2 + 50% fine-tuned ConvNeXt-Tiny**, averaging seven-provider probabilities. Its local selection result is **1,919/1,976 = 97.1154%**. This is not a Kaggle score. No Kaggle upload was performed in this batch.

The ready-to-upload selected file is at the top level as `submission_dinov2_convnext_ensemble.csv`. The `style/` and component directories retain intermediate attempts; use the selected top-level CSV for the first comparison after the team's upload allowance resets.

## Tested environment

The actual runs used Python 3.12.14 on Linux, an eight-CPU quota, CPU BF16 computation, PyTorch 2.8.0+cpu and TorchVision 0.23.0+cpu. No GPU or paid training service was used. The following commands target a matching Linux/Colab environment. Numerical reproducibility on different hardware or library versions is not guaranteed.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0+cpu torchvision==0.23.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m zipfile -e 2-computer-vision-2026-b-sc-aidams-final-proj.zip dataset
python reproduce_submission.py --data ./dataset --out ./reproduced --threads 8
```

Run these commands from this package's top-level directory. Keep `selected_ensemble.json` unchanged to replay the chosen model. The dataset directory must contain `train/`, `test/` and `train_labels.csv`. The scripts validate the complete dataset split even during test inference, so retain both image directories. The original Kaggle data archive is not duplicated in this package.

The output is `reproduced/submission.csv`, alongside the blended probability array and a provenance receipt. The wrapper performs inference only. It does not train, download model weights, access Kaggle or use test labels. Both original pretrained weight files and the selected fine-tuned checkpoints are already included. Feature caches are regenerated locally; allow a few minutes and several GB of working space.

## Training again

The component READMEs document the executed recipes. Pass explicit paths when working outside the original workspace. For ConvNeXt, for example:

```bash
python convnext/train_convnext.py --stage cache --cache-set train --data ./dataset --split ./dino_original/run_224/split.csv --hashes ./dino_original/image_hashes.csv --checkpoint ./convnext/convnext_tiny-983f1562.pth --out ./convnext_retrained --threads 8
python convnext/train_convnext.py --stage train --data ./dataset --split ./dino_original/run_224/split.csv --hashes ./dino_original/image_hashes.csv --checkpoint ./convnext/convnext_tiny-983f1562.pth --out ./convnext_retrained --threads 8
python convnext/train_convnext.py --stage logreg --data ./dataset --split ./dino_original/run_224/split.csv --hashes ./dino_original/image_hashes.csv --checkpoint ./convnext/convnext_tiny-983f1562.pth --out ./convnext_retrained --threads 8
```

The final-stage training recipe uses at most 10 epochs, stopping after three non-improving validation epochs. The delivered checkpoint comes from epoch 4 of seven completed epochs. The frozen-feature logistic baseline is a separate, weaker comparator and is not part of the selected ensemble.

## Package contents

- `EXPERIMENT_RESULTS.md`: all executed comparisons, actual results and limitations.
- `METHOD_NOTES.md` and `experiment_plan.json`: experiment design and selection conventions.
- `reproduce_submission.py` and `selected_ensemble.json`: selected end-to-end inference.
- `dino_original/`: the previous Meta DINOv2 experiment, pretrained weights and selected checkpoint.
- `dino_tta/`: actual original/flip inference code and validation results.
- `convnext/`: new training code, official weights, selected checkpoint and frozen-feature baseline.
- `style/`: two pixel-style models, their code, blends and verified intermediate outputs.
- `results/` and `review/`: probability arrays, paired comparisons and independent receipts.
- `AI_USAGE.json`: actual request and AI-assistance disclosure; earlier prompts are also retained in the original DINOv2 package.

The official DINOv2 source is https://github.com/facebookresearch/dinov2, with its Apache license included. ConvNeXt weights come from https://download.pytorch.org/models/convnext_tiny-983f1562.pth and the implementation from TorchVision; its BSD license is included in `convnext/LICENSE_TORCHVISION.txt`.

The final report remains the students' responsibility. These AI-generated technical records support transparent reporting and reproducible code.
