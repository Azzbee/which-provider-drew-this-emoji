# Emoji provider classification: fusion experiments

**Recommendation: keep the existing equal DINOv2 + ConvNeXt ensemble.** It scored 1,919/1,976 (97.12%) on the reused development holdout. The best new fusion model, a small MLP, scored 1,917/1,976 (97.01%); the requested transformer fusion scored 1,904/1,976 (96.36%). None of the new models improved the observed reference score. These are development results, not Kaggle scores.

The main archive contains actual fitted models, all four experimental submissions, exact training and inference code, source checkpoints, logs and verification records. The optional `CV_Fusion_Cached_Features_2026-10-17.zip` supplies the cached image features for fast head replay and head retraining. Extract both ZIPs into the same parent directory to merge their `CV_Fusion_Experiments_2026-10-17/` contents. Raw-image inference needs only the main archive and the original dataset. The separate `Jev_Style_Decision_Experiment_2026-10-17.zip` contains the open Julia-1 decision-model experiment. Proprietary TypeSafe Jev was not run.

## Choose a CSV

| File in `submissions/` | Role | Correct / 1,976 |
|---|---|---:|
| `submission_dinov2_convnext_ensemble.csv` | Recommended existing ensemble | 1,919 |
| `submission_tuned_style_mlp_experimental.csv` | Best new fusion; optional backup | 1,917 |
| `submission_tuned_style_transformer_experimental.csv` | Requested three-expert transformer | 1,904 |
| `submission_frozen_style_mlp_experimental.csv` | Control using original pretrained encoder features | 1,895 |
| `submission_frozen_style_transformer_experimental.csv` | Transformer control using original pretrained encoder features | 1,892 |

Each CSV has exactly 9,879 test rows, columns `Id,Label`, unique IDs 10001–19879 in the original order, and the original seven provider labels. None was uploaded to Kaggle during this experiment. The new standalone MLP and transformer files supplied alongside this archive are byte-identical copies of their respective tuned-feature CSVs.

## Installation on a fresh machine

The measured and verified environment was Linux, Python 3.12, CPU PyTorch 2.8.0 and torchvision 0.23.0. From this archive's root directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Weights are included. Inference does not download models, use a Kaggle API key or contact Kaggle. Install the dependencies before going offline. Exact floating-point equality can depend on the CPU and numerical-library build; all validation reported here used the recorded environment.

## Fastest replay: use the included image features

First extract `CV_Fusion_Cached_Features_2026-10-17.zip` beside the main archive so its `features/` directory merges into this bundle. Then run the three saved seed heads on all test features:

```bash
python heads.py predict --run runs/tuned_style --manifest tuned_style.json --output replay_tuned --threads 2
python heads.py predict --run runs/frozen_style --manifest frozen_style.json --output replay_frozen --threads 2
```

Each command creates `mlp/submission.csv` and `transformer/submission.csv` beneath its output directory, plus probabilities and an inference record. The default seed set is 17, 43 and 101; probabilities are averaged equally. No seed is chosen using the development holdout.

## Reproduce directly from the original test images

Unzip the original competition dataset yourself. The data directory must contain the unchanged `train_labels.csv` and `test/*.png`. The labels file anchors dataset identity and class mapping; training images are not read for inference. Example:

```bash
python reproduce_fusion.py --data /absolute/path/to/dataset --out replay_baseline --threads 4
python reproduce_fusion.py --data /absolute/path/to/dataset --out replay_mlp --candidate tuned_style_mlp --threads 4
python reproduce_fusion.py --data /absolute/path/to/dataset --out replay_transformer --candidate tuned_style_transformer --threads 4
```

The first command uses `selected_candidate.json`, which selects the existing ensemble. Available explicit candidates are `baseline_dinov2_convnext_equal`, `frozen_style_mlp`, `frozen_style_transformer`, `tuned_style_mlp` and `tuned_style_transformer`.

The wrapper computes features directly from images without the large temporary prefix caches used during development. It verifies original weights, source code, trained checkpoints, class order and IDs. A small optional replay can be run with `--sample-ids 10001 10779 19879`; its output is deliberately called `sample_predictions.csv`, which is not a full competition submission.

A 12-image replay of the tuned transformer was actually executed from raw images. Its DINOv2, ConvNeXt and style features, each seed's probabilities, final probabilities and labels matched the saved arrays bit-for-bit. Evidence is in `review/independent_replay_comparison.json` and `review/inference_receipt.json`.

## Refit the fusion heads

The optional cached-feature archive supplies features for all 9,879 labelled images; the main archive includes the exact split metadata. To repeat internal epoch selection and refitting from scratch, use new output directories:

```bash
python heads.py fit --manifest frozen_style.json --output retrain/frozen_style --threads 2
python heads.py fit --manifest tuned_style.json --output retrain/tuned_style --threads 2
```

These commands train both families for all three fixed seeds. They select epochs on a grouped inner split of the original 7,903 training examples, refit on those 7,903 examples and produce the corresponding test probabilities. They do not train the image encoders again. The original encoder training sources and selected checkpoints are under `base_models/`; feature extraction code is in `extract_features.py`.

The supplied heads are the actual validation-selected, 7,903-example refits used for the reported results. They were not subsequently retrained on all 9,879 labelled examples. Cached feature manifests use portable relative paths; immutable historical records retain the paths from the execution environment for provenance.

## What was implemented

Each image produces three fixed feature vectors: DINOv2 ViT-S/14 (768 dimensions: CLS plus mean patch features), ConvNeXt-Tiny (768 dimensions), and a pixel-style descriptor (1,460 dimensions). The style branch represents color, transparency and texture without filename or metadata features.

The two neural encoders are tested in two conditions: original public pretrained weights, and earlier task-fine-tuned weights. Both conditions freeze encoder weights during these fusion experiments. Each branch is standardized on the applicable training partition and projected to 64 dimensions.

The requested transformer treats the three projected branches as expert tokens, adds model identity embeddings and a classification token, and uses one attention layer with four heads. The matched control concatenates the same projected features and uses a small MLP. Their trainable parameter counts are 226,631 and 224,839 respectively, so the comparison is not simply between a large head and a tiny one.

Training uses AdamW, learning rate 0.001, weight decay 0.01, batch size 256, dropout 0.15, gradient clipping at 5, at most 60 epochs, patience 8 and a fixed 60-epoch cosine schedule. Full configuration, selected epochs, checksums, histories and checkpoints are in `runs/`.

## Evaluation and interpretation

Read `EXPERIMENT_RESULTS.md` for the complete comparison, selection limits and Julia findings. Machine-readable metrics, confusion matrices and per-example probabilities are in `results/`.

The 1,976-image holdout had already influenced earlier checkpoint and ensemble decisions. Consequently these results are exploratory development comparisons, not an unbiased estimate of unseen competition performance. The internal head validation for the tuned-encoder condition is additionally optimistic: the encoders had previously trained on all 7,903 examples. The public-pretrained condition removes that supervised encoder exposure, but does not make the reused outer holdout new.

All new model recipes, inner epoch choices and Julia prompt/routing choices were fixed before the central outer scoring pass. A more complex replacement required at least 1,927 correct predictions, eight more than the reference, under a predeclared practical rule. No candidate passed, and none improved even the raw reference score. This rule is not a statistical significance test.

To regenerate the complete central comparison after unpacking the separate Julia archive:

```bash
python evaluate_results.py --experiment . --previous base_models --julia ../Jev_Style_Decision_Experiment_2026-10-17/jev --out comparison_replay
```

## Audit trail and AI disclosure

- `experiment_plan.json`: frozen experiment design and replacement rule.
- `runs/*/selection_lock.json`: selected epochs and family decisions before refitting and outer scoring.
- `runs/*/test_prediction_record.json`: later completed test inference. Earlier fit-time `selection_summary.json` files intentionally retain their original `test_prediction_completed: false` state.
- `runs/outer_evaluation_verification.json`: independent label/ID joins, ensemble reconstruction and seed checks.
- `runs/submission_verification.json`: full CSV shape, ID, label and probability checks.
- `review/`: source, runtime and portable image replay checks.
- `AI_USAGE.json`: user prompts, assistant contributions and the Jev correction.
- `release_manifest.json`: final file sizes and SHA256 hashes.

A documentation-only correction in `heads.py` changed “No competition labels” to “No hidden test labels.” The exact executed source is retained under `runs/provenance/`; the wrapper accepts only that recorded phrase change after verifying both source hashes and executable AST equality. Training records were not rewritten.

This README and the results note are AI-generated technical experiment records. The students should write their own course report, understand the models and disclose the actual AI contribution. Original base-model provenance and relevant licenses are included. This is an image-classification adaptation inspired by TRIBE v2's fusion idea, not a reproduction of its neuroscience model.

## Primary sources

- TRIBE v2 project: https://github.com/facebookresearch/tribev2
- TRIBE v2 paper: https://arxiv.org/html/2605.04326v1
- DINOv2 project: https://github.com/facebookresearch/dinov2
- ConvNeXt project: https://github.com/facebookresearch/ConvNeXt
- TypeSafe Jev documentation: https://docs.typesafe.ai/introduction
- Open decision model actually tested: https://huggingface.co/SupersonicLabs/Julia-1
