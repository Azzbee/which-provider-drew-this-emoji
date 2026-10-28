# Emoji provider classification

Predict image providers from `train_labels.csv`, using emoji rendering style.

## Improved submission

Upload **`submission_improved.csv`** from this project directory. It is also copied to `submission.csv`. The earlier CSV remains in `submission_baseline.csv` and `submission_before_improvement.csv`. A copy you previously placed elsewhere, such as your Desktop, is not updated by this code.

Measured results:

| Model | Seed-42 validation accuracy |
| --- | ---: |
| Original Extra Trees | 82.14% |
| Handcrafted style features + RBF SVM | 96.05% |
| Frozen ResNet18 descriptors + RBF SVM | 91.65% |
| Combined descriptors + RBF SVM (selected) | **96.76%** |

The selected model also scored **97.42%** on a second stratified split, seed 123, without changing its settings. These splits overlap and are not independent test sets. The user reported **0.81838** for the original Kaggle submission. The improved Kaggle score is unknown until uploaded; local validation is not a leaderboard score.

The selected model uses 1,607 color, opacity, gradient, texture and spatial statistics, plus 1,920 channel means/standard deviations from four frozen ResNet18 stages. StandardScaler and an RBF SVM (`C=10`, `gamma='scale'`) classify their combined 3,527 features. The image model is not fine-tuned. IDs and image encoding metadata are not features.

## Data layout

Download and extract the competition images. Keep `train_labels.csv`, `train/00001.png ...`, `test/10001.png ...`, and optionally the official `sample_submission.csv` together. The scripts validate IDs and submission columns. The Kaggle rules/invitation could not be accessed during development; confirm ImageNet pretrained weights are allowed. The official template takes precedence over the inferred `Id,Label` format.

Competition: https://www.kaggle.com/competitions/2-computer-vision-2026-b-sc-aidams-final-proj

## Reproduce the improved CSV locally

Use **Python 3.12–3.14**, the data above, and these code files: `improve.py`, `style_features.py`, `learned_features.py`, `requirements.txt`, `requirements-improved.txt`, and `requirements-baseline.txt`.

```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python improve.py
```

On Windows activate with `.venv\Scripts\activate`. In the existing project environment, run `.venv/bin/python improve.py`. If data are elsewhere, use `python improve.py --data-dir /path/to/data`.

The script extracts/caches descriptors, creates a stratified 80/20 split with seed 42, fits the scaler and SVM only on the training fold, saves validation metrics and predictions, then refits on all labels and exports predictions. No test labels are used. Feature extraction uses CUDA/Mac GPU when available, otherwise CPU; the SVM runs on CPU. The first run requires internet to download the official ImageNet ResNet18 weights into `artifacts/torch_cache/`. It can take several minutes or longer depending on hardware. Subsequent runs reuse descriptors.

`artifacts/improved_model.joblib` stores the final fitted scaler/SVM and feature specification. Load only trusted joblib files. Exact versions, settings, class order and a labels checksum are recorded in `artifacts/improved_metrics.json`. If you change images under the same IDs, delete `artifacts/style_*.npz` and `artifacts/learned_*.npz` to prevent stale caches. No cached features or pretrained weights need to be included in the final code hand-in; they are recreated automatically.

For the optional second-split check:

```
python improve.py --validation-seed 123 --validate-only
```

It writes separate metrics and leaves the submission intact. Seed 42 was used for model selection; seed 123 checks robustness on the same labeled dataset and is not an independent test set. Related emoji content can cross folds; semantic grouping would provide a stricter evaluation if reliable group IDs become available.

## Run the main notebook on Colab

Open **`Computer Vision.ipynb`**, select a GPU runtime, upload/extract the project code and requirements files, and set `DATA_DIR` to the extracted dataset directory. If using Google Drive, mount it with the commented lines in the setup cell. `CODE_DIR` must contain the Python helper files. Run cells in order. Restart the runtime after package installation if Colab requests it, then continue from imports. The main notebook reproduces the measured model above and displays validation errors and a confusion matrix.

All teammates must still complete the course's individual Colab run/submission requirement.

## Earlier model and development experiments

`baseline.py` reproduces the original CPU submission. Install `requirements-baseline.txt` for its tested environment and run `python baseline.py`.

`ResNet fine-tuning.ipynb` preserves the earlier optional training workflow; it has not been run end to end by the assistant. Its older environment is listed in `requirements-finetuning.txt`. It is separate from the measured frozen-feature model and does not underpin the reported improved results. Running that older notebook can overwrite `submission.csv`; use the explicitly named improved CSV to avoid confusion.

`experiment_style.py`, `experiment_learned.py`, and `experiment_combined.py` record the development search: C in {1,10,100} for each feature family, and C in {10,100} for combined features. Run from a copy of the project with data alongside scripts, in that order; the latter two require `learned_features.py train` first. Their JSON results and `artifacts/improvement_comparison.csv` preserve the measured comparisons. C=10 and C=100 tied for the combined model; C=10 was selected. `cache_style_test.py` is an optional development helper, not needed by `improve.py`.

The implementation uses the frozen-backbone transfer-learning pattern described in the [official PyTorch tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html), with custom channel statistics and an SVM classifier.

## Hand-in and AI disclosure

Write the report independently. Include actual validation/Kaggle results, explain features, scaling, the RBF SVM, model selection and limitations, and disclose prompts/AI assistance using [codebase/AI_USAGE.md](../../../codebase/AI_USAGE.md). No report prose is supplied. Have another teammate reproduce the CSV from a fresh environment using these instructions before hand-in.

## Repository packaging

The 78 MB fitted `artifacts/improved_model.joblib` is deliberately excluded from this repository. Running `python improve.py --data-dir /path/to/data` recreates the fitted scaler/SVM model and submission files. The model file was not committed. Competition images, `train_labels.csv`, pretrained weights, `torch_cache/`, and feature caches are also excluded. Download the competition data separately; the scripts recreate descriptors and download official pretrained weights as needed.

This README is named `method-README.md` in the team repository, and the shared AI disclosure is maintained in [codebase/AI_USAGE.md](../../../codebase/AI_USAGE.md). No DINOv2 files are part of this import.
