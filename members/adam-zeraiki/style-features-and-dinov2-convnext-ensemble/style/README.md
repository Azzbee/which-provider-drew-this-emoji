# Emoji provider style experiments

This supplementary experiment tests whether simple rendered-image style features add useful information to the previously fine-tuned DINOv2 model. It does not use Kaggle scores, source-image dimensions, filenames, file size, provider metadata, image matching, or inferred test labels. It does not refit on the validation rows.

## Data and validation

Use the competition's original `train/`, `test/`, and `train_labels.csv` dataset and the existing DINOv2 experiment's `run_224/split.csv`. The split contains 7,903 fitting images and 1,976 held-out images. The fixed class order is:

`apple, facebook, google, messenger, mozilla, samsung, whatsapp`

All feature calculations are fixed per-image operations without fitted dataset statistics. Both classifiers fit exclusively on the training partition. Histogram boosting runs its prespecified 140 iterations without early stopping. No hyperparameter sweep is performed.

The same held-out split previously selected DINOv2's epoch. Comparing these attempts and choosing blend weights on this split makes the comparison exploratory; its accuracy is not an independent final generalization estimate. Kaggle's hidden test accuracy remains unknown.

## Executed results

| Attempt | Correct / 1,976 | Accuracy | DINO errors rescued | DINO correct predictions lost |
| --- | ---: | ---: | ---: | ---: |
| Previous DINOv2 | 1,895 | 95.9008% | — | — |
| ExtraTrees style model alone | 1,674 | 84.7166% | 37 | 258 |
| Histogram boosting style model alone | 1,779 | 90.0304% | 42 | 158 |
| DINOv2 90% + ExtraTrees 10% | 1,894 | 95.8502% | 0 | 1 |
| DINOv2 75% + ExtraTrees 25% | 1,897 | 96.0020% | 3 | 1 |
| DINOv2 90% + histogram boosting 10% | 1,899 | 96.1032% | 4 | 0 |
| DINOv2 75% + histogram boosting 25% | 1,900 | 96.1538% | 5 | 0 |

The highest accuracy among these bounded attempts is the 25% histogram boosting blend. It changes 31 test predictions compared with the previous DINOv2 CSV. Its gain is five held-out images, or 0.253 percentage points; this is a small exploratory change and does not establish improved hidden-test accuracy. Neither standalone style classifier improves on DINOv2. The weaker style models nevertheless make some complementary correct predictions, which explains the small blend gains.

Feature extraction took approximately 38.5 seconds for the labeled images and 39.6 seconds for the test images. Fitting plus probability inference took 167.1 seconds for ExtraTrees and 16.8 seconds for histogram boosting. Model serialization time is separate from those fitting/inference measurements.

## Image rendering and features

Images are converted to RGBA, resized to fit a 64 × 64 canvas while preserving aspect ratio, then alpha-composited onto white. The rendered image contributes 1,460 features:

- 16 × 16 and 8 × 8 RGB grids.
- Alpha-weighted RGB and HSV color histograms, joint hue/saturation histogram, and foreground RGB summaries.
- An 8 × 8 rendered alpha map and alpha-distribution summaries.
- Unsigned gradient orientation histograms on 4 × 4 spatial cells, gradient energy and quantiles.
- Laplacian summaries and uniform rotation-invariant local binary pattern histograms, globally and in four quadrants.

The original width and height are used only for aspect-preserving rendering; they are never classifier inputs. Alpha is used as an image channel. See `results/feature_schema.json` for exact feature slices.

## Fixed classifier settings

ExtraTrees uses 400 trees, `max_features=0.5`, `min_samples_leaf=2`, and seed 42. It uses all 1,460 features and two parallel workers.

Histogram Gradient Boosting uses 140 iterations, learning rate 0.08, 15 maximum leaves, 64 histogram bins, L2 regularization 1.0, minimum leaf size 20, no early stopping, and seed 42. It uses the remaining 692 features after excluding the 16 × 16 RGB grid. Native thread pools are limited to two.

## Reproduce

The executed environment used Python **3.12.14**, NumPy **2.3.5**, Pillow **12.3.0**, SciPy **1.17.0**, scikit-learn **1.8.0**, joblib **1.5.3**, and threadpoolctl **3.6.0**. These package versions are pinned in `requirements.txt`; `environment.json` records the executed environment. **scikit-image is not used or required**: HOG and LBP are implemented directly with NumPy. No GPU, pretrained neural network, or external API is needed for the style experiment. After installing the pinned dependencies and obtaining the competition dataset, training and inference run offline.

Run the following commands from the directory containing this README and the Python scripts. The examples assume the extracted competition dataset is in `dataset/`, the prior experiment is in `DINOv2_Emoji_Experiment/`, and the supplied style models are in `results/`. Adjust these relative paths to match your extracted project layout.

```bash
python -m pip install -r requirements.txt
```

### Predict with an existing classifier

`predict_style.py` loads an existing classifier and performs inference only. It requires the test images but no training labels or validation split. For histogram boosting:

```bash
python predict_style.py \
  --dataset ./dataset \
  --model ./results/hist_gradient_boosting_model.joblib \
  --out ./style_predictions
```

Replace the model path with `./results/extra_trees_model.joblib` to run ExtraTrees. The command writes `style_predictions/submission_style.csv`, `test_probabilities.npy`, ordered `test_ids.json`, `classes.json`, and `inference_receipt.json`. This CSV contains the standalone style classifier's predictions; use the existing blend candidates or the blend evaluation script for a DINOv2 mixture.

If the optional deterministic test feature cache is available, it can avoid re-rendering all images:

```bash
python predict_style.py \
  --dataset ./dataset \
  --model ./results/hist_gradient_boosting_model.joblib \
  --features ./results/test_features.npy \
  --out ./style_predictions
```

The cache must have a `test_ids.json` sidecar in the same directory. Inference checks the exact ID range 10001–19879, probability class order, feature schema, cache dimensions and values, and a sample of 12 cached rows against freshly rendered images. Without `--features`, it computes features in batches. CPU workers and native thread pools are limited to two.

### Repeat training

```bash
python run_style_experiment.py \
  --dataset ./dataset \
  --split ./DINOv2_Emoji_Experiment/run_224/split.csv \
  --out ./new_style_results
```

Use a new output directory when changing the dataset, rendering, features, or split. The saved feature arrays are caches for the original ordered images; they should not be reused for different images.

For the optional, prespecified exploratory blends:

```bash
python evaluate_blends.py \
  --results ./new_style_results \
  --prior ./DINOv2_Emoji_Experiment/run_224
```

The blend script aligns validation IDs explicitly, checks the test ID order against the existing DINOv2 CSV, and reports weights 0.1 and 0.25 for each style classifier. It reports both corrected DINOv2 mistakes and DINOv2 correct predictions harmed by the blend.

## Saved artifacts

`results/metrics.json` and individual `*_metrics.json` files contain the executed scores, confusion matrices, timing, settings, and probability hashes. `results/provenance.json` records the input split, label file and training script hashes. The validation target, class order, and all probability row IDs are saved explicitly.

Each `*_model.joblib` stores the fitted classifier, selected feature indices, feature schema, class order and split hash. Only load these files from a trusted source. Raw feature caches can be omitted when distributing the experiment because the image rendering code regenerates them.

All `submission_*.csv` files contain `Id,Label` and one prediction per test image. Their existence does not mean they were submitted or evaluated on Kaggle. The main project should choose a small number of justified candidates rather than treating every experimental CSV as a required submission.

## AI usage

The user requested continued model improvement after the daily Kaggle submission quota had been reached. AI selected this bounded complementary experiment, wrote the feature extraction and classifier code, executed it, checked the output alignment, and reported the actual results. No Kaggle upload was made by this experiment. The instructor permits AI assistance for code and model understanding and requires disclosure; students must write their own report and understand the chosen method.

Exact user instruction: “Okay, cool. I can't upload it unfortunately because I only have five uploads a day and we've already done five. I think the team has also uploaded until we hit the limit. However, feel free to carry on and do a bunch of other things as well and see if potentially we'll have a better score or better model or etc., etc.”
