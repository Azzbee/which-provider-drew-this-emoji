# File-forensics shortcut model (28 October)

How far can a model go on what the PNG files reveal, without the drawing (paper section 3.2)? `png_features.py` reads 57 features from each file's raw bytes: header fields, which optional chunks are present and in what order, how the compressed data is split, zlib settings and the first Deflate block type, the row filters the encoder chose, palette and transparency chunks, file size, compression ratios, and the colour stored under fully transparent pixels. A histogram gradient-boosting classifier is scored with the same 5 grouped folds as the final model (StratifiedGroupKFold, seed 42, duplicates grouped by pixel hash).

| Model | Out-of-fold accuracy (9,879 images) |
|---|---|
| Simple metadata, random forest (paper, section 3.2) | 87.37% (stored split) |
| **Full file forensics only**, gradient boosting | **93.48%** |
| Final ConvNeXt-Tiny, pixels only | 97.69% |
| ConvNeXt-Tiny probabilities + file forensics, stacked | 97.95% (95% CI 97.66 to 98.22) |

**Why it stops at about 93%:** the dataset's files were re-encoded before release. No file has a colour profile, gamma, text, timestamp or optional colour chunk; only some Mozilla and Messenger files carry a DPI chunk; every file has exactly one data block. The encoder fingerprint that survives (PNG mode, zlib level, row filters, hidden colour) separates the export families but not Apple from Google or Samsung inside the 72 px family. The image IDs carry no order either: neighbouring IDs share a vendor 16.5% of the time, close to the 18% expected by chance.

**Conclusion:** the shortcut we chose not to use was worth about a quarter of a point on top of the pixel model, not a jump to 99%.

Files: `png_features.py` (feature extraction), `evaluate.py` (file-only models), `stack.py` (pixels + file stacking), `predict_test.py` (test predictions), `features.csv` (features for all 19,758 files, no labels), `results.json`, and two test-set CSVs that were never submitted. The stacked CSV changes 164 test labels compared with the final model, more than the out-of-fold gain suggests, because the stacker sees fold-model probabilities in training and the 5-model average at test time. To rerun: extract the Kaggle archive into `codebase/data/extracted/`, then run the scripts from this folder.
