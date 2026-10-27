# File-metadata shortcut diagnostic

Classifiers that never see a drawn pixel: logistic regression and a 300-tree random forest on file metadata (native size, PNG mode, alpha flag, file size, transparent fraction, RGB stored under transparent pixels).

| Features | Logistic regression | Random forest |
|---|---|---|
| size + mode (8) | 64.63% | 66.40% |
| + file statistics (10) | 74.13% | 84.44% |
| + hidden colour (13) | 76.20% | 87.37% |

Stored 1,979-image validation split. This is why every model in the paper composites images on white and never sees the metadata (paper section 3.2).

Files: `cpu_results.json` (all CPU results), `eda_summary.json` (file-format statistics). Code: `codebase/src/emoji_provider/metadata_model.py`, `codebase/src/emoji_provider/eda.py`.
