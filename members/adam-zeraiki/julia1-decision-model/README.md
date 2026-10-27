# JEV-style decision experiment — complete replay archive

This archive contains the finished **Julia-1 decision-layer experiment** for the emoji-provider classification project. Julia-1 is an open pretrained typed decision model published by Supersonic Labs. **It is distinct from TypeSafe's proprietary Jev**, which was not called in this experiment. No API key, paid inference, or Kaggle upload was used.

Julia receives the saved probability estimates from three image classifiers: DINOv2, ConvNeXt and a colour/edge/texture style classifier. It receives seven caller-supplied provider choices and a fixed question. This archive includes the exact expert inputs, model code, tokenizer, requests, predictions, diagnostics and provenance needed to replay that decision layer. The public 577,189,056-byte model checkpoint is fetched separately using the included helper, which verifies its pinned revision, byte size and SHA256 before installing it.

## Result and submission recommendation

All three results below use the same 1,976-image development holdout. That holdout had already influenced earlier image-model choices, so these are exploratory validation results, not an untouched test estimate or a Kaggle score.

| Final decision method | Correct | Validation accuracy |
|---|---:|---:|
| Existing equal DINOv2/ConvNeXt ensemble | 1,919 / 1,976 | **97.12%** |
| Julia only when the two neural experts disagree | 1,857 / 1,976 | **93.98%** |
| Julia on every image | 131 / 1,976 | **6.63%** |

**The Julia hybrid CSV is experimental and is not recommended for Kaggle.** The hybrid fixed three baseline errors but introduced 65 new errors. The existing equal neural ensemble remains the stronger candidate in this comparison.

The final experimental CSV is:

`jev/test_hybrid/submission_julia_decision_hybrid.csv`

It contains all 9,879 test IDs. Julia processed only the 574 cases where the neural experts disagreed; the remaining 9,305 rows retain the equal neural ensemble. The bounded test pass took 137.97 seconds with two CPU threads. The earlier full validation pass took 495.13 seconds with two CPU threads. Both sets of outputs passed independent ID, class, probability, request and routing checks.

The unusually poor standalone result was checked against the upstream engine on nine existing requests spanning all seven predicted classes. Its raw choice index, logit argmax and exported argmax agreed in every case, and decoded option markers matched the class order. The outputs varied but strongly favoured `google` and `messenger`. This audit found no class-index export or constant-output fault; it does not establish the cause of the model's bias on this fixed numeric score-state task. The prompt and weights were left unchanged.

## Contents

- `jev/`: 92 files from the original 93-file experiment allowlist, copied unchanged, plus its original `package_manifest.json`. The only omitted file is the publicly downloadable `julia_model/model.safetensors`. The tokenizer, upstream code, Apache 2.0 license, fixed protocol, code, requirements, AI disclosure, validation results, diagnostics and experimental test CSV are all retained.
- `expert_inputs/`: the eleven exact source files needed by `decision_pipeline.load_experts`, with their original directory structure. Four are `.npy` probability arrays, four are CSV files, and three are JSON ID/class manifests.
- `fetch_model_weights.py`: a Python standard-library downloader for the exact official public checkpoint. It streams to a temporary file, checks the expected size and SHA256, and atomically installs the file only after both checks pass. No credential or account is required.
- `release_manifest.json`: SHA256 and byte size for every staged file other than the release manifest itself, plus an explicit record of the one externally required weight file. The original `jev/package_manifest.json` and other historical manifests remain unchanged and continue to record the weight used in the completed experiment.

The original `jev/README.md` is retained byte-for-byte as part of that historical experiment record. For the smaller release archive's setup, follow this top-level README, including the weight-fetch step below.

The decisive result records are `jev/validation/central_comparison.json`, `jev/validation/COMPLETE.json`, `jev/diagnostics/diagnostics.json` and `jev/test_hybrid/COMPLETE.json`. The test completion record explicitly marks the CSV as experimental and not recommended for Kaggle.

## Install the recorded runtime

The commands below use a POSIX shell and Python 3.12. Start in the extracted archive's top-level directory, the directory containing this README and `expert_inputs/`.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install pandas==2.2.3 numpy==2.3.5
python fetch_model_weights.py
cd jev
python -m pip install --target ./vendor -r requirements_jev_runtime.txt
```

Dependency installation and the one-time checkpoint fetch need internet access. The downloader uses the official Hugging Face URL at revision `a85b127321d580d65176c89ced8273f305745d85` and places the verified file at `jev/julia_model/model.safetensors`, regardless of the shell's current directory. It leaves an existing matching checkpoint in place and refuses to replace an existing different file. The tokenizer and model source are already included. After the fetch, inference loads local model files and sets Hugging Face and Transformers offline flags. `vendor/` is created by the final install command so Julia's Transformers 5.0 dependencies remain isolated from other project environments.

**All remaining commands assume the current directory is `jev/`.** Consequently, `--package ../expert_inputs` resolves to the sibling source-input directory inside this archive. Outputs under `../replay/` are separate from the preserved original results.

## Replay validation

```bash
python decision_pipeline.py --package ../expert_inputs --out ../replay/validation --split validation --threads 2 --batch-size 8
OPENBLAS_NUM_THREADS=1 python verify_decision_outputs.py --package ../expert_inputs --run ../replay/validation
```

This produces Julia probabilities for all 1,976 validation images and the predeclared disagreement-only hybrid. The verifier checks IDs, exact requests, completion, softmax and routing, then writes `COMPLETE.json`. The fixed seven-column order is `apple`, `facebook`, `google`, `messenger`, `mozilla`, `samsung`, `whatsapp`.

## Replay the experimental test CSV

```bash
python decision_pipeline.py --package ../expert_inputs --out ../replay/test_hybrid --split test --disagreements-only --threads 2 --batch-size 8
OPENBLAS_NUM_THREADS=1 python verify_test_submission.py --package ../expert_inputs --run ../replay/test_hybrid
```

The reproduced CSV is `../replay/test_hybrid/submission_julia_decision_hybrid.csv`. In a routed test run, `ids.json` and `probabilities.npy` describe the 574 Julia requests; `hybrid_ids.json` and `hybrid_probabilities.npy` describe all 9,879 final rows. No standalone full-test Julia pass is required. These commands generate local files and do not submit anything to Kaggle.

To inspect the already recorded outputs without running the model again:

```bash
OPENBLAS_NUM_THREADS=1 python verify_decision_outputs.py --package ../expert_inputs --run ./validation
OPENBLAS_NUM_THREADS=1 python verify_test_submission.py --package ../expert_inputs --run ./test_hybrid
```

## Scope, attribution and report transparency

The image experts' source probabilities are included so the decision comparison can be replayed independently. Training and inference of those image experts from raw images belong to the main `CV_Improvement_Experiments_2026-10-17` project package. No emoji labels trained Julia, selected its prompt or calibrated its outputs. The decision request contains only the task description, 21 expert probabilities, the fixed question and seven provider options; image IDs and correct labels are excluded from the encoded request.

Julia model revision: `a85b127321d580d65176c89ced8273f305745d85`.

Model-weight SHA256: `df853bf7fe424420011f3d0c47a05d7341aa9eefa7fb9f203ea4aada4ad95b72`.

The model card declares Apache 2.0. Original attribution, source files and license text are preserved in `jev/`. The initial JEPA interpretation was cancelled after the user's correction and is excluded from this archive. `jev/AI_USAGE.json` preserves the correction and AI assistance; students should write their own course report and disclose that assistance.

Primary sources: [Julia-1 official model](https://huggingface.co/SupersonicLabs/Julia-1), [TypeSafe Jev models](https://docs.typesafe.ai/models), and [TypeSafe state reference](https://docs.typesafe.ai/concepts/state).
