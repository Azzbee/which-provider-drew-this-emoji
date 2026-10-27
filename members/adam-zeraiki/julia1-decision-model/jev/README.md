# JEV-style decision-model comparison

This experiment uses **Julia-1 by Supersonic Labs**, an open pretrained typed decision model. It is not TypeSafe's proprietary Jev. Julia-1 takes a state, a question and caller-supplied choices, then scores the choices in one forward pass. We use it as a final classifier over three existing image models' probability estimates.

The actual Jev service was not called: the available environment had neither a `JEV_API_KEY` nor a `TYPESAFE_API_KEY`. No signup, paid inference, or Kaggle upload occurred. A previous interpretation of "jev" as JEPA was cancelled after the user clarified the intended decision models; that branch is not part of this experiment.

## What was fixed before evaluation

`experiment_plan.json` contains the complete fixed question and protocol. Every model request contains only:

- The task description, "emoji rendering provider classification".
- Seven provider probabilities from each of DINOv2, ConvNeXt and the colour/edge/texture histogram-boosting classifier, rounded to four decimal places.
- One question asking which provider is most likely, and the seven provider names as choices.

No image IDs, file paths, correct labels, model validation accuracies or validation-derived heuristics enter the request. An identifier is stored outside the request only to align the outputs. The 21 scores are the entire image evidence supplied to Julia: this is a decision-layer test, not a new image encoder or a test of Julia seeing the pixels.

The model, prompt, option order and rule are frozen. There is no emoji fine-tuning, prompt search, learned temperature or validation fitting.

Two candidates were declared before scores were computed:

1. Julia's predictions on all 1,976 validation images.
2. A fixed hybrid that uses Julia only where DINOv2 and ConvNeXt disagree on the top label. On other images, the existing equal DINOv2/ConvNeXt probabilities are retained. This mask depends only on predictions, never on the correct label.

The original development holdout has already been used for earlier image-model choices. Results are exploratory development evidence, not a fresh test-set or Kaggle score. Returned probabilities have not been calibrated on this emoji task.

## Observed result and recommendation

The central aligned evaluator produced these results on the same 1,976-image development holdout:

| Decision method | Correct | Accuracy |
|---|---:|---:|
| Existing equal DINOv2/ConvNeXt ensemble | 1,919 / 1,976 | 97.12% |
| Julia-1 decides every image | 131 / 1,976 | 6.63% |
| Julia-1 decides only neural-expert disagreements | 1,857 / 1,976 | 93.98% |

The hybrid fixed three baseline mistakes but introduced 65 new mistakes, a net loss of 62 correct predictions. **This experiment did not improve the classifier. Its test CSV is retained for reproducibility and is not recommended for Kaggle.** The existing equal neural ensemble remains the stronger candidate in this comparison. `validation/central_comparison.json` preserves the central evaluator's metrics and source hash.

Because the standalone result was unusually poor, we audited nine existing requests spanning all seven predicted classes against the upstream engine's own `predict` method. Its option index, fresh raw-logit argmax and the exported argmax agreed in all nine cases. Decoding each option-marker segment confirmed that indices zero through six correspond to the recorded seven class names in order. Rebatching changed logits by at most 0.0000163, without changing any audited winner. We did not alter the prompt, weights or predictions.

The model was not returning one constant vector: the validation run contained 1,965 distinct logit rows. However, its winners strongly favoured two labels:

| Predicted provider | Validation predictions |
|---|---:|
| apple | 20 |
| facebook | 46 |
| google | 1,211 |
| messenger | 656 |
| mozilla | 29 |
| samsung | 1 |
| whatsapp | 13 |

This rules out an observed class-index export error and a constant-output implementation fault. It does not establish why Julia has this strong bias on the numeric score-state task. The supported conclusion concerns this frozen model and fixed prompt; it does not measure TypeSafe Jev or establish that all decision models fail at model fusion. The complete diagnostics and upstream comparison outputs are in `diagnostics/`.

## Exact model and runtime

- Model: `SupersonicLabs/Julia-1`.
- Revision: `a85b127321d580d65176c89ced8273f305745d85`.
- Checkpoint: `julia_model/model.safetensors`, 577,189,056 bytes.
- Checkpoint SHA256: `df853bf7fe424420011f3d0c47a05d7341aa9eefa7fb9f203ea4aada4ad95b72`.
- Architecture: mmBERT-small encoder, two transformer decision-head layers, option-marker scorer. There are 144,292,867 parameter entries; the three temperature-buffer entries bring checkpoint values to 144,292,870.
- Inference: upstream `TransformerEngine`, CPU float32, no task training, PyTorch SDPA, eight requests per batch and two CPU threads.
- Encoder is created through standard Transformers with `trust_remote_code=False`; weights load from safetensors with `strict=True`. The upstream Python model code is present and was read before import.
- The optional native router, compiled kernels and remote API paths are not used.
- Every request is checked against the upstream strict lossless serializer. Context limit is 1,024 and question/option head budget is 256; nothing is silently truncated.
- The model and tokenizer only load local files. The run sets Hugging Face and Transformers offline flags.

`validation/model_load_receipt.json`, the source manifest and weight receipt record the exact versions and source hashes. Native CPU inference uses the upstream direct logits-to-softmax behavior, without an extra temperature change. The complete 1,976-image validation pass took 495.13 seconds with two CPU threads, after model loading and tokenization, or 3.99 requests per second.

## Reproduce

Use Python 3.12 and the parent project's CPU PyTorch environment. Julia requires Transformers 5.0; use a separate virtual environment or install its pinned dependencies into a local directory so the old vision-model environment remains usable.

For a separate environment, install the CPU version of PyTorch and the exact pandas version first. From this directory:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install pandas==2.2.3
```

Then install the pinned decision-model dependencies and run inference. If using the existing project environment, its PyTorch 2.8.0+cpu and pandas 2.2.3 already satisfy the two requirements above.

```bash
python -m pip install --target ./vendor -r requirements_jev_runtime.txt
python decision_pipeline.py --package /path/to/CV_Improvement_Experiments_2026-10-17 --out ./validation_replay --split validation --threads 2 --batch-size 8
OPENBLAS_NUM_THREADS=1 python verify_decision_outputs.py --package /path/to/CV_Improvement_Experiments_2026-10-17 --run ./validation_replay
```

`--package` is the previous complete image-model experiment package. It supplies the recorded DINOv2, ConvNeXt and style probabilities and their ID manifests. No raw images are needed to replay this decision layer. Its original image-model scripts reproduce those expert predictions from the dataset.

To make a competition CSV from the predeclared hybrid rule, run Julia only on the test images where the two neural experts disagree:

```bash
python decision_pipeline.py --package /path/to/CV_Improvement_Experiments_2026-10-17 --out ./test_hybrid_replay --split test --disagreements-only --threads 2 --batch-size 8
OPENBLAS_NUM_THREADS=1 python verify_test_submission.py --package /path/to/CV_Improvement_Experiments_2026-10-17 --run ./test_hybrid_replay
```

This command writes all 9,879 rows in `submission_julia_decision_hybrid.csv` and uses the existing equal ensemble on the images that were not routed to Julia. Producing this file does not establish that the hybrid is better; consult the actual validation comparison before using a Kaggle submission slot.

The recorded test run was limited to the 574 test images where the two neural experts disagree; the remaining 9,305 rows retain the equal ensemble. This pass took 137.97 seconds with two CPU threads, or 4.16 requests per second. The resulting 9,879-row file under `test_hybrid/` passed independent ID, label, probability and routing verification; it is experimental and not recommended for Kaggle given the validation regression. No full standalone Julia test pass was run.

To reproduce the bounded post-evaluation implementation audit:

```bash
python inspect_decision_outputs.py --run ./validation --out ./diagnostics_replay
```

The full model, config and tokenizer are retained under `julia_model/`. If reconstructing from upstream, use the exact revision listed above and verify the weight SHA256 before inference. The model card declares Apache 2.0; its attribution and license text are retained.

## Output contract

- `ids.json`: ordered image identifiers, outside model state.
- `requests.jsonl`: exact frozen request for each image, with its external alignment ID.
- `logits.npy`: seven raw decision scores per processed image.
- `probabilities.npy`: seven softmax scores per processed image, float32.
- `complete.npy`: Boolean completion mask; incomplete arrays must not be scored.
- `predictions.csv`: IDs, predicted labels and seven probabilities.
- `prediction_receipt.json`: performance, hashes and input provenance.
- `hybrid_probabilities.npy`: the full predeclared hybrid distribution when the requested mode produces one.
- `hybrid_ids.json`: all 9,879 ordered test IDs for the full hybrid array; in a test run, `ids.json` describes only the 574 processed Julia requests.
- `hybrid_routing.json`: validation routing rule and routed IDs.
- `COMPLETE.json`: independent verification receipt, written last after full validation. Its recorded hashes must match the arrays before they are scored.

For full validation, probability arrays are `(1976, 7)` in this class order: `apple`, `facebook`, `google`, `messenger`, `mozilla`, `samsung`, `whatsapp`.

The verification command checks every request against the fixed renderer, ID and class alignment, the full completion mask, exact softmax outputs and the exact prediction-only hybrid routing. It does not compute accuracy. The parent experiment performs the final aligned comparison only after verification succeeds. `AI_USAGE.json` preserves the user's request, correction and code/analysis assistance for the course's transparency requirement. Students should write their own report.

## Primary sources

- Julia-1 official model card and files: https://huggingface.co/SupersonicLabs/Julia-1
- TypeSafe model reference: https://docs.typesafe.ai/models
- TypeSafe state reference: https://docs.typesafe.ai/concepts/state
- TypeSafe image asset SDK reference: https://docs.typesafe.ai/sdk/python/api/types/assets
- Open decision-model research: https://arxiv.org/abs/2610.17205

The TypeSafe model table lists text inputs while its SDK documents image assets. Those pages do not establish that our unavailable account could use an image-capable Jev endpoint. This experiment therefore makes no claim about measured TypeSafe Jev accuracy or image support.
