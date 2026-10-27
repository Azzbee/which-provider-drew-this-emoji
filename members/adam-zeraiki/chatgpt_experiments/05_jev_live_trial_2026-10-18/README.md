# Actual Jev experiment for emoji-provider classification

## Current status

**The live experiment is complete. Keep the existing DINOv2/ConvNeXt submission.** The actual Jev API completed 324 inference requests: both stages for 50 technical-pilot images and 112 development disagreement cases. Metered cost was **$0.035226030**. The final-decision router fell from 1,919/1,976 correct (97.1154%) to 1,890/1,976 (95.6478%). The fixed third-expert blend also lost accuracy. Mathematical upper bounds show that no remaining responses could make either global candidate catch the baseline, so the remaining API calls and test deployment were stopped.

The delivered cache contains the actual requests, raw responses, usage and outcomes. The API key is absent. `Jev_Live_Results_2026-10-18.md` gives the measured results; `jev_feasibility.py` reproduces the conservative bounds. `recommended_submission/submission_dinov2_convnext_ensemble.csv` is an unchanged copy of the earlier two-model submission, not a new Jev prediction file.

The package uses the exact model ID `jev-1.14.0`. This is a hosted model called through TypeSafe's API. It supplies the final decision role requested for the architecture; there is no jointly trained local Jev transformer or customer fine-tuning of Jev.

## The two configurations

1. **Jev as a third image expert.** DINOv2 and ConvNeXt keep their existing seven-provider probability distributions. A separate Jev request sees the query image and seven labelled training reference images. Its output supplies a third seven-provider distribution. The fixed blend is `0.4 * DINOv2 + 0.4 * ConvNeXt + 0.2 * Jev`.
2. **Jev as the final decision model.** A second, dependent Jev request sees the image, the same training references, and all three distributions: DINOv2, ConvNeXt, and the first Jev pass. Its seven-class decision is evaluated directly. A restricted variant uses this final decision only when DINOv2 and ConvNeXt disagree, keeping the original ensemble elsewhere.

The control is the existing 50:50 DINOv2/ConvNeXt ensemble. Jev alone is also reported as a diagnostic. These configurations do not use the earlier Julia-1 proxy.

## Frozen evaluation protocol

The original split has 7,903 training images and 1,976 development images, with exact-duplicate groups separated. The development split has been used in previous model selection, so results here are **exploratory reused-development evidence**, not an unbiased final-test score.

The original two model predictions reproduce **1,919 correct out of 1,976 (97.1154%)**. There are **112 DINOv2/ConvNeXt disagreements**. The new models that were refitted on all 9,879 labels are used only for test-set deployment, never for evaluation on this development split.

Seven reference images are chosen from the original training partition, one per provider. Selection uses class medoids in the original public-pretrained DINO representation: normalize feature rows, compute each training-class centroid, and choose the closest eligible training row with deterministic tie handling. Development labels and model correctness are not used to choose references. The references are a small prompt aid and do not represent every emoji style.

The 50-image technical pilot also uses training data and excludes reference and development duplicate groups. It checks image acceptance, response format, latency and token usage. It does not provide a generalization score or choose a prompt by performance. Both prompts and the reference set are frozen before API evaluation.

For development, all 112 model disagreements run first, followed by the remaining 1,864 images in a deterministic order. Once all 112 routed cases have valid responses from both Jev stages, the restricted router can be evaluated over all 1,976 rows because its output elsewhere is the known control. Whole-set Jev, blend and direct-final metrics require all corresponding predictions; partial batches are not presented as complete results.

Each candidate reports accuracy, negative log-likelihood, errors corrected, new errors and net additional correct predictions. An exact paired p-value is descriptive only. The predeclared promotion gate is at least **8 net additional correct predictions** and a negative log-likelihood no worse than the control. This gate is an experiment decision rule, not a claim of statistical confirmation.

Every probability loss in this experiment uses the same explicit float64 normalization and a true-class probability floor of `1e-12`. The earlier historical loss used float32 clipping, so its slightly different number must not be substituted into the new gate.

## Inputs and reproducibility

The prepared run directory contains the frozen protocol, relative image paths, source and content hashes, the 50 pilot images, 1,976 development images, seven training references, original model probabilities for those rows, and small full-refit test probability arrays. Images are composited onto white, resized with preserved aspect ratio and centered in RGB 280-by-280 PNGs. No query label, original filename, row ID or validation score is sent to Jev.

`jev_client.py` owns authentication, the cache and the shared budget. `jev_pipeline.py` prepares inputs, constructs the two requests, compares outcomes and exports eligible test predictions. `Jev_Trial.ipynb` is an optional Colab launcher with a private runtime key prompt. `protocol.json` records both complete prompts for AI-use transparency.

In this completed checkpoint, each request and raw response is cached with its model version, content hashes and token usage. The evaluator and submission replay use those recorded responses. A reviewer can reproduce the measured Jev comparison and its infeasibility certificate from the cache without buying API access. No Jev submission was promoted or generated. Obtaining new Jev predictions still requires access to TypeSafe and the pinned model.

## Spending and credentials

The two stages, pilot, development evaluation and any deployment share **one $5 experiment cap and one API cache directory**. At the documented price, input tokens cost $0.042 per million and output tokens are free. Image tokens are included in the API's reported input usage; the vendor does not publish a dependable image-token formula, so whole-run estimates must use the measured pilot.

Before sending a request, the client reserves $0.002752512, conservatively treating the standard 64k request limit as 65,536 tokens. It replaces this reserve with actual usage when available. Uncertain billing keeps the reserve. A process lock prevents concurrent budget races. Failed or interrupted payloads are not automatically retried, and cached successful payloads are free to replay. The ceiling relies on TypeSafe honoring its documented price and standard context limit; long-context mode is disabled.

Set `TYPESAFE_API_KEY` through a private runtime environment or the notebook's `getpass` cell. Do not put it in a notebook cell, output, report, chat message, source file or ZIP. The key is sent only as the Authorization header to the official TypeSafe API. Credentials are never included in request logs. The notebook's resume download contains the experiment state and responses, not the key.

**Keep and reuse the same cache and budget ledger when resuming.** Starting a new directory or re-uploading the original zero-spend ZIP after a paid run would discard the spending history. Download the resume archive from the notebook before closing a paid session, and resume from that archive.

## Reproducing the completed result

Use Python 3.12. The tested dependencies are `numpy==2.3.5` and `Pillow==12.3.0`; no GPU is needed for this cached comparison. Extract the ZIP and open its package directory:

```bash
python -m pip install -r requirements.txt
python jev_pipeline.py evaluate --run-dir prepared
python jev_feasibility.py --run-dir prepared
python jev_pipeline.py budget --run-dir prepared
python jev_pipeline.py replay --run-dir prepared
```

These four experiment commands need no API key and make no API calls. `evaluate` gives the complete router result and records the other candidates as incomplete. `jev_feasibility.py` reports their conservative maximum possible accuracy, which is sufficient to exclude them without pretending to have all their predictions. `replay` validates the available recorded outcomes; no Jev CSV exists because no candidate passed.

The optional Colab notebook now defaults to cached replay with new API calls disabled. Upload `Jev_Trial.ipynb` to Colab, then select the accompanying ZIP when its upload cell asks. It checks the 50-image pilot and the 112 completed development cases and reproduces the feasibility certificate. No credential is required for this completed checkpoint.

The original frozen pipeline remains available for a separately justified future experiment. For this exact protocol, running the remaining 1,864 development rows cannot change candidate eligibility. Its prompts, references, weights and promotion gate were not changed after observing development results.

### Integration corrections and recovery

The standard-library HTTP transport honors the runtime proxy and certificate configuration and rejects redirects. A technical-pilot response contained seven complete probabilities rounded to hundredths that summed to 0.99. Before development inference, the client was updated to accept only bounded hundredth-grid rounding discrepancies and normalize those analysis vectors. General probability vectors retain the strict sum tolerance; raw API responses remain unchanged. `Jev_Probability_Rounding_Adapter.json` records the rule and `prepared/validation_adapter_audit/` preserves the original entry and its offline recovery. No paid request was repeated for that correction.

Several outcome files retained their intermediate expert-only checkpoints even though both API responses were durably cached. Replaying the affected partition restored these from the cache. Development recovery ran with the API key absent, and the 324-attempt count and cost stayed unchanged. The underlying cause was not established. Final checks validated all 50 pilot and 112 development outcomes before packaging.

## Test CSVs

The recommended existing CSV contains 9,879 rows with `Id,Label` columns and has SHA256 `2b3266bbfea272f6d4e0e802d55e7f357e1d76c98dea12c43bc7433cadf43156`. It is included unchanged in `recommended_submission/`. There is no new Jev CSV, no Jev test-set inference and no Kaggle upload in this experiment.

Generating a submission requires actual successful Jev predictions for every test row used by the chosen method. The file must have exactly 9,879 unique IDs from `10001` through `19879` and one of the seven allowed provider labels per row. A router can retain the baseline on nonrouted rows by definition, but missing predictions on routed rows are an error. The code does not silently convert an incomplete Jev run into a purported Jev submission.

Test images are supplied separately from the original competition dataset. The earlier all-label refits provide the two neural test distributions, but no independent validation score for that refit is claimed. Passing a development gate does not guarantee a better Kaggle score. This package does not upload anything to Kaggle.

## Course transparency

AI assistance was used to design the experiment, write the integration code and draft the prompts and technical documentation. Keep the prompts, response cache and AI-use record with the code. Students should write the course PDF report themselves, explaining which configurations actually ran, the measured outcomes and the limits of reused validation.

## Official API references

- [Images and their limits](https://docs.typesafe.ai/images)
- [HTTP request and response contract](https://docs.typesafe.ai/api)
- [Model versions, pricing and customization](https://docs.typesafe.ai/models)
- [Choice decisions](https://docs.typesafe.ai/primitives/choice)
- [Secure local setup and the TypeSafe console](https://docs.typesafe.ai/introduction/quickstart)

API documentation was checked on 18 October 2026. Experimental image behavior can change even with a pinned model version; cached responses are the exact record of a run.
