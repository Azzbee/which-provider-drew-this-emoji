# Jev live experiment: results and submission recommendation

**18 October 2026 — completed experiment record**

## Submission recommendation

Use **`submission_dinov2_convnext_ensemble.csv`**, the existing DINOv2/ConvNeXt submission. It contains all **9,879** test predictions with the required `Id,Label` columns, unique IDs `10001` through `19879`, and the seven allowed provider labels. An unchanged copy is included in the package's `recommended_submission/` directory.

Actual Jev inference did not improve the tested configurations. Using its final decision on model disagreements reduced the complete development result from **1,919/1,976 correct (97.1154%)** to **1,890/1,976 (95.6478%)**. The fixed three-expert blend also lost accuracy on the queried cases. Conservative mathematical bounds show that neither global candidate could catch the baseline even with ideal remaining responses. No new Jev test CSV was generated, and nothing was uploaded to Kaggle.

## What actually ran

The hosted model was pinned to **`jev-1.14.0`**. These were real TypeSafe API calls, rather than a local proxy for Jev.

| Configuration | Inputs to Jev | Role |
|---|---|---|
| Independent third expert | Query image and seven labelled training reference images | A seven-provider image classification distribution, independent of DINOv2 and ConvNeXt outputs |
| Final decision model | The same image and references, plus the complete DINOv2, ConvNeXt and independent Jev distributions | A second, dependent Jev decision using all three expert outputs |
| Fixed three-expert blend | 40% DINOv2, 40% ConvNeXt, 20% independent Jev | Deterministic probability combination |
| Disagreement router | Final Jev only where DINOv2 and ConvNeXt disagree; baseline elsewhere | Restricted final-decision configuration |

Jev was not fine-tuned, and no jointly trained local Jev transformer was created. Its final API call performed the requested decision role. The two existing vision-model distributions remained fixed for the experiment.

The technical pilot used **50 training images**, excluding reference and development duplicate groups. It checked API acceptance, response format, latency and usage. It was not used to choose prompts by accuracy. Development inference then covered exactly the **112 images** where the original DINOv2 and ConvNeXt class predictions disagreed. Both Jev stages ran for every one of those images. This gives **162 images and 324 successful inference requests** in total.

The original split contains 7,903 training images and 1,976 development images, with exact duplicates grouped apart. Seven reference images, one per provider, were selected using only the original training partition and public-pretrained DINO features. Images were composited onto white and resized with aspect ratio preserved into 280-by-280 RGB PNGs. Query labels, IDs and filenames were not sent to Jev. The prompts, references, blend weights and promotion gate were frozen before development inference.

## Observed results on the 112 difficult cases

These are deliberately difficult **subset** results. They must not be reported as whole-dataset accuracy for standalone Jev, the global blend or direct-final Jev.

| Method on the same 112 images | Correct | Accuracy | Errors corrected versus baseline | New errors |
|---|---:|---:|---:|---:|
| DINOv2 + ConvNeXt, 50:50 | **78** | **69.64%** | — | — |
| Independent image-only Jev | 23 | 20.54% | 7 | 62 |
| DINOv2 / ConvNeXt / Jev, 40:40:20 | 69 | 61.61% | 5 | 14 |
| Final Jev receiving all three distributions | 49 | 43.75% | 8 | 37 |

The independent image expert was weak on this particular provider-recognition subset. Receiving the vision-model outputs improved Jev's decisions, but the final stage still introduced substantially more errors than it corrected. The evidence concerns these specific prompts, references, weights and model version; it does not establish that every possible future Jev setup will fail.

## Complete development result for the router

The router's complete result is known without asking Jev about the other 1,864 images: keeping the baseline on those images is its definition.

| Method across all 1,976 development images | Correct | Accuracy | Negative log-likelihood |
|---|---:|---:|---:|
| Existing two-model baseline | **1,919** | **97.1154%** | **0.1376897646** |
| Jev disagreement router | 1,890 | 95.6478% | 0.1561514791 |

Jev corrected **8** baseline errors and introduced **37**, for **29 fewer correct predictions** and a decline of about **1.47 percentage points**. Both accuracy and probability loss worsened. Every loss here uses the same float64 normalization and true-class probability floor of `1e-12`; the older historical float32 loss is not substituted into this comparison.

The predeclared promotion rule required at least eight additional correct predictions—**1,927/1,976**—and no worse negative log-likelihood. The router failed it.

## Why the remaining API calls were unnecessary

This stop was supported by a conservative bound, rather than treating missing Jev predictions as completed results.

**Direct-final Jev** already had 49 correct among the 112 queried images. Even if every remaining image were correct, it could finish with at most:

\[
49 + 1,864 = 1,913.
\]

That is below both the 1,919 baseline and the 1,927 promotion threshold.

For the **fixed three-expert blend**, let \(b\) be the existing 50:50 DINOv2/ConvNeXt distribution and \(j\) be any possible Jev distribution. The blend is \(0.8b + 0.2j\). The best possible Jev distribution for a row places probability one on its true class. A baseline error can therefore be corrected only when:

\[
\max_{k\ne y} b_k - b_y \leq 0.25.
\]

The remaining 1,864 images contain 1,841 baseline-correct rows and 23 baseline errors. Only **one** of those 23 errors satisfies the condition, even counting numerical ties as potentially correct. Granting that every baseline-correct row stays correct, the blend's optimistic maximum is:

\[
69 + 1,841 + 1 = 1,911.
\]

This is also below the baseline. These numbers are **maximum possible totals**, not measured complete scores. The standalone image-only Jev diagnostic has an analogous maximum of 23 + 1,864 = 1,887.

`jev_feasibility.py` reproduces the bounds from the actual cached responses, original development labels and fixed vision-model probabilities. It conservatively includes ties within `1e-12` and records the derivation in `prepared/feasibility_receipt.json`. No further responses could make these frozen configurations eligible, so no test-set Jev inference was started.

## Actual cost and engineering record

| Item | Recorded amount |
|---|---:|
| Successful model inference requests | 324 |
| Input tokens, including images | 838,715 |
| Output tokens | 23,092 |
| Metered API usage cost | **$0.035226030** |
| Unresolved attempts or billing reservations | 0 |
| Agreed trial cap | $5 |

The cost is calculated from the provider-reported usage at the documented input price of $0.042 per million tokens, with free output tokens. It is about **3.5 US cents**, rather than a $5 charge. This is the metered usage calculation, not a statement about taxes, credit purchases or an account invoice. Sources: [TypeSafe models and pricing](https://docs.typesafe.ai/models) and [image token accounting](https://docs.typesafe.ai/images).

One pilot response contained a complete distribution rounded to hundredths that summed to 0.99. Before development inference, the client was updated to accept only bounded hundredth-grid rounding discrepancies and normalize the analysis vector. General vectors retain a strict sum check. The raw response and charge were preserved, and the response was recovered offline without paying for it again. The adapter is documented and covered by focused offline tests.

Several intermediate outcome checkpoints also needed to be regenerated from already complete response-cache entries. The development repair ran without an API key, so it could not make new requests. The cause of the incomplete intermediate files was not established; all delivered pilot and development outcomes were subsequently validated. The final ledger remains at 324 completed requests and the same cost. The private credential process was closed, and the API key is absent from the deliverables.

## Reproduce the decision without API access

After extracting the package, use Python 3.12 in its root directory:

```bash
python -m pip install -r requirements.txt
python jev_pipeline.py evaluate --run-dir prepared
python jev_feasibility.py --run-dir prepared
python jev_pipeline.py budget --run-dir prepared
python jev_pipeline.py replay --run-dir prepared
```

The completed checkpoint includes the actual requests, raw responses, token usage, spending ledger, both frozen prompts, training references, prepared images, original model probabilities and outcome records. The notebook defaults to cached replay with new API calls disabled. The evaluation and bound can be reproduced without credentials or additional paid inference.

The recommended CSV is an unchanged copy of the earlier two-model file, with SHA256:

`2b3266bbfea272f6d4e0e802d55e7f357e1d76c98dea12c43bc7433cadf43156`

These development images were reused in earlier model selection. The reported scores are exploratory development evidence, not an unbiased final-test estimate or a new Kaggle leaderboard result. Students should write the course PDF themselves and disclose the AI-assisted code, prompts and technical analysis recorded in `AI_USAGE.json`.
