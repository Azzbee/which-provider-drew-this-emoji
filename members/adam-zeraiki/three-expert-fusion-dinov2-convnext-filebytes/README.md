# Three-expert fusion: DINOv2 + ConvNeXt + file-byte model (28 October)

Adam's multimodal idea, rerun with a different third expert: instead of hand-crafted style features, the third expert is the file-byte model of `../file-forensics-shortcut-model/`, which never looks at the pixels. Two image experts look at what is drawn; the third looks at how the PNG file was encoded. The question is which way of combining them works best, including Jev as the final decision-maker.

## Setup

All numbers are on the DINOv2 branch's 1,976 development images (exact duplicates never straddle its 7,903 / 1,976 split).

| Expert | Trained on | Dev accuracy |
|---|---|---|
| DINOv2 ViT-S/14, last 2 blocks fine-tuned (saved by the DINOv2 branch) | 7,903 | 95.90% |
| ConvNeXt-Tiny, partially fine-tuned (same branch) | 7,903 | 95.55% |
| ConvNeXt-Tiny run B, the final model (its own out-of-fold predictions) | 5-fold CV | 97.87% |
| File-byte model: HistGradientBoosting on 64 PNG byte features, trained here | 7,903 | 93.67% |

Fixed rules need no training. Learned combiners are trained and scored by grouped 5-fold cross-validation inside the 1,976 images, repeated over 5 seeds (mean shown). Jev rows replay the cached Jev answers from 18 October.

## Results: DINOv2 + branch ConvNeXt + file-byte (`results.json`, set A)

| Combiner | Accuracy | Fixed | New errors |
|---|---|---|---|
| DINOv2 + ConvNeXt equal average (reference) | 97.12% | | |
| All three, equal average | 97.57% | 18 | 9 |
| **All three, geometric mean** | **97.98%** | 21 | 4 |
| All three, majority vote on one-hot labels | 97.47% | 17 | 10 |
| Agreement router (keep if the two image models agree, else the file-byte model picks between them) | 97.32% | 15 | 11 |
| Agreement router (else three-way average) | 97.52% | 17 | 9 |
| Weighted average, weights learned | 97.46% | 18 | 10 |
| **Logistic regression on the 21 log-probabilities (stacking)** | **98.12%** | 23 | 3 |
| Logistic regression on one-hot labels only | 97.25% | 19 | 15 |
| Logistic regression on one-hot labels + agreement features | 97.02% | 15 | 19 |
| Logistic regression on log-probabilities + agreement features | 97.86% | 18 | 3 |
| Gradient boosting on log-probabilities + agreement features | 97.65% | 21 | 13 |
| Jev final decision on the 112 image-model disagreements (cached, 18 Oct) | 95.65% | 8 | 37 |
| Majority of DINOv2, ConvNeXt, file-byte and Jev-image on those 112 | 97.42% | 15 | 9 |
| File-byte model breaks the tie; Jev decides when it sides with neither | 97.37% | 15 | 10 |
| Oracle: right whenever DINOv2 or ConvNeXt is right (bound) | 98.28% | 23 | 0 |
| Oracle: right whenever any of the three is right (bound) | 99.09% | 39 | 0 |

What it shows:

1. **The file-byte model is the useful third expert that style features were not.** With style features, no combiner beat the plain average (paper, Table "Fusion and decision models"). With the file-byte model, a geometric mean with no training at all gains 17 images net, more than double the +8 gate fixed in advance. Its errors are different in kind: it fails on images that were encoded alike, not on images that look alike.
2. **Probabilities beat one-hot votes.** Every combiner given only labels (majority vote, one-hot logistic regression, agreement routers) lost most of the gain. How sure each expert is matters as much as what it says.
3. **Simple beats elaborate.** The geometric mean (no parameters) is within 3 images of the best learned stacker; gradient boosting on the same inputs does worse than both.
4. **Jev never helped.** Every Jev configuration, from 18 October and the three new ones below, stays below the three-expert geometric mean and even below the two-model average.
5. **With the strong final ConvNeXt (set B), the gain shrinks** to 2 to 3 images (98.08% reference, 98.18% geometric mean, 98.22% stacking): the file-byte model mainly fixes errors the weaker branch ConvNeXt makes and run B already gets right.

## Jev with three experts (`jev_three_experts.py`)

Jev has never seen the file-byte model, so this needs new API calls. It is consulted only on the 205 development images where the three experts are not unanimous (on the other 1,771 the unanimous answer is 99.32% correct). Three encodings are compared: `onehot` (each expert's answer as a one-hot vector), `agreement` (only the proposed labels and who proposed them; Jev must choose among the proposals) and `probs` (full distributions). Each request also carries the query image and the same seven labelled reference images as on 18 October. 615 requests ($0.064 when run).

```bash
python jev_three_experts.py dryrun          # builds and validates all 615 payloads, sends nothing
export TYPESAFE_API_KEY=...                 # your own key, in your own shell; never written to disk
python jev_three_experts.py run --budget 5  # paid, the client caps budgets at $5, never retries
python jev_three_experts.py evaluate        # writes jev_results.json
```

Run `python jev_three_experts.py prepare` first: it rebuilds `images/` (the query and reference images, not committed because they are competition data) from `codebase/data/extracted/`. The ceiling for any decider on these 205 images is 99.09% overall.

### Results (run on 28 October from Adam's Mac with his key; 615 requests, $0.064)

| Decider on the 205 non-unanimous images | Right of 205 | Overall accuracy | Fixed | New errors |
|---|---|---|---|---|
| Geometric mean of the three experts (no Jev) | 177 | 97.98% | 21 | 4 |
| Jev, one-hot labels | 158 | 97.01% | 18 | 20 |
| Jev, proposed labels and who proposed them | 156 | 96.91% | 13 | 17 |
| Jev, full probability distributions | 153 | 96.76% | 15 | 22 |
| Any expert right (bound) | 198 | 99.09% | 39 | 0 |

Fixed and new errors are relative to the DINOv2 + ConvNeXt average (97.12%). Jev always picked one of the experts' labels and followed a two-to-one majority on 171 to 179 of the 186 such images; on the 7 to 15 per variant where it overruled the majority it was right at most once (`jev_analysis.json`). One answer (image 01814, probabilities variant) named Mozilla while giving Google the higher probability (0.39 vs 0.38); the strict client rejects such answers, so the script keeps Jev's stated choice and flags it. Answers are in `answers/`; `jev_results.json` is the evaluation. The request cache is not committed because it embeds the competition images.

## Other combination rules (`combine_rules.py`, `results_rules.json`)

Same protocol (grouped 5-fold CV inside the 1,976 development images, 5 seeds).

| Rule | Set A (branch ConvNeXt) | Set B (ConvNeXt run B) |
|---|---|---|
| Reference: two image models, equal average | 97.12% | 98.08% |
| Geometric mean of the three | 97.98% | 98.18% |
| Bayes on the experts' confusion matrices (answers only) | 97.27% | 97.95% |
| Most confident expert wins, after per-expert temperature calibration | 97.93% | **98.49%** |
| Per-class trust weights | 97.95% | 98.07% |
| Output codes, 3 yes/no classifiers (minimum, "poisoned cups") | 96.98% | 97.80% |
| Output codes, 7 random bits | 97.88% | 98.12% |
| Output codes, 15 random bits | 98.05% | 98.33% |
| Output codes, all 63 bits | 98.09% | 98.24% |

The minimum code (3 classifiers for 7 providers, like finding a poisoned cup with the fewest tasters) is the worst rule: with no spare bits, one wrong classifier gives a wrong provider. Spare bits correct such mistakes. In set B the most-confident rule reaches 98.49% (1,946 of 1,976): 16 of run B's errors fixed, 4 new (exact McNemar p = 0.012).

## Test-set file

DINOv2's test probabilities were not saved, only its labels, so the test file uses DINOv2 as a smoothed one-hot vector (0.9 on its label), the final ConvNeXt run B probabilities and the file-byte model refitted on all 9,879 images, combined by geometric mean. The same recipe scores 98.08% on the development images (run B alone 97.87%). `submission_dinov2_convnext_filebytes_geomean.csv` changes 152 of 9,879 test labels relative to the final model. It was not submitted to Kaggle.

## Files

`build_experts.py` collects the experts' development probabilities into `experts_dev.npz`; `combine.py` runs every combiner and writes `results.json`; `jev_three_experts.py` and `jev_client.py` (unchanged copy of the 18 October budgeted client) run the Jev part; `make_test_submission.py` writes the test file.

Adam's experiment, 28 October 2026, with code written with Claude's help and reviewed by Adam; see `codebase/AI_USAGE.md`.
