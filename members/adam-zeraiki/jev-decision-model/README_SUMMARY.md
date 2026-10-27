# Jev decision model, live trial (18 October)

TypeSafe's hosted Jev (`jev-1.14.0`), queried for the 112 development images where DINOv2 and ConvNeXt disagreed: first as an image expert (query image + one labelled reference per vendor), then as the final decision maker given all three experts' distributions. 324 requests, $0.035 of metered usage.

| On the 112 disagreements | Correct |
|---|---|
| DINOv2 + ConvNeXt average | 78 |
| Jev as image expert only | 23 |
| 40 / 40 / 20 blend with Jev | 69 |
| Jev final decision | 49 |

Over all 1,976 development images the Jev router scores 95.65% against 97.12% for the reference (fixed 8 errors, made 37). Bounds from the cached answers showed no further queries could beat the reference, so the test set was never sent. Full write-up: `Jev_Live_Results_2026-10-18.md`; code: `jev_pipeline.py`, `jev_client.py`, `jev_feasibility.py`. No API key is stored here.
