# Julia-1 decision model (17 October)

The open Julia-1 model (SupersonicLabs, 144 M parameters) picks one answer from a list given a state and a question. It received the 21 class probabilities of the three experts (no pixels, no labels), zero-shot.

| Rule | Development accuracy (1,976 images) |
|---|---|
| Julia-1 decides only where DINOv2 and ConvNeXt disagree (112 images) | 93.98% (fixed 3 errors, made 65) |
| Julia-1 decides every image | 6.63% (chose Google or Messenger 94% of the time) |

Reference: 97.12%. Details in `../learned-fusion-transformer-and-mlp/RESULTS_fusion_and_julia1.md` and in this folder.
