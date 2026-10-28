# Double descent, 2,000 epochs (28 October)

A direct test of epoch-wise and model-wise double descent on the emoji data, following the protocol of Nakkiran et al. (2020, "Deep Double Descent"), run at Adam's request on his RunPod RTX 4090.

## Setup

- **Model:** pre-activation ResNet18 with stage widths k, 2k, 4k, 8k (k = 64 is the standard ResNet18, 11,170,631 parameters), trained from scratch.
- **Data:** 32 x 32 RGB, alpha composited on white (`build_data.py`), on the DINOv2 branch's grouped split: 7,903 training, 1,976 development images.
- **Training:** Adam, constant learning rate 1e-4, batch 128, random crop (pad 4) and horizontal flip, bf16 autocast, 2,000 epochs, no schedule, no early stopping (`train_dd.py`).
- **Label noise:** a fixed random 10% or 20% of the training labels replaced by a different class (same corrupted set for every width); development labels are always clean.
- **Runs:** width 64 at 0, 10 and 20% noise; widths 4, 8, 16, 32 at 20% (`run_all.sh`). All seven ran concurrently and took 70 minutes (about $1 of GPU time at $0.89/h).
- **Logged every epoch** (`logs/*.jsonl`): training loss and accuracy on the noisy labels, accuracy on the clean and on the corrupted training subsets, development loss and accuracy.

## Results

Development error, smoothed over 25 epochs, for width 64:

| Label noise | First minimum | Peak after it | Later | End (last 50 epochs) |
|---|---|---|---|---|
| 0% | none: falls throughout (22.4% at epoch 100, 20.1% at 2,000) | | | 20.4% |
| 10% | 30.5% (epoch 30) | small | 28.6% (epoch 750) | 28.4% |
| 20% | **35.3% (epoch 25)** | **39.2% (epoch 214)** | **37.3% mean over 1,000 to 1,500; lowest 36.3% (epoch 1,568)** | 37.5% |

With 20% wrong labels the curve has the epoch-wise double-descent shape: error rises while the network memorises the corrupted labels (95% memorised by epoch 200) and then descends a second time, without returning to the first minimum. Widths 32 and 16 show the same shape (second descents of 1.9 and 1.2 points). With clean labels there is no bump: error keeps falling for 1,900 epochs after the training set is fitted.

Model-wise, at 20% noise, error at the end of training falls monotonically with width: 47.3, 44.5, 40.9, 38.3 and 37.5% for k = 4, 8, 16, 32, 64; width 4 does not fit its training labels (79% train accuracy), every width from 8 up fits at least 97%. The width grid is too coarse to show a model-wise peak.

These networks start from random weights at 32 x 32 pixels, so their accuracy is far below the final model's; the experiment is about the shape of the curves. Figure: `paper/figs/fig_dd2000.pdf`, built by `paper/make_dd_figure.py` from these logs. The model checkpoints were not kept (they were on the pod's temporary disk).

Written and run by Claude (Anthropic) at Adam's request, 28 October 2026; see `codebase/AI_USAGE.md`.
