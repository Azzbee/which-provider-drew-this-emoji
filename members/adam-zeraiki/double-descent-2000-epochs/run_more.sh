#!/bin/bash
# Second batch: finer width grid around the interpolation threshold, 4,000 epochs at width 64,
# and the sample-wise test (half the data) at the widths nearest the threshold. All at 20% noise.
cd "$(dirname "$0")"; mkdir -p logs
nohup python train_dd.py --width 64 --noise 0.2 --epochs 4000 > logs/run_w64_e4000.out 2>&1 &
for w in 3 5 6 7 10 12; do nohup python train_dd.py --width $w --noise 0.2 --epochs 2000 > logs/run_w$w.out 2>&1 & done
for w in 6 8; do nohup python train_dd.py --width $w --noise 0.2 --epochs 2000 --frac 0.5 > logs/run_w${w}_f50.out 2>&1 & done
wait
