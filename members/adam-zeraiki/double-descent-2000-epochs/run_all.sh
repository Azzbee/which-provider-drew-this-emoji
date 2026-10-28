#!/bin/bash
# Runs on the RunPod GPU. Epoch-wise: width 64 at 0/10/20% label noise. Model-wise: widths 4..32 at 20%.
cd "$(dirname "$0")"
#cat emoji32.npz.part* > emoji32.npz
mkdir -p logs
for n in 0.0 0.1 0.2; do nohup python train_dd.py --width 64 --noise $n --epochs 2000 > logs/run_w64_n$n.out 2>&1 & done
for w in 4 8 16 32; do nohup python train_dd.py --width $w --noise 0.2 --epochs 2000 > logs/run_w${w}_n0.2.out 2>&1 & done
wait
