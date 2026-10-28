#!/usr/bin/env bash
# Reproduces every experiment in the report. Usage: ./run_experiments.sh E2 E3 ...  (no args = list)
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
mkdir -p logs
WHITE="Alpha-composite on white background, then resize to SIZE x SIZE (bicubic), ImageNet normalisation"
WHITE_WHY="Default RGB conversion drops alpha and exposes vendor-specific junk colours under transparent pixels; compositing on white shows the emoji as a human sees it and removes that artefact."

run() { local id=$1; shift; echo ">> $id"; "$PY" -m src.train "$@" 2>&1 | tee "logs/$id.log" | grep -E "ep +[0-9]+/|exp" ; }

case_exp() {
  case $1 in
    E1a) $PY -m src.classical meta ;;
    E1b) $PY -m src.classical hog ;;
    E2) run E2 --exp E2_simple_cnn_naive --model simple_cnn --mode naive --size 96 --epochs 15 \
          --method "Small CNN from scratch: 3 conv blocks (32-64-128 ch), AdamW lr 1e-3, OneCycle, batch 64" \
          --preproc "PIL convert('RGB') (default), resize 96x96 bicubic, ImageNet normalisation" \
          --why "Baseline reproducing the team's first approach: the most common default way to load PNGs." ;;
    E3) run E3 --exp E3_simple_cnn_white --model simple_cnn --mode white --size 96 --epochs 15 \
          --method "Same CNN and hyper-parameters as E2 (only preprocessing changes)" \
          --preproc "${WHITE//SIZE/96}" --why "$WHITE_WHY" ;;
    E4) run E4 --exp E4_resnet18_frozen --model resnet18 --pretrained --freeze --mode white --size 128 --epochs 10 --lr 3e-3 \
          --method "Transfer learning, feature extraction: ImageNet ResNet18 backbone frozen, only the linear head is trained (AdamW lr 3e-3, OneCycle)" \
          --preproc "${WHITE//SIZE/128}" --why "$WHITE_WHY 128 px: closer to the resolution ImageNet features were learned at, still cheap." ;;
    E5) run E5 --exp E5_resnet18_finetune --model resnet18 --pretrained --mode white --size 128 --epochs 10 --lr 5e-4 \
          --method "Transfer learning, full fine-tuning: all ResNet18 layers updated (AdamW lr 5e-4, OneCycle)" \
          --preproc "${WHITE//SIZE/128}" --why "Same preprocessing as E4 so that only frozen-vs-fine-tuned changes." ;;
    E6) run E6 --exp E6_effnetb0_finetune --model efficientnet_b0 --pretrained --mode white --size 128 --epochs 10 --lr 1e-3 \
          --method "Full fine-tuning of ImageNet EfficientNet-B0 (AdamW lr 1e-3, OneCycle)" \
          --preproc "${WHITE//SIZE/128}" --why "Same preprocessing as E5; only the architecture changes." ;;
    E6b) run E6b --exp E6b_convnext_tiny_finetune --model convnext_tiny --pretrained --mode white --size 128 --epochs 10 --lr 2e-4 --bs 32 \
          --method "Full fine-tuning of ImageNet ConvNeXt-Tiny (AdamW lr 2e-4, OneCycle, batch 32)" \
          --preproc "${WHITE//SIZE/128}" --why "Same preprocessing as E5; only the architecture changes." ;;
    E7) run E7 --exp E7_convnext_tiny_aug --model convnext_tiny --pretrained --mode white --size 128 --epochs 15 --lr 2e-4 --bs 32 \
          --aug geo --label-smoothing 0.1 \
          --method "ConvNeXt-Tiny fine-tune (as E6b) + geometric augmentation (h-flip, +-10 deg rotation, 5% shift, 0.9-1.1 scale) + label smoothing 0.1, 15 epochs" \
          --preproc "${WHITE//SIZE/128}; train-time random flip/affine (no colour jitter)" \
          --why "E6b overfits (train 1.000 vs val 0.959). Geometric augmentation creates plausible variants without touching colour, which is itself a vendor signal; label smoothing discourages over-confident memorisation." ;;
    E8) run E8 --exp E8_convnext_tiny_224 --model convnext_tiny --pretrained --mode white --size 224 --epochs 10 --lr 2e-4 --bs 32 \
          --method "ConvNeXt-Tiny fine-tune exactly as E6b; only input resolution changes 128 -> 224" \
          --preproc "${WHITE//SIZE/224}" \
          --why "224 px is ConvNeXt's pre-training resolution and close to the native 240 px of Facebook/Messenger/Mozilla files, so fine stroke and shading details are preserved instead of being downsampled." ;;
    E10a) run E10a --exp E10a_convnext_tiny_128_full --full --model convnext_tiny --pretrained --mode white --size 128 --epochs 10 --lr 2e-4 --bs 32 \
          --method "Final-model recipe of E6b retrained on 100% of the labelled data (no held-out split); scored on Kaggle only" \
          --preproc "${WHITE//SIZE/128}" \
          --why "Hyper-parameters were chosen on the 80/20 split; retraining on all 9879 images adds 25% more data, which matters most for the small classes (mozilla 397 images)." ;;
    E10b) run E10b --exp E10b_convnext_tiny_224_full --full --model convnext_tiny --pretrained --mode white --size 224 --epochs 10 --lr 2e-4 --bs 32 \
          --method "Final-model recipe of E8 retrained on 100% of the labelled data (no held-out split); scored on Kaggle only" \
          --preproc "${WHITE//SIZE/224}" \
          --why "Same as E10a: 25% more training data with the best validated recipe (E8)." ;;
    E11) $PY -m src.ensemble E11_ens_full_128_224 E10a_convnext_tiny_128_full E10b_convnext_tiny_224_full ;;
    E12a) run E12a --exp E12a_convnext_tiny_144_bicubic --model convnext_tiny --pretrained --mode white --size 144 --epochs 10 --lr 2e-4 --bs 32 \
          --method "ConvNeXt-Tiny fine-tune as E6b at 144 px (= 2 x 72) - control run for E12b" \
          --preproc "${WHITE//SIZE/144}" \
          --why "Control: identical to E12b except the resampling filter. 144 = exact 2x of the 72 px files that cause 32 of the 41 E9 validation errors." ;;
    E12b) run E12b --exp E12b_convnext_tiny_144_nearest --model convnext_tiny --pretrained --mode white --size 144 --resample nearest --epochs 10 --lr 2e-4 --bs 32 \
          --method "Same as E12a; only the resize filter changes: nearest-neighbour instead of bicubic" \
          --preproc "Alpha-composite on white background, then resize to 144x144 with NEAREST neighbour, ImageNet normalisation" \
          --why "Error analysis: 32/41 E9 errors are 72 px images. Bicubic upscaling smooths away the pixel-level rendering (anti-aliasing, outline pixels) that may identify the vendor; nearest at an integer factor (2x) keeps every source pixel intact." ;;
    E13) run E13 --exp E13_convnext_tiny_216_nearest --model convnext_tiny --pretrained --mode white --size 216 --resample nearest --epochs 10 --lr 2e-4 --bs 32 \
          --method "Same as E12b (nearest resize) at 216 px instead of 144 px" \
          --preproc "Alpha-composite on white background, then resize to 216x216 with NEAREST neighbour, ImageNet normalisation" \
          --why "E12b showed nearest > bicubic (+0.9 pts) and E8 showed higher resolution helps (+1.7 pts). 216 = exact 3x of the 72 px files, close to the 224 px of E8 and to the native 240 px of the large files." ;;
    E14a) run E14a --exp E14a_convnext_tiny_144_nearest_full --full --model convnext_tiny --pretrained --mode white --size 144 --resample nearest --epochs 10 --lr 2e-4 --bs 32 \
          --method "E12b recipe retrained on 100% of the labelled data; ensemble member" \
          --preproc "Alpha-composite on white background, then resize to 144x144 with NEAREST neighbour, ImageNet normalisation" \
          --why "Full-data version of E12b (see E10 for why full-data retraining helps)." ;;
    E14b) run E14b --exp E14b_convnext_tiny_216_nearest_full --full --model convnext_tiny --pretrained --mode white --size 216 --resample nearest --epochs 10 --lr 2e-4 --bs 32 \
          --method "E13 recipe retrained on 100% of the labelled data; ensemble member" \
          --preproc "Alpha-composite on white background, then resize to 216x216 with NEAREST neighbour, ImageNet normalisation" \
          --why "Full-data version of E13." ;;
    E15) $PY -m src.ensemble E15_ens_full_3models E10b_convnext_tiny_224_full E14a_convnext_tiny_144_nearest_full E14b_convnext_tiny_216_nearest_full ;;
    E16) run E16 --exp E16_convnext_tiny_144_nearest_pseudo --model convnext_tiny --pretrained --mode white --size 144 --resample nearest --epochs 10 --lr 2e-4 --bs 32 \
          --pseudo E13v_ens_val_3models --pseudo-thr 0.99 \
          --method "E12b recipe + pseudo-labelling (semi-supervised): test images predicted with >=0.99 confidence by the E13v ensemble (trained on the 80% split only) are added to training with their predicted class" \
          --preproc "Alpha-composite on white background, then resize to 144x144 with NEAREST neighbour, ImageNet normalisation" \
          --why "Single change vs E12b: ~9000 extra (unlabelled) test images. On validation, predictions with >=0.99 confidence cover 91.5% of images and are 99.7% correct, so the added labels are almost noise-free. Pseudo-labels come from models that never saw the validation split -> validation score stays unbiased." ;;
    *) echo "unknown experiment $1"; exit 1 ;;
  esac
}

[ $# -eq 0 ] && { grep -oE '^    E[0-9a-z]+\)' "$0" | tr -d ' )'; exit 0; }
for e in "$@"; do case_exp "$e"; done
