# Runbook: what was run, where things are, what is left

Deadlines (from the evidence guide): **Kaggle closes Sunday 18 October, 22:00 Paris. Report + code on Moodle: 28 October, 20:00.**

## A. Done on 17 October (GPU runs on RunPod)

* Pod: RunPod on-demand, 1x NVIDIA GeForce RTX 4090 (24 GB), PyTorch 2.8.0 template (torch 2.8.0+cu128, driver 570.169),
  $0.89 per hour. Setup: upload the Kaggle archive (SHA-256 checked, `7f8c12a2...44ad8d`) and the codebase to `/workspace`,
  `pip install -r requirements-gpu-runtime.txt` (the image had no scikit-learn and no timm).
* Run A: `scripts/train_backbone.py ... --out outputs/convnext224` (checkpoint = lowest validation loss per fold).
  Out-of-fold accuracy 97.29%. Run B: same with `--select-on last`, 97.69%. Each run about 16 minutes of pod time. The
  two runs reproduced each other exactly, epoch by epoch (same seeds, same GPU): they differ only in the epoch kept.
* Final model = run B (`outputs/convnext224_last/submission.csv`, copied as `outputs/submissions/submission_convnext224_last.csv`).
  Ablation: `scripts/make_final_submission.py` on the two `test_proba_mean.npy` files, equal weights, no post-processing
  rule, written to `outputs/submissions/submission_convnext224_AB_10models.csv` (10-checkpoint out-of-fold accuracy 97.70%,
  differs from run B's file on 20 test images).
* The pod was stopped afterwards. Total credit used: about $1.10. A **stopped** pod still bills its volume disk
  ($0.20 per GB per month, RunPod pricing page); **terminate** it once you are sure nothing else is needed from it.
  Everything needed is already off the pod:
  * `outputs/convnext224/` and `outputs/convnext224_last/` in this folder (metrics, histories, probabilities, figures,
    submissions, fold tables); `outputs/runpod_training_logs.txt` (every epoch of both runs);
  * `emoji_convnext224_checkpoints.zip` (1.1 GB, `fold0-4.pt` + `config.json` of both runs) in Adam's `~/Downloads`.
    Not part of the Moodle delivery; keep it in case the teacher asks for the weights.

## B. Kaggle (closes Sunday 18 October, 22:00 Paris)

1. Done. Public scores on 18 October: **0.98319** for `submission_convnext224_last.csv` (final model, submitted 21:51)
   and 0.98279 for `submission_convnext224_AB_10models.csv` (ten-checkpoint average, submitted about 11:30); the other
   18 October entries scored 0.96881 (DINOv2 + ConvNeXt ensemble) and 0.95586 (fine-tuned DINOv2), neither produced by
   this codebase. The two **selected** entries at the close were the final model and Erdem's E15 (Kaggle keeps the best
   private score of the two).
2. Done. Private scores (read 27 October): final model 0.98218, ten-checkpoint average 0.98198, E15 0.98238 (team
   result, 5th of 18). All entries with both scores are in Table 9 of the paper.

## C. Final hand-in (Moodle, 28 October, 20:00)

1. The report is the LaTeX paper `paper/Paper_The_Winners.pdf` (source in `paper/`, built with `make`). It replaces the
   18 October draft. Every verified number comes from `outputs/`; reported numbers come from the team's log and Kaggle.
2. `AI_USAGE.md` lists every AI tool used and the prompts; each teammate confirms their line.
3. Upload the paper PDF and the codebase zip on Moodle.

## D. If something fails when re-running

* `ModuleNotFoundError: sklearn` / `timm` on a GPU image → `pip install -r requirements-gpu-runtime.txt`.
* `FileNotFoundError: codebase not found` (notebook) → run it from the `notebooks/` folder of the unzipped codebase,
  or set `CODEBASE_ZIP`.
* `timm` cannot download weights → the machine has no internet; copy a `convnext_tiny.fb_in1k` checkpoint into
  `$HF_HOME` or point `HF_HOME` at a folder that has it.
* Out of GPU memory → `--batch 16` (OneCycle adapts to the number of steps automatically).
* Anything else: `python scripts/train_backbone.py --smoke` exercises the whole GPU path in minutes on any machine and
  prints the failing step; `python -m pytest tests -q` checks the library.
