"""CPU-friendly experiment ladder on the fixed validation split.

  1. metadata-only diagnostic (no pixels)
  2. the saved September baseline checkpoint, re-scored
  3. EmojiCNNv2 from scratch, pixels only
  4. EmojiCNNv2 + metadata late fusion (ablation)

Everything is scored on the same 1,979 validation images (data/split_seed42.csv) so the numbers are comparable.
Run:  python scripts/run_cpu_experiments.py --zip <archive> --out outputs/cpu [--epochs 25]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emoji_provider import data as D  # noqa: E402
from emoji_provider import evaluate as E  # noqa: E402
from emoji_provider.metadata_model import run_metadata_diagnostic, size_family  # noqa: E402
from emoji_provider.models import EmojiCNN, EmojiCNNv2, MetaFusion, count_parameters  # noqa: E402
from emoji_provider.submit import write_submission  # noqa: E402
from emoji_provider.train import TrainConfig, fit, predict_proba  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--zip", default=None)
p.add_argument("--data-dir", default=str(ROOT / "data" / "extracted"))
p.add_argument("--out", default=str(ROOT / "outputs" / "cpu"))
p.add_argument("--split", default=str(ROOT / "data" / "split_seed42.csv"))
p.add_argument("--baseline-ckpt", default=str(ROOT / "data" / "baseline_sept30" / "best_model.pt"))
p.add_argument("--epochs", type=int, default=25)
p.add_argument("--size", type=int, default=96)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--smoke", action="store_true", help="tiny subset, 2 epochs: checks the code path only")
p.add_argument("--skip", nargs="*", default=[], help="any of: meta baseline v2 fusion")
args = p.parse_args()

OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
FIG = OUT / "figures"; FIG.mkdir(exist_ok=True)
D.seed_everything(args.seed)
device = D.pick_device()
print("device:", device, "| torch", torch.__version__)

# ------------------------------------------------------------------ data
data_dir = Path(args.data_dir)
if not (data_dir / "train_labels.csv").is_file():
    D.extract_zip(D.find_zip(args.zip), data_dir)
cat = D.load_catalogue(data_dir)
C = cat.class_names
y = cat.y
print(f"{len(cat.train_ids):,} train / {len(cat.test_ids):,} test images; classes {C}")

t0 = time.perf_counter()
train_paths = [cat.train_files[i] for i in cat.train_ids]
test_paths = [cat.test_files[i] for i in cat.test_ids]
meta_cache = OUT / "metadata_train.csv"
if meta_cache.is_file():
    meta_tr = pd.read_csv(meta_cache, dtype={"Id": str})
    meta_te = pd.read_csv(OUT / "metadata_test.csv", dtype={"Id": str})
else:
    print("scanning native metadata ...")
    meta_tr = D.scan_all_metadata(train_paths); meta_tr.to_csv(meta_cache, index=False)
    meta_te = D.scan_all_metadata(test_paths); meta_te.to_csv(OUT / "metadata_test.csv", index=False)
assert meta_tr["Id"].tolist() == cat.train_ids and meta_te["Id"].tolist() == cat.test_ids

print(f"loading pixels at {args.size}px ...")
Xtr = D.load_pixels(train_paths, args.size, cache=ROOT / "data" / f"cache_train_{args.size}.npy")
Xte = D.load_pixels(test_paths, args.size, cache=ROOT / "data" / f"cache_test_{args.size}.npy")
print(f"pixels ready in {time.perf_counter() - t0:.0f}s: {Xtr.shape} {Xte.shape}")

hashes = D.pixel_hashes(Xtr)
dups = D.duplicate_report(hashes, y)
print("duplicates:", dups)

# fixed split (regenerated and cross-checked when the stored file is absent)
regen_tr, regen_va = D.make_group_split(y, hashes, seed=args.seed)
if Path(args.split).is_file():
    train_idx, val_idx = D.load_fixed_split(Path(args.split), cat.labels)
    same = set(train_idx) == set(regen_tr) and set(val_idx) == set(regen_va)
    print(f"stored split loaded; identical to regenerated StratifiedGroupKFold fold 0: {same}")
else:
    train_idx, val_idx = regen_tr, regen_va
    s = cat.labels.copy(); s["split"] = "train"; s.loc[val_idx, "split"] = "validation"
    Path(args.split).parent.mkdir(parents=True, exist_ok=True); s.to_csv(args.split, index=False)
if args.smoke:
    train_idx = np.concatenate([train_idx[y[train_idx] == c][:12] for c in range(len(C))])
    val_idx = np.concatenate([val_idx[y[val_idx] == c][:6] for c in range(len(C))])
print(f"train {len(train_idx):,} | validation {len(val_idx):,}")
yv = y[val_idx]
majority = int(np.bincount(y[train_idx]).argmax())
results: dict[str, dict] = {}
if (OUT / "results.json").is_file() and not args.smoke:   # partial reruns (--skip ...) keep earlier entries
    results = json.loads((OUT / "results.json").read_text())
results["majority_class"] = {"accuracy": float((yv == majority).mean()), "label": C[majority]}
(OUT / "duplicates.json").write_text(json.dumps(dups, indent=2))

# ------------------------------------------------------------------ 1. metadata-only diagnostic
if "meta" not in args.skip:
    results = {k: v for k, v in results.items() if not k.startswith("metadata-only")}
    print("\n== metadata-only diagnostic ==")
    diag = run_metadata_diagnostic(meta_tr, y, train_idx, val_idx, seed=args.seed)
    for k, v in diag.items():
        print(f"  {k:40s} val acc {v['val_accuracy']:.4f}  ({v['n_features']} features)")
        results[f"metadata-only / {k}"] = {"accuracy": v["val_accuracy"], "n_features": v["n_features"]}
    best_key = max(diag, key=lambda k: diag[k]["val_accuracy"])
    s = E.summarize(yv, diag[best_key]["val_proba"], C)
    results[f"metadata-only / {best_key}"].update({"macro_f1": s["macro_f1"], "per_class": s["per_class"]})
    E.plot_confusion(yv, diag[best_key]["val_proba"].argmax(1), C, FIG / "cm_metadata_only.png",
                     f"Metadata only ({best_key}): {s['accuracy']:.1%}")
    # family table: which providers are separable by native size alone
    fam = meta_tr.assign(family=[size_family(w, h) for w, h in zip(meta_tr["width"], meta_tr["height"])],
                         Label=cat.labels["Label"])
    pd.crosstab(fam["Label"], fam["family"]).to_csv(OUT / "size_family_by_provider.csv")

# ------------------------------------------------------------------ loaders
def loader(idx, augment, meta=None, shuffle=False):
    ds = D.EmojiDataset(Xtr[idx], y[idx], None if meta is None else meta[idx], augment=augment)
    return DataLoader(ds, batch_size=args.batch, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(args.seed))

val_plain = loader(val_idx, augment=False)
test_plain = DataLoader(D.EmojiDataset(Xte), batch_size=args.batch, shuffle=False)
Xv_disp = Xtr[val_idx]
val_ids = np.asarray(cat.train_ids)[val_idx]

# ------------------------------------------------------------------ 2. saved September baseline
if "baseline" not in args.skip and Path(args.baseline_ckpt).is_file():
    print("\n== saved baseline checkpoint (30 Sept) ==")
    ck = torch.load(args.baseline_ckpt, map_location="cpu", weights_only=True)
    assert ck["class_names"] == C and ck["image_size"] == 96
    base = EmojiCNN(len(C)); base.load_state_dict(ck["model_state_dict"]); base.eval()
    # the baseline was trained on x/255 without mean/std normalisation -> feed it that way
    ds_raw = D.EmojiDataset(Xtr[val_idx], y[val_idx]); ds_raw.mean = torch.zeros(3, 1, 1); ds_raw.std = torch.ones(3, 1, 1)
    pb = predict_proba(base, DataLoader(ds_raw, batch_size=256), torch.device("cpu"))
    s = E.summarize(yv, pb, C); results["baseline_sept30 / EmojiCNN 96px"] = s
    print(f"  val acc {s['accuracy']:.4%} ({s['accuracy_correct']}/{s['n']}), macro F1 {s['macro_f1']:.4f}")
    E.plot_confusion(yv, pb.argmax(1), C, FIG / "cm_baseline_sept30.png", f"Baseline CNN (30 Sept): {s['accuracy']:.1%}")
    E.plot_confusion(yv, pb.argmax(1), C, FIG / "cm_baseline_sept30_norm.png", f"Baseline CNN (30 Sept), row-normalised", normalize=True)
    np.save(OUT / "val_proba_baseline.npy", pb)

cfg = TrainConfig(epochs=2 if args.smoke else args.epochs, batch_size=args.batch, lr=2e-3, weight_decay=1e-4,
                  onecycle=True, patience=0, select_on="val_loss", seed=args.seed)
summaries_for_plot: dict[str, dict] = {}
probas: dict[str, np.ndarray] = {}

# ------------------------------------------------------------------ 3. EmojiCNNv2 pixels only
if "v2" not in args.skip:
    print("\n== EmojiCNNv2 (pixels only) ==")
    D.seed_everything(args.seed)
    m = EmojiCNNv2(len(C)); print("  parameters:", f"{count_parameters(m):,}")
    hist = fit(m, loader(train_idx, augment=True, shuffle=True), val_plain, cfg, device, OUT, C, tag="emojicnnv2")
    run = json.loads((OUT / "emojicnnv2_run.json").read_text())
    pv = predict_proba(m, val_plain, device); pt = predict_proba(m, test_plain, device)
    s = E.summarize(yv, pv, C); s.update({"best_epoch": run["best_epoch"], "training_seconds": run["training_seconds"],
                                          "parameters": count_parameters(m)})
    results["EmojiCNNv2 96px / pixels only"] = s; summaries_for_plot["EmojiCNNv2 (pixels)"] = s; probas["v2"] = pv
    print(f"  val acc {s['accuracy']:.4%} ({s['accuracy_correct']}/{s['n']}), macro F1 {s['macro_f1']:.4f}")
    E.plot_learning_curves(hist, FIG / "curves_emojicnnv2.png", "EmojiCNNv2, pixels only (96 px)", run["best_epoch"])
    E.plot_confusion(yv, pv.argmax(1), C, FIG / "cm_emojicnnv2.png", f"EmojiCNNv2, pixels only: {s['accuracy']:.1%}")
    E.plot_confusion(yv, pv.argmax(1), C, FIG / "cm_emojicnnv2_norm.png", "EmojiCNNv2, pixels only (row-normalised)", normalize=True)
    fails = E.plot_failures(Xv_disp, yv, pv, C, val_ids, FIG / "failures_emojicnnv2.png",
                            "EmojiCNNv2: most confident validation mistakes (true→predicted, confidence)")
    fails.to_csv(OUT / "failures_emojicnnv2.csv", index=False)
    np.save(OUT / "val_proba_emojicnnv2.npy", pv); np.save(OUT / "test_proba_emojicnnv2.npy", pt)
    write_submission(cat.test_ids, pt, C, OUT / "submission_emojicnnv2.csv", set(cat.test_files))
    # Grad-CAM on the last conv stage, 2 correct examples per provider
    rng = np.random.default_rng(0)
    picks = []
    for c in range(len(C)):
        ok = np.flatnonzero((yv == c) & (pv.argmax(1) == c))
        picks.extend(rng.choice(ok, size=min(2, len(ok)), replace=False).tolist())
    last_conv = m.features[-2]  # conv_bn block of the last stage (before AdaptiveAvgPool)
    E.plot_gradcam_grid(m, last_conv, D.EmojiDataset(Xtr[val_idx], y[val_idx]), picks, C, FIG / "gradcam_emojicnnv2.png",
                        "Grad-CAM (EmojiCNNv2): where the network looks", device, Xv_disp, yv)

elif (OUT / "val_proba_emojicnnv2.npy").is_file() and (OUT / "emojicnnv2_run.json").is_file() and not args.smoke:
    # skipped retraining: re-score the saved probabilities so results.json stays complete
    pv = np.load(OUT / "val_proba_emojicnnv2.npy"); run = json.loads((OUT / "emojicnnv2_run.json").read_text())
    if len(pv) == len(yv):
        s = E.summarize(yv, pv, C); s.update({"best_epoch": run["best_epoch"], "training_seconds": run["training_seconds"],
                                              "parameters": count_parameters(EmojiCNNv2(len(C)))})
        results["EmojiCNNv2 96px / pixels only"] = s; probas["v2"] = pv
        print(f"\n== EmojiCNNv2 (pixels only), from saved probabilities: val acc {s['accuracy']:.4%} ==")

# ------------------------------------------------------------------ 4. EmojiCNNv2 + metadata fusion
if "fusion" not in args.skip:
    print("\n== EmojiCNNv2 + metadata late fusion (ablation) ==")
    D.seed_everything(args.seed)
    Mtr = D.metadata_features(meta_tr, level="format"); Mte = D.metadata_features(meta_te, level="format")
    mu, sd = Mtr[train_idx].mean(0), Mtr[train_idx].std(0) + 1e-6
    Mtr = (Mtr - mu) / sd; Mte = (Mte - mu) / sd
    fm = MetaFusion(EmojiCNNv2(len(C)), meta_dim=Mtr.shape[1], n_classes=len(C))
    val_meta = loader(val_idx, augment=False, meta=Mtr)
    hist = fit(fm, loader(train_idx, augment=True, meta=Mtr, shuffle=True), val_meta, cfg, device, OUT, C, tag="emojicnnv2_fusion")
    run = json.loads((OUT / "emojicnnv2_fusion_run.json").read_text())
    pv = predict_proba(fm, val_meta, device)
    pt = predict_proba(fm, DataLoader(D.EmojiDataset(Xte, None, Mte), batch_size=args.batch), device)
    s = E.summarize(yv, pv, C); s.update({"best_epoch": run["best_epoch"], "training_seconds": run["training_seconds"],
                                          "parameters": count_parameters(fm)})
    results["EmojiCNNv2 96px / pixels + file metadata"] = s; summaries_for_plot["EmojiCNNv2 + metadata"] = s; probas["fusion"] = pv
    print(f"  val acc {s['accuracy']:.4%} ({s['accuracy_correct']}/{s['n']}), macro F1 {s['macro_f1']:.4f}")
    E.plot_learning_curves(hist, FIG / "curves_emojicnnv2_fusion.png", "EmojiCNNv2 + file metadata (96 px)", run["best_epoch"])
    E.plot_confusion(yv, pv.argmax(1), C, FIG / "cm_emojicnnv2_fusion.png", f"EmojiCNNv2 + metadata: {s['accuracy']:.1%}")
    np.save(OUT / "val_proba_emojicnnv2_fusion.npy", pv); np.save(OUT / "test_proba_emojicnnv2_fusion.npy", pt)
    write_submission(cat.test_ids, pt, C, OUT / "submission_emojicnnv2_fusion.csv", set(cat.test_files))

# figures and paired comparison use the merged results, so partial reruns (--skip) still produce complete plots
summaries_for_plot = {label: results[k] for label, k in [("Baseline CNN (30 Sept)", "baseline_sept30 / EmojiCNN 96px"),
                                                         ("EmojiCNNv2 (pixels)", "EmojiCNNv2 96px / pixels only"),
                                                         ("EmojiCNNv2 + metadata", "EmojiCNNv2 96px / pixels + file metadata")]
                      if k in results and "per_class" in results[k]}
if summaries_for_plot:
    E.plot_per_class_f1(summaries_for_plot, C, FIG / "per_class_f1_cpu_models.png", "Per-class validation F1")
for key, fname in (("v2", "val_proba_emojicnnv2.npy"), ("fusion", "val_proba_emojicnnv2_fusion.npy")):
    if key not in probas and (OUT / fname).is_file() and not args.smoke:
        probas[key] = np.load(OUT / fname)
if "v2" in probas and "fusion" in probas and len(probas["v2"]) == len(yv):
    a, b = probas["v2"].argmax(1) == yv, probas["fusion"].argmax(1) == yv
    results["paired / v2_vs_fusion"] = {"both_right": int((a & b).sum()), "only_v2_right": int((a & ~b).sum()),
                                        "only_fusion_right": int((~a & b).sum()), "both_wrong": int((~a & ~b).sum())}

(OUT / "results.json").write_text(json.dumps(results, indent=2, default=float))
print("\nsaved", OUT / "results.json")
for k, v in results.items():
    if isinstance(v, dict) and "accuracy" in v:
        print(f"  {k:55s} {v['accuracy']:.4f}")
