"""GPU recipe: ImageNet-pretrained backbone, full fine-tuning, k-fold cross-validation, flip TTA.

Default = ConvNeXt-Tiny at 224 px, 5 folds (StratifiedGroupKFold on preprocessed-pixel hashes, seed 42),
10 epochs per fold, AdamW + OneCycle, mixed precision, horizontal-flip TTA at inference.

Outputs per fold: fold{k}.pt, fold{k}_history.csv, fold{k}_run.json, oof probabilities and test probabilities.
Aggregate: oof_proba.npy (every training image predicted by the fold that did not see it), oof_summary.json,
test_proba_mean.npy, submission.csv, confusion matrices and curves.

Examples
  python scripts/train_backbone.py --zip /Volumes/.../2-computer-vision-2026-b-sc-aidams-final-proj.zip --out outputs/convnext224
  python scripts/train_backbone.py --folds 1 --out outputs/convnext224_fixedsplit      # one model on the stored split
  python scripts/train_backbone.py --smoke --out outputs/smoke_gpu                      # code-path check (minutes on CPU)
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
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emoji_provider import data as D  # noqa: E402
from emoji_provider import evaluate as E  # noqa: E402
from emoji_provider.models import TimmBackbone, count_parameters  # noqa: E402
from emoji_provider.submit import write_submission  # noqa: E402
from emoji_provider.train import TrainConfig, fit, predict_proba  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--zip", default=None)
p.add_argument("--data-dir", default=str(ROOT / "data" / "extracted"))
p.add_argument("--out", default=str(ROOT / "outputs" / "convnext224"))
p.add_argument("--split", default=str(ROOT / "data" / "split_seed42.csv"), help="used when --folds 1")
p.add_argument("--model", default="convnext_tiny.fb_in1k", help="timm name; fb_in1k = the ImageNet-1k weights of the ConvNeXt paper")
p.add_argument("--size", type=int, default=224)
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--weight-decay", type=float, default=0.05)
p.add_argument("--drop-path", type=float, default=0.1)
p.add_argument("--label-smoothing", type=float, default=0.0)
p.add_argument("--select-on", default="val_loss", choices=["val_loss", "val_accuracy", "last"],
               help="which epoch's weights to keep per fold: lowest val loss (default), highest val accuracy, or the last epoch")
p.add_argument("--folds", type=int, default=5, help="1 = single model on the stored split; k = k-fold ensemble")
p.add_argument("--only-fold", type=int, default=None, help="train just this fold index (to parallelise)")
p.add_argument("--no-augment", action="store_true", help="disable flip + integer shift (replicates the E8 setting)")
p.add_argument("--no-tta", action="store_true")
p.add_argument("--no-pretrained", action="store_true")
p.add_argument("--workers", type=int, default=4)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--smoke", action="store_true")
args = p.parse_args()

OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
FIG = OUT / "figures"; FIG.mkdir(exist_ok=True)
D.seed_everything(args.seed)
device = D.pick_device()
print(f"device: {device} | torch {torch.__version__} | model {args.model} @ {args.size}px | folds {args.folds}")
if args.smoke:
    args.size, args.epochs, args.batch, args.workers = min(args.size, 64), 1, 16, 0
    pretrained = False
    print("SMOKE TEST: tiny subset, 1 epoch, random init; checks the code path only")
else:
    pretrained = not args.no_pretrained

# ------------------------------------------------------------------ data
data_dir = Path(args.data_dir)
if not (data_dir / "train_labels.csv").is_file():
    D.extract_zip(D.find_zip(args.zip), data_dir)
cat = D.load_catalogue(data_dir)
C, y = cat.class_names, cat.y
train_paths = [cat.train_files[i] for i in cat.train_ids]
test_paths = [cat.test_files[i] for i in cat.test_ids]
t0 = time.perf_counter()
Xtr = D.load_pixels(train_paths, args.size, cache=ROOT / "data" / f"cache_train_{args.size}.npy")
Xte = D.load_pixels(test_paths, args.size, cache=ROOT / "data" / f"cache_test_{args.size}.npy")
print(f"pixels ready in {time.perf_counter() - t0:.0f}s: {Xtr.shape}, {Xte.shape}")
# groups from the 96px recipe so duplicates are defined the same way in every experiment
X96 = D.load_pixels(train_paths, 96, cache=ROOT / "data" / "cache_train_96.npy")
hashes = D.pixel_hashes(X96)
del X96

if args.folds == 1:
    tr_idx, va_idx = D.load_fixed_split(Path(args.split), cat.labels)
    folds = [(tr_idx, va_idx)]
else:
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    folds = [(a, b) for a, b in splitter.split(np.zeros(len(y)), y, groups=hashes)]
    g = np.asarray(hashes)
    for a, b in folds:
        assert set(g[a]).isdisjoint(g[b])
    fold_table = pd.DataFrame({"Id": cat.train_ids, "Label": cat.labels["Label"], "fold": -1})
    for k, (_, b) in enumerate(folds):
        fold_table.loc[b, "fold"] = k
    fold_table.to_csv(OUT / "folds.csv", index=False)
test_ids = list(cat.test_ids)
if args.smoke:
    folds = [(np.concatenate([a[y[a] == c][:8] for c in range(len(C))]),
              np.concatenate([b[y[b] == c][:4] for c in range(len(C))])) for a, b in folds[:2]]
    Xte, test_ids = Xte[:64], test_ids[:64]   # a handful of test images is enough to exercise the path

cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch, lr=args.lr, weight_decay=args.weight_decay,
                  label_smoothing=args.label_smoothing, onecycle=True, patience=0, select_on=args.select_on,
                  amp=True, num_workers=args.workers, seed=args.seed,
                  notes=f"{args.model} {args.size}px pretrained={pretrained} augment={not args.no_augment} "
                        f"drop_path={args.drop_path} tta={not args.no_tta} select_on={args.select_on}")
(OUT / "config.json").write_text(json.dumps(vars(args) | {"pretrained": pretrained}, indent=2))


def make_loader(X, idx=None, targets=None, augment=False, shuffle=False):
    px = X if idx is None else X[idx]
    tg = None if targets is None else targets[idx]
    ds = D.EmojiDataset(px, tg, augment=augment, imagenet_norm=True)
    return DataLoader(ds, batch_size=args.batch, shuffle=shuffle, num_workers=args.workers,
                      pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(args.seed),
                      persistent_workers=args.workers > 0)


test_loader = make_loader(Xte)
oof = np.full((len(y), len(C)), np.nan, dtype=np.float32)
test_probas = []
fold_rows = []
for k, (tr_idx, va_idx) in enumerate(folds):
    if args.only_fold is not None and k != args.only_fold:
        continue
    print(f"\n===== fold {k}: train {len(tr_idx):,} | val {len(va_idx):,} =====")
    D.seed_everything(args.seed + k)
    model = TimmBackbone(args.model, len(C), pretrained=pretrained, drop_path_rate=args.drop_path)
    print(f"parameters: {count_parameters(model):,}")
    tag = f"fold{k}"
    hist = fit(model, make_loader(Xtr, tr_idx, y, augment=not args.no_augment, shuffle=True),
               make_loader(Xtr, va_idx, y), cfg, device, OUT, C, tag=tag)
    run = json.loads((OUT / f"{tag}_run.json").read_text())
    pv = predict_proba(model, make_loader(Xtr, va_idx, y), device, tta_hflip=not args.no_tta)
    pt = predict_proba(model, test_loader, device, tta_hflip=not args.no_tta)
    oof[va_idx] = pv
    np.save(OUT / f"{tag}_val_proba.npy", pv); np.save(OUT / f"{tag}_val_idx.npy", va_idx)
    np.save(OUT / f"{tag}_test_proba.npy", pt)
    test_probas.append(pt)
    s = E.summarize(y[va_idx], pv, C)
    fold_rows.append({"fold": k, "val_accuracy": s["accuracy"], "macro_f1": s["macro_f1"], "log_loss": s["log_loss"],
                      "best_epoch": run["best_epoch"], "minutes": run["training_seconds"] / 60})
    print(f"fold {k}: val acc {s['accuracy']:.4%} ({s['accuracy_correct']}/{s['n']}), macro F1 {s['macro_f1']:.4f}")
    E.plot_learning_curves(hist, FIG / f"curves_{tag}.png", f"{args.model} {args.size}px, fold {k}", run["best_epoch"])
    E.plot_confusion(y[va_idx], pv.argmax(1), C, FIG / f"cm_{tag}.png", f"{args.model} fold {k}: {s['accuracy']:.2%}")
    pd.DataFrame(fold_rows).to_csv(OUT / "fold_summary.csv", index=False)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

# ------------------------------------------------------------------ aggregate
if args.only_fold is None:
    covered = ~np.isnan(oof[:, 0])
    s = E.summarize(y[covered], oof[covered], C)
    s["covered_images"] = int(covered.sum())
    if args.folds > 1 and Path(args.split).is_file() and not args.smoke:
        # the stored validation set, scored out-of-fold -> directly comparable with every CPU experiment
        _, fixed_val = D.load_fixed_split(Path(args.split), cat.labels)
        fv = fixed_val[covered[fixed_val]]
        s_fixed = E.summarize(y[fv], oof[fv], C)
        s["stored_split_validation_oof"] = {"accuracy": s_fixed["accuracy"], "macro_f1": s_fixed["macro_f1"],
                                            "n": s_fixed["n"], "correct": s_fixed["accuracy_correct"]}
    s["folds"] = fold_rows
    s["device"] = str(device)
    s["device_name"] = torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)
    s["model"] = args.model
    s["image_size"] = args.size
    E.save_summary(s, OUT / "oof_summary.json")
    np.save(OUT / "oof_proba.npy", oof)
    print(f"\nOUT-OF-FOLD over {covered.sum():,} images: acc {s['accuracy']:.4%}, macro F1 {s['macro_f1']:.4f}")
    if "stored_split_validation_oof" in s:
        print("stored validation split (OOF):", s["stored_split_validation_oof"])
    E.plot_confusion(y[covered], oof[covered].argmax(1), C, FIG / "cm_oof.png",
                     f"{args.model} {args.size}px, out-of-fold ({covered.sum():,} images): {s['accuracy']:.2%}")
    E.plot_confusion(y[covered], oof[covered].argmax(1), C, FIG / "cm_oof_norm.png",
                     f"{args.model} {args.size}px, out-of-fold (row-normalised)", normalize=True)
    ids = np.asarray(cat.train_ids)
    fails = E.plot_failures(Xtr[covered], y[covered], oof[covered], C, ids[covered], FIG / "failures_oof.png",
                            f"{args.model}: most confident out-of-fold mistakes (true→predicted, confidence)")
    fails.to_csv(OUT / "failures_oof.csv", index=False)
    mean_test = np.mean(test_probas, axis=0)
    np.save(OUT / "test_proba_mean.npy", mean_test)
    sub_name = "smoke_submission.csv" if args.smoke else "submission.csv"
    write_submission(test_ids, mean_test, C, OUT / sub_name, None if args.smoke else set(cat.test_files))
    print("wrote", OUT / sub_name)
    print(pd.Series([C[i] for i in mean_test.argmax(1)]).value_counts().to_string())
