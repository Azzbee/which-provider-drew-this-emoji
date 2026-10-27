"""Regenerate the explainability figures (Grad-CAM, most confident mistakes) from a saved EmojiCNNv2 checkpoint.

  python scripts/explain_checkpoint.py --ckpt outputs/cpu/emojicnnv2.pt --out outputs/cpu/figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from emoji_provider import data as D  # noqa: E402
from emoji_provider import evaluate as E  # noqa: E402
from emoji_provider.models import EmojiCNNv2  # noqa: E402
from emoji_provider.train import predict_proba  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default=str(ROOT / "outputs" / "cpu" / "emojicnnv2.pt"))
p.add_argument("--data-dir", default=str(ROOT / "data" / "extracted"))
p.add_argument("--split", default=str(ROOT / "data" / "split_seed42.csv"))
p.add_argument("--out", default=str(ROOT / "outputs" / "cpu" / "figures"))
p.add_argument("--per-class", type=int, default=2)
args = p.parse_args()
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)

cat = D.load_catalogue(Path(args.data_dir))
C, y = cat.class_names, cat.y
X = D.load_pixels([cat.train_files[i] for i in cat.train_ids], 96, cache=ROOT / "data" / "cache_train_96.npy")
_, val_idx = D.load_fixed_split(Path(args.split), cat.labels)
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
m = EmojiCNNv2(len(C)); m.load_state_dict(ck["model_state_dict"]); m.eval()
ds = D.EmojiDataset(X[val_idx], y[val_idx])
pv = predict_proba(m, DataLoader(ds, batch_size=256), torch.device("cpu"))
yv = y[val_idx]
print(f"checkpoint epoch {ck['epoch']}: val acc {(pv.argmax(1) == yv).mean():.4%}")
rng = np.random.default_rng(0)
picks = []
for c in range(len(C)):
    ok = np.flatnonzero((yv == c) & (pv.argmax(1) == c))
    picks.extend(rng.choice(ok, size=min(args.per_class, len(ok)), replace=False).tolist())
E.plot_gradcam_grid(m, m.features[-2], ds, picks, C, OUT / "gradcam_emojicnnv2.png",
                    "Grad-CAM (EmojiCNNv2): where the network looks", torch.device("cpu"), X[val_idx], yv)
E.plot_failures(X[val_idx], yv, pv, C, np.asarray(cat.train_ids)[val_idx], OUT / "failures_emojicnnv2.png",
                "EmojiCNNv2: most confident validation mistakes (true→predicted, confidence)")
print("figures written to", OUT)
