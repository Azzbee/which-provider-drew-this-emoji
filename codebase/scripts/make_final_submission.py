"""Combine saved test probabilities into the final submission (soft voting), with optional post-processing.

Every input is a .npy of shape (n_test, 7) written by train_backbone.py / run_cpu_experiments.py, rows in
sorted test-ID order and columns in sorted class order (apple, facebook, google, messenger, mozilla, samsung,
whatsapp). Equal weights unless --weights is given.

  python scripts/make_final_submission.py outputs/convnext224/test_proba_mean.npy --out submissions/final.csv
  python scripts/make_final_submission.py a.npy b.npy --weights 0.7 0.3 --duplicate-lookup --consistency

The two rules are OFF by default and are reported in the console and in <out>.json when used:
  --duplicate-lookup : a test file byte-identical to a training file takes that training label
  --consistency      : identical test images share one averaged prediction
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emoji_provider import data as D  # noqa: E402
from emoji_provider.submit import consistency_within_test_duplicates, exact_duplicate_lookup, write_submission  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("probas", nargs="+")
p.add_argument("--weights", nargs="*", type=float, default=None)
p.add_argument("--data-dir", default=str(ROOT / "data" / "extracted"))
p.add_argument("--out", default=str(ROOT / "submissions" / "submission.csv"))
p.add_argument("--duplicate-lookup", action="store_true")
p.add_argument("--consistency", action="store_true")
p.add_argument("--compare", default=None, help="an earlier submission.csv to diff against")
args = p.parse_args()

cat = D.load_catalogue(Path(args.data_dir))
C = cat.class_names
mats = [np.load(f) for f in args.probas]
for f, m in zip(args.probas, mats):
    assert m.shape == (len(cat.test_ids), len(C)), f"{f}: shape {m.shape}"
w = np.ones(len(mats)) if not args.weights else np.asarray(args.weights, float)
assert len(w) == len(mats)
w = w / w.sum()
proba = sum(wi * m for wi, m in zip(w, mats))
report = {"inputs": args.probas, "weights": w.tolist(), "rules": {}}
base_pred = proba.argmax(1)

if args.duplicate_lookup or args.consistency:
    train_paths = [cat.train_files[i] for i in cat.train_ids]
    test_paths = [cat.test_files[i] for i in cat.test_ids]
    import hashlib
    tr_h = [hashlib.sha256(p_.read_bytes()).hexdigest() for p_ in train_paths]
    te_h = [hashlib.sha256(p_.read_bytes()).hexdigest() for p_ in test_paths]
    if args.consistency:
        before = proba.argmax(1)
        proba = consistency_within_test_duplicates(te_h, proba)
        report["rules"]["consistency_changed"] = int((proba.argmax(1) != before).sum())
    if args.duplicate_lookup:
        proba, changed = exact_duplicate_lookup(te_h, tr_h, cat.y, proba, len(C))
        report["rules"]["duplicate_lookup_matches"] = int(sum(h in set(tr_h) for h in te_h))
        report["rules"]["duplicate_lookup_changed"] = changed

sub = write_submission(cat.test_ids, proba, C, Path(args.out), set(cat.test_files))
report["changed_vs_plain_average"] = int((proba.argmax(1) != base_pred).sum())
report["label_counts"] = sub["Label"].value_counts().to_dict()
report["mean_max_proba"] = float(proba.max(1).mean())
if args.compare:
    other = pd.read_csv(args.compare, dtype=str).set_index("Id")["Label"]
    agree = (sub.set_index("Id")["Label"] == other.reindex(sub["Id"]).to_numpy()).mean()
    report["agreement_with_compare"] = float(agree)
Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
print("wrote", args.out)
