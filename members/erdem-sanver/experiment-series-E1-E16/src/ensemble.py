"""Soft-voting ensemble: average saved softmax probabilities of finished experiments.

  python -m src.ensemble E9_ens_convnext128_224 E6b_convnext_tiny_finetune E8_convnext_tiny_224
"""
import sys

import numpy as np

from src.config import EXPERIMENTS_DIR
from src.data import load_labels, split_indices
from src.results import ExperimentRecord, scores
from src.submission import write_submission


def main() -> None:
    exp_id, members = sys.argv[1], sys.argv[2:]
    if len(members) < 2:
        raise SystemExit("need at least 2 member experiments")
    pred_dir = EXPERIMENTS_DIR / "preds"
    p_te = sum(np.load(pred_dir / f"{m}_test.npy") for m in members) / len(members)
    val_files = [pred_dir / f"{m}_val.npy" for m in members]
    va_acc = va_prec = None  # members trained with --full have no held-out predictions
    if all(f.exists() for f in val_files):
        p_va = sum(np.load(f) for f in val_files) / len(members)
        y = load_labels()["y"].to_numpy()
        _, va = split_indices(y)
        va_acc, va_prec = scores(y[va], p_va.argmax(1))
    np.save(pred_dir / f"{exp_id}_test.npy", p_te)
    if va_acc is not None:
        np.save(pred_dir / f"{exp_id}_val.npy", p_va)
    sub = write_submission(p_te.argmax(1), exp_id)
    ExperimentRecord(
        exp_id=exp_id, model="Ensemble: " + " + ".join(members),
        method="Soft voting: average of the members' softmax probabilities (equal weights)", epochs=None,
        train_acc=None, train_prec=None, val_acc=va_acc, val_prec=va_prec, explanation="",
        preprocessing="As in each member experiment",
        preprocessing_why="Members see the image at different resolutions, so their errors are partly independent.",
        extra={"members": members, "submission": sub.name}).save()
    print(f"{exp_id}: val acc {va_acc}  macro prec {va_prec}  -> {sub.name}")


if __name__ == "__main__":
    main()
