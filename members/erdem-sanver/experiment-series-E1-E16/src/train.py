"""Train one CNN experiment end-to-end: fit on the 80% split, score on the 20% validation split,
predict the Kaggle test set, save record (+ Excel row source), probabilities and a submission CSV.

Example:
  python -m src.train --exp E2_simple_cnn_naive --model simple_cnn --mode naive --size 96 --epochs 15 \
      --why "..." --method "..."
"""
import argparse
import json
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from src.config import CLASSES, EXPERIMENTS_DIR, SEED
from src.data import load_array, load_labels, split_indices
from src.models import build_model, count_params
from src.results import ExperimentRecord, scores
from src.submission import write_submission

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray | None, tf: v2.Transform) -> None:
        self.x, self.y, self.tf = x, y, tf

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        img = self.tf(torch.from_numpy(self.x[i]).permute(2, 0, 1))
        return img, (-1 if self.y is None else int(self.y[i]))


def transforms(n_ch: int, size: int, aug: str) -> tuple[v2.Transform, v2.Transform]:
    mean = IMAGENET_MEAN + [0.5] * (n_ch - 3)
    std = IMAGENET_STD + [0.5] * (n_ch - 3)
    base = [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)]
    if aug == "none":
        train_aug = []
    elif aug == "geo":   # geometry only: colour is a vendor signal, so no colour jitter
        train_aug = [v2.RandomHorizontalFlip(),
                     v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1))]
    else:
        raise ValueError(aug)
    return v2.Compose(train_aug + base), v2.Compose(base)


@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader, dev: torch.device, tta: bool) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
        x = x.to(dev)
        p = model(x).softmax(1)
        if tta:
            p = (p + model(torch.flip(x, dims=[3])).softmax(1)) / 2
        out.append(p.cpu())
    return torch.cat(out).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--model", default="simple_cnn")
    ap.add_argument("--mode", default="white", choices=["naive", "white", "rgba"])
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--resample", default="bicubic", choices=["bicubic", "nearest"])
    ap.add_argument("--seed", type=int, default=SEED, help="training seed (the train/val split always uses config.SEED)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--aug", default="none", choices=["none", "geo"])
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--pseudo", help="experiment id whose saved TEST probabilities become pseudo-labels")
    ap.add_argument("--pseudo-thr", type=float, default=0.99, help="min confidence for a test image to be pseudo-labelled")
    ap.add_argument("--full", action="store_true", help="train on 100%% of train (no validation) for final submit")
    ap.add_argument("--method", required=True, help="Excel 'Method Tried'")
    ap.add_argument("--preproc", required=True, help="Excel 'Data preprocessing'")
    ap.add_argument("--why", required=True, help="Excel 'Explain' (why this preprocessing)")
    a = ap.parse_args()

    seed_everything(a.seed)
    dev = device()
    df = load_labels()
    y = df["y"].to_numpy()
    X = load_array("train", a.mode, a.size, a.resample)
    Xt = load_array("test", a.mode, a.size, a.resample)
    tr, va = split_indices(y)
    if a.full:
        tr = np.arange(len(y))
    X_fit, y_fit = X[tr], y[tr]
    n_pseudo = 0
    if a.pseudo:  # semi-supervised: add confidently predicted test images with their predicted class
        p_src = np.load(EXPERIMENTS_DIR / "preds" / f"{a.pseudo}_test.npy")
        keep = p_src.max(1) >= a.pseudo_thr
        X_fit, y_fit = np.concatenate([X_fit, Xt[keep]]), np.concatenate([y_fit, p_src[keep].argmax(1)])
        n_pseudo = int(keep.sum())
        print(f"[{a.exp}] pseudo-labels from {a.pseudo}: {n_pseudo} test images (conf >= {a.pseudo_thr})", flush=True)
    n_ch = X.shape[-1]
    tf_train, tf_eval = transforms(n_ch, a.size, a.aug)
    g = torch.Generator().manual_seed(a.seed)
    dl_tr = DataLoader(ArrayDataset(X_fit, y_fit, tf_train), batch_size=a.bs, shuffle=True, generator=g)
    dl_tr_eval = DataLoader(ArrayDataset(X[tr], y[tr], tf_eval), batch_size=256)
    dl_va = DataLoader(ArrayDataset(X[va], y[va], tf_eval), batch_size=256)
    dl_te = DataLoader(ArrayDataset(Xt, None, tf_eval), batch_size=256)

    model = build_model(a.model, n_ch, len(CLASSES), a.pretrained, a.freeze).to(dev)
    n_total, n_train = count_params(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(dl_tr), pct_start=0.15)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)

    history = []
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        if a.freeze:  # keep frozen BatchNorm statistics fixed
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
                    m.eval()
        tot, n = 0.0, 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(dev), yb.to(dev)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        # --full: the validation images are in the training set, so val accuracy would be fake -> not computed
        va_acc = None if a.full else scores(y[va], predict_proba(model, dl_va, dev, False).argmax(1))[0]
        history.append({"epoch": ep, "loss": round(tot / n, 4), "val_acc": None if va_acc is None else round(va_acc, 4)})
        shown = "n/a (full)" if va_acc is None else f"{va_acc:.4f}"
        print(f"[{a.exp}] ep {ep:2d}/{a.epochs} loss {tot / n:.4f} val_acc {shown}  ({time.time() - t0:.0f}s)", flush=True)

    tr_acc, tr_prec = scores(y[tr], predict_proba(model, dl_tr_eval, dev, False).argmax(1))
    p_va = None if a.full else predict_proba(model, dl_va, dev, a.tta)
    va_acc, va_prec = (None, None) if p_va is None else scores(y[va], p_va.argmax(1))
    p_te = predict_proba(model, dl_te, dev, a.tta)

    pred_dir = EXPERIMENTS_DIR / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    if p_va is not None:
        np.save(pred_dir / f"{a.exp}_val.npy", p_va)
    np.save(pred_dir / f"{a.exp}_test.npy", p_te)
    sub = write_submission(p_te.argmax(1), a.exp)

    from sklearn.metrics import confusion_matrix
    rec = ExperimentRecord(
        exp_id=a.exp,
        model=a.model + (" (ImageNet pretrained" + (", frozen backbone)" if a.freeze else ")") if a.pretrained else " (from scratch)"),
        method=a.method, epochs=a.epochs,
        train_acc=tr_acc, train_prec=tr_prec, val_acc=va_acc, val_prec=va_prec,
        explanation="", preprocessing=a.preproc, preprocessing_why=a.why,
        ratio_train=1.0 if a.full else 0.8, ratio_test=0.0 if a.full else 0.2,
        extra={"seed": a.seed, "args": vars(a), "params_total": n_total, "n_pseudo": n_pseudo, "params_trainable": n_train,
               "device": str(dev), "runtime_s": round(time.time() - t0, 1), "history": history,
               "val_confusion": None if p_va is None else confusion_matrix(y[va], p_va.argmax(1)).tolist(),
               "submission": sub.name})
    rec.save()
    print(json.dumps({"exp": a.exp, "train_acc": round(tr_acc, 4), "val_acc": va_acc, "val_prec": va_prec,
                      "runtime_s": rec.extra["runtime_s"]}))


if __name__ == "__main__":
    main()
