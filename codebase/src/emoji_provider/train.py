"""Training loop, checkpointing and prediction."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    onecycle: bool = True
    patience: int = 0               # 0 = no early stopping (OneCycle schedules need the full budget)
    select_on: str = "val_loss"      # "val_loss" (lowest), "val_accuracy" (highest) or "last" (keep the final epoch)
    amp: bool = True                 # bf16/fp16 autocast on CUDA only
    num_workers: int = 0
    seed: int = 42
    notes: str = ""
    extra: dict = field(default_factory=dict)


def package_versions() -> dict:
    """Versions of the packages that matter for reproducing a run (stored in every *_run.json)."""
    import platform
    from importlib import metadata
    out = {"python": platform.python_version(), "torch": torch.__version__,
           "cuda": torch.version.cuda, "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    for pkg in ("torchvision", "timm", "scikit-learn", "numpy", "pandas", "pillow"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def _forward(model: nn.Module, batch: tuple, device: torch.device):
    if len(batch) == 3:          # (x, meta, y)
        x, m, y = batch
        return model(x.to(device, non_blocking=True), m.to(device, non_blocking=True)), y.to(device)
    x, y = batch
    return model(x.to(device, non_blocking=True)), y.to(device)


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None, scaler=None, amp=False):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits, y = _forward(model, batch, device)
                loss = criterion(logits, y)
            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            total_loss += loss.item() * y.size(0)
            correct += int((logits.argmax(1) == y).sum().item())
            n += y.size(0)
    return total_loss / n, correct / n


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, cfg: TrainConfig,
        device: torch.device, out_dir: Path, class_names: list[str], tag: str = "model") -> pd.DataFrame:
    """Train, keep the best checkpoint (by cfg.select_on) and write history.csv + run.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * len(train_loader)
    # OneCycle needs a few steps per phase; a smoke run with a handful of batches falls back to a constant LR
    scheduler = (torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr, total_steps=steps, pct_start=0.25)
                 if cfg.onecycle and steps >= 16 else None)
    scaler = torch.amp.GradScaler("cuda") if (cfg.amp and device.type == "cuda") else None
    history, best, best_epoch, stale = [], None, 0, 0
    ckpt = out_dir / f"{tag}.pt"
    t0 = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        te = time.perf_counter()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer, scheduler, scaler, cfg.amp)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        rec = {"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "train_accuracy": tr_acc,
               "val_accuracy": va_acc, "lr": optimizer.param_groups[0]["lr"], "seconds": time.perf_counter() - te}
        history.append(rec)
        pd.DataFrame(history).to_csv(out_dir / f"{tag}_history.csv", index=False)
        print(f"[{tag}] epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3%} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3%} | {rec['seconds']:.0f}s", flush=True)
        score = {"val_loss": -va_loss, "val_accuracy": va_acc, "last": float(epoch)}[cfg.select_on]
        if best is None or score > best:
            best, best_epoch, stale = score, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "class_names": class_names, "epoch": epoch,
                        "val_loss": va_loss, "val_accuracy": va_acc, "config": asdict(cfg)}, ckpt)
        else:
            stale += 1
            if cfg.patience and stale >= cfg.patience:
                print(f"[{tag}] early stopping; best epoch {best_epoch}")
                break
    seconds = time.perf_counter() - t0
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    run = {"tag": tag, "best_epoch": best_epoch, "epochs_run": len(history), "training_seconds": seconds,
           "device": str(device), "best_val_loss": state["val_loss"], "best_val_accuracy": state["val_accuracy"],
           "config": asdict(cfg), "class_names": class_names, "versions": package_versions()}
    (out_dir / f"{tag}_run.json").write_text(json.dumps(run, indent=2))
    print(f"[{tag}] restored epoch {best_epoch} (val acc {state['val_accuracy']:.4%}); {seconds / 60:.1f} min")
    return pd.DataFrame(history)


@torch.inference_mode()
def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device, tta_hflip: bool = False) -> np.ndarray:
    """Softmax probabilities; with tta_hflip the image and its mirror are averaged."""
    model.eval().to(device)
    out = []
    for batch in loader:
        has_meta = len(batch) == 3 or (len(batch) == 2 and batch[1].dtype == torch.float32 and batch[1].ndim == 2)
        x = batch[0].to(device)
        meta = batch[1].to(device) if has_meta else None
        logits = model(x, meta) if meta is not None else model(x)
        p = torch.softmax(logits.float(), dim=1)
        if tta_hflip:
            xf = torch.flip(x, dims=[3])
            lf = model(xf, meta) if meta is not None else model(xf)
            p = 0.5 * (p + torch.softmax(lf.float(), dim=1))
        out.append(p.cpu().numpy())
    return np.concatenate(out)

