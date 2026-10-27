"""Metrics and figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
                             f1_score)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})
PALETTE = {"apple": "#1f77b4", "facebook": "#2ca02c", "google": "#d62728", "messenger": "#9467bd",
           "mozilla": "#ff7f0e", "samsung": "#17becf", "whatsapp": "#8c564b"}


def summarize(y_true: np.ndarray, proba: np.ndarray, class_names: list[str]) -> dict:
    pred = proba.argmax(1)
    rep = classification_report(y_true, pred, labels=list(range(len(class_names))), target_names=class_names,
                                output_dict=True, zero_division=0)
    n = len(y_true)
    acc = accuracy_score(y_true, pred)
    return {"accuracy": float(acc), "accuracy_correct": int((pred == y_true).sum()), "n": int(n),
            "accuracy_stderr": float(np.sqrt(acc * (1 - acc) / n)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
            "macro_f1": float(f1_score(y_true, pred, average="macro")),
            "macro_precision": float(rep["macro avg"]["precision"]),
            "macro_recall": float(rep["macro avg"]["recall"]),
            "log_loss": float(-np.log(np.clip(proba[np.arange(n), y_true], 1e-12, 1)).mean()),
            "per_class": {c: rep[c] for c in class_names}}


def save_summary(summary: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(summary, indent=2))


def plot_confusion(y_true, pred, class_names, path: Path, title: str, normalize: bool = False) -> None:
    cm = confusion_matrix(y_true, pred, labels=list(range(len(class_names))))
    show = cm.astype(float) / cm.sum(1, keepdims=True) if normalize else cm
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(show, cmap="Blues", vmin=0, vmax=(1 if normalize else None))
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted provider"); ax.set_ylabel("true provider")
    thr = show.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            v = show[i, j]
            txt = f"{v:.2f}" if normalize else str(int(v))
            if (normalize and v >= 0.005) or (not normalize and v > 0):
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="white" if v > thr else "black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_learning_curves(history: pd.DataFrame, path: Path, title: str, best_epoch: int | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="validation")
    axes[0].set_title("cross-entropy loss"); axes[0].set_xlabel("epoch")
    axes[1].plot(history["epoch"], history["train_accuracy"], label="train")
    axes[1].plot(history["epoch"], history["val_accuracy"], label="validation")
    axes[1].set_title("accuracy"); axes[1].set_xlabel("epoch")
    for ax in axes:
        if best_epoch:
            ax.axvline(best_epoch, color="grey", ls=":", label="selected epoch")
        ax.legend(frameon=False)
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_per_class_f1(summaries: dict[str, dict], class_names: list[str], path: Path, title: str) -> None:
    """summaries: {model label: summary dict from summarize()}"""
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    k = len(summaries)
    width = 0.8 / k
    for m, (label, s) in enumerate(summaries.items()):
        vals = [s["per_class"][c]["f1-score"] for c in class_names]
        ax.bar(np.arange(len(class_names)) + (m - (k - 1) / 2) * width, vals, width, label=label)
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1.02); ax.set_ylabel("F1 (validation)")
    ax.legend(frameon=False, ncol=min(k, 3), fontsize=8)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_failures(pixels: np.ndarray, y_true, proba, class_names, ids, path: Path, title: str, n: int = 24) -> pd.DataFrame:
    """Grid of the most confident mistakes (true -> predicted, confidence)."""
    pred = proba.argmax(1)
    wrong = np.flatnonzero(pred != y_true)
    conf = proba[wrong, pred[wrong]]
    order = wrong[np.argsort(-conf)][:n]
    cols = 8
    rows = int(np.ceil(len(order) / cols)) or 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.95))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), order):
        ax.imshow(pixels[i]); ax.axis("off")
        ax.set_title(f"{ids[i]}\n{class_names[y_true[i]]}→{class_names[pred[i]]} {proba[i, pred[i]]:.2f}", fontsize=6.5)
    fig.suptitle(title, y=1.0)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    return pd.DataFrame({"Id": np.asarray(ids)[order], "true": [class_names[t] for t in y_true[order]],
                         "pred": [class_names[p] for p in pred[order]], "confidence": proba[order, pred[order]]})


# ----------------------------------------------------------------------------- Grad-CAM
class GradCAM:
    """Gradient-weighted class activation map on the last convolutional feature map."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model, self.acts, self.grads = model, None, None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.acts = o.detach()

    def _bwd(self, m, gi, go):
        self.grads = go[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None, meta: torch.Tensor | None = None) -> tuple[np.ndarray, int]:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        x = x.requires_grad_(True)
        logits = self.model(x, meta) if meta is not None else self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        logits[0, class_idx].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.acts).sum(1, keepdim=True))[0, 0]
        cam = cam / (cam.max() + 1e-8)
        cam = torch.nn.functional.interpolate(cam[None, None], size=x.shape[-2:], mode="bilinear",
                                              align_corners=False)[0, 0]
        return cam.cpu().numpy(), class_idx


def plot_gradcam_grid(model, target_layer, dataset, indices, class_names, path: Path, title: str,
                      device: torch.device, pixels_for_display: np.ndarray, y_true: np.ndarray) -> None:
    cam = GradCAM(model, target_layer)
    cols = min(7, len(indices))
    bands = int(np.ceil(len(indices) / cols))          # each band = a row of images + a row of heatmaps
    fig, axes = plt.subplots(2 * bands, cols, figsize=(cols * 1.6, bands * 3.6))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for k, i in enumerate(indices):
        r, c = 2 * (k // cols), k % cols
        item = dataset[i]
        x = item[0][None].to(device)
        meta = item[1][None].to(device) if len(item) == 3 else None
        heat, cls = cam(x, meta=meta)
        axes[r, c].imshow(pixels_for_display[i])
        axes[r, c].set_title(f"{class_names[y_true[i]]}", fontsize=8)
        axes[r + 1, c].imshow(pixels_for_display[i]); axes[r + 1, c].imshow(heat, cmap="jet", alpha=0.45)
        axes[r + 1, c].set_title(f"pred {class_names[cls]}", fontsize=8)
    fig.suptitle(title, y=1.0)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
