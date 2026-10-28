"""Image loading, preprocessing and train/validation split.

Preprocessing modes (the dataset mixes RGBA / P / LA / RGB / L PNGs of 72, 160 and 240 px):
- "naive": PIL ``convert("RGB")``. Drops alpha and exposes whatever colour sits under
  transparent pixels (vendor-specific junk such as (76,105,113) for Samsung) -> shortcut.
- "white": alpha-composite onto a white background. Same canvas for every vendor.
- "rgba":  keep 4 channels (alpha as its own channel), transparent RGB zeroed.
"""
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from src.config import CACHE_DIR, CLASS_TO_IDX, LABELS_CSV, SEED, TEST_DIR, TRAIN_DIR, VAL_RATIO

Mode = Literal["naive", "white", "rgba"]
MODES: tuple[str, ...] = ("naive", "white", "rgba")
Resample = Literal["bicubic", "nearest"]
RESAMPLE = {"bicubic": Image.Resampling.BICUBIC, "nearest": Image.Resampling.NEAREST}


def load_labels() -> pd.DataFrame:
    """Return train labels with columns Id (zero-padded str), Label (str), y (int)."""
    df = pd.read_csv(LABELS_CSV, dtype={"Id": str})
    unknown = set(df["Label"]) - set(CLASS_TO_IDX)
    if unknown:
        raise ValueError(f"Unknown labels in train_labels.csv: {unknown}")
    df["y"] = df["Label"].map(CLASS_TO_IDX).astype(int)
    return df


def test_ids() -> list[str]:
    return sorted(p.stem for p in TEST_DIR.glob("*.png"))


def preprocess(img: Image.Image, mode: Mode, size: int, resample: Resample = "bicubic") -> np.ndarray:
    """PIL image -> uint8 array (size, size, C), C=3 or 4 depending on mode.
    resample="nearest" keeps the original pixel grid when upscaling (no smoothing of outlines)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "naive":
        out = img.convert("RGB")
    else:
        rgba = img.convert("RGBA")
        if mode == "white":
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            out = Image.alpha_composite(bg, rgba).convert("RGB")
        else:
            arr = np.array(rgba)
            arr[arr[..., 3] == 0, :3] = 0
            out = Image.fromarray(arr, "RGBA")
    out = out.resize((size, size), RESAMPLE[resample])
    return np.asarray(out, dtype=np.uint8)


def load_array(split: Literal["train", "test"], mode: Mode, size: int, resample: Resample = "bicubic") -> np.ndarray:
    """Load every image of a split as (N, size, size, C) uint8, cached on disk as .npy."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if resample == "bicubic" else f"_{resample}"  # keeps existing bicubic cache names valid
    cache = CACHE_DIR / f"{split}_{mode}_{size}{suffix}.npy"
    if cache.exists():
        return np.load(cache)
    if split == "train":
        ids, folder = load_labels()["Id"].tolist(), TRAIN_DIR
    else:
        ids, folder = test_ids(), TEST_DIR
    arr = np.stack([preprocess(Image.open(folder / f"{i}.png"), mode, size, resample) for i in ids])
    np.save(cache, arr)
    return arr


def raw_metadata(folder: Path, ids: list[str]) -> pd.DataFrame:
    """File-level metadata (no pixels): width, height, PIL mode, colour hidden under alpha=0."""
    rows = []
    for i in ids:
        img = Image.open(folder / f"{i}.png")
        rgba = np.array(img.convert("RGBA"))
        hidden = rgba[rgba[..., 3] == 0, :3]
        h = hidden[0] if len(hidden) else np.array([-1, -1, -1])
        rows.append({"Id": i, "w": img.size[0], "h": img.size[1], "pil_mode": img.mode,
                     "hid_r": int(h[0]), "hid_g": int(h[1]), "hid_b": int(h[2])})
    return pd.DataFrame(rows)


def split_indices(y: np.ndarray, val_ratio: float = VAL_RATIO, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Stratified train/val split -> (train_idx, val_idx). Same seed => same split for every experiment."""
    idx = np.arange(len(y))
    tr, va = train_test_split(idx, test_size=val_ratio, stratify=y, random_state=seed)
    return np.sort(tr), np.sort(va)
