"""Data access, preprocessing, caching, deduplication and splitting."""
from __future__ import annotations

import hashlib
import os
import random
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

ZIP_NAME = "2-computer-vision-2026-b-sc-aidams-final-proj.zip"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODES = ("RGBA", "P", "LA", "RGB", "L")  # every PIL mode observed in the dataset


# ----------------------------------------------------------------------------- reproducibility
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    return torch.device("cpu")


# ----------------------------------------------------------------------------- archive
def find_zip(configured: str | os.PathLike | None = None) -> Path:
    """Return the Kaggle archive: the configured path, else the usual Colab/local locations."""
    if configured:
        p = Path(configured).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"ZIP_PATH does not exist: {p}")
        return p
    for cand in (Path("/content") / ZIP_NAME, Path.cwd() / ZIP_NAME, Path.cwd() / "data" / ZIP_NAME,
                 Path.home() / "Downloads" / ZIP_NAME):
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"Put {ZIP_NAME} in /content (Colab), next to the notebook, in ./data, or set ZIP_PATH explicitly.")


def extract_zip(archive: Path, destination: Path) -> Path:
    """Extract once (skipped when train_labels.csv is already there). Rejects path traversal."""
    destination = Path(destination)
    if (destination / "train_labels.csv").is_file() and (destination / "train").is_dir():
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as z:
        for m in z.infolist():
            if not (root / m.filename).resolve().is_relative_to(root):
                raise ValueError(f"Unsafe path in archive: {m.filename}")
        z.extractall(root)
    return destination


# ----------------------------------------------------------------------------- labels and files
@dataclass
class Catalogue:
    data_dir: Path
    labels: pd.DataFrame          # Id, Label (sorted by Id)
    train_files: dict[str, Path]
    test_files: dict[str, Path]
    class_names: list[str]

    @property
    def class_to_index(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.class_names)}

    @property
    def y(self) -> np.ndarray:
        return self.labels["Label"].map(self.class_to_index).to_numpy(dtype=np.int64)

    @property
    def train_ids(self) -> list[str]:
        return self.labels["Id"].tolist()

    @property
    def test_ids(self) -> list[str]:
        return sorted(self.test_files)


def load_catalogue(data_dir: Path) -> Catalogue:
    data_dir = Path(data_dir)
    labels = pd.read_csv(data_dir / "train_labels.csv", dtype={"Id": str, "Label": str})
    assert labels.columns.tolist() == ["Id", "Label"], "unexpected label columns"
    assert labels.notna().all().all() and labels["Id"].is_unique
    train_files = {p.stem: p for p in (data_dir / "train").glob("*.png")}
    test_files = {p.stem: p for p in (data_dir / "test").glob("*.png")}
    assert set(labels["Id"]) == set(train_files), "label IDs and train files differ"
    assert set(train_files).isdisjoint(test_files), "train/test ID overlap"
    labels = labels.sort_values("Id").reset_index(drop=True)
    class_names = sorted(labels["Label"].unique().tolist())
    return Catalogue(data_dir, labels, train_files, test_files, class_names)


# ----------------------------------------------------------------------------- native metadata
def scan_native_metadata(path: Path) -> dict:
    """What the PNG file says before any preprocessing (used for EDA and the metadata diagnostic)."""
    raw = path.read_bytes()
    with Image.open(path) as im:
        w, h, mode = im.width, im.height, im.mode
        has_transparency_key = "transparency" in im.info
        arr = np.asarray(im.convert("RGBA"))
    alpha = arr[..., 3]
    transparent = alpha == 0
    if transparent.any():
        under = arr[transparent][:, :3].astype(np.float64).mean(axis=0)
    else:
        under = np.full(3, np.nan)
    return {"Id": path.stem, "width": w, "height": h, "mode": mode,
            "has_alpha": bool(mode in ("RGBA", "LA") or has_transparency_key),
            "frac_transparent": float(transparent.mean()),
            "under_r": under[0], "under_g": under[1], "under_b": under[2],
            "bytes": len(raw), "raw_sha256": hashlib.sha256(raw).hexdigest()}


def scan_all_metadata(paths: list[Path], log_every: int = 2000) -> pd.DataFrame:
    rows = []
    for i, p in enumerate(paths):
        rows.append(scan_native_metadata(p))
        if log_every and (i + 1) % log_every == 0:
            print(f"  scanned {i + 1:,}/{len(paths):,}", flush=True)
    return pd.DataFrame(rows)


def metadata_features(meta: pd.DataFrame, level: str = "format", include_under_colour: bool | None = None) -> np.ndarray:
    """Numeric feature matrix for the metadata-only diagnostic and for late fusion.

    level = "format"      : log2(width), log2(height), one-hot PIL mode (5), has_alpha          -> 8 features
            "file"        : + fraction of transparent pixels, log10(file size in bytes)        -> 10 features
            "under-colour": + mean RGB stored under fully transparent pixels / 255             -> 13 features
    The fusion model uses "format" only (what the container says, nothing derived from the drawing).
    """
    if include_under_colour:  # backward compatibility
        level = "under-colour"
    cols = [np.log2(meta["width"].to_numpy(float)), np.log2(meta["height"].to_numpy(float))]
    for m in MODES:
        cols.append((meta["mode"] == m).to_numpy(float))
    cols.append(meta["has_alpha"].to_numpy(float))
    if level in ("file", "under-colour"):
        cols.append(meta["frac_transparent"].to_numpy(float))
        cols.append(np.log10(meta["bytes"].to_numpy(float)))
    if level == "under-colour":
        for c in ("under_r", "under_g", "under_b"):
            cols.append(np.nan_to_num(meta[c].to_numpy(float), nan=255.0) / 255.0)
    return np.stack(cols, axis=1).astype(np.float32)


META_DIM = 2 + len(MODES) + 1  # "format" level


# ----------------------------------------------------------------------------- pixels
def load_rgb_white(path: Path, size: int) -> np.ndarray:
    """Alpha-composite on white, fit inside size x size keeping aspect ratio, pad with white.

    This is the one preprocessing recipe used by every model in the project. It shows the emoji
    the way a chat app renders it on a light background, and it removes the colour that PNG
    encoders leave under fully transparent pixels (which differs by provider).
    """
    with Image.open(path) as im:
        rgba = im.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(white, rgba).convert("RGB")
    fitted = ImageOps.contain(rgb, (size, size), Image.Resampling.LANCZOS)
    square = Image.new("RGB", (size, size), (255, 255, 255))
    square.paste(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
    return np.asarray(square, dtype=np.uint8).copy()


def load_pixels(paths: list[Path], size: int, cache: Path | None = None, log_every: int = 2000) -> np.ndarray:
    """uint8 array (N, size, size, 3). Cached as .npy so a rerun does not decode 20k PNGs again."""
    if cache is not None and Path(cache).is_file():
        arr = np.load(cache)
        if arr.shape == (len(paths), size, size, 3):
            return arr
    arr = np.empty((len(paths), size, size, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        arr[i] = load_rgb_white(p, size)
        if log_every and (i + 1) % log_every == 0:
            print(f"  loaded {i + 1:,}/{len(paths):,} images at {size}px", flush=True)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, arr)
    return arr


def pixel_hashes(pixels: np.ndarray) -> list[str]:
    return [hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest() for p in pixels]


# ----------------------------------------------------------------------------- split
def make_group_split(y: np.ndarray, groups: list[str], seed: int = 42, n_splits: int = 5,
                     fold: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Stratified, group-aware split: identical images (same preprocessed pixels) never straddle the split."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(y)), y, groups=groups))
    train_idx, val_idx = folds[fold]
    g = np.asarray(groups)
    assert set(g[train_idx]).isdisjoint(g[val_idx]), "a duplicate image crossed the split"
    return train_idx, val_idx


def load_fixed_split(split_csv: Path, labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Reuse a stored split.csv (Id, Label, split) so every experiment is scored on the same images."""
    s = pd.read_csv(split_csv, dtype={"Id": str, "Label": str, "split": str})
    merged = labels.merge(s, on="Id", how="left", suffixes=("", "_stored"))
    assert merged["split"].notna().all(), "split.csv does not cover every training image"
    assert (merged["Label"] == merged["Label_stored"]).all(), "labels differ from stored split"
    train_idx = np.flatnonzero(merged["split"].to_numpy() == "train")
    val_idx = np.flatnonzero(merged["split"].to_numpy() == "validation")
    return train_idx, val_idx


def duplicate_report(hashes: list[str], y: np.ndarray) -> dict:
    counts = Counter(hashes)
    df = pd.DataFrame({"h": hashes, "y": y})
    conflicting = int((df.groupby("h")["y"].nunique() > 1).sum())
    return {"duplicate_images_beyond_first": int(sum(c - 1 for c in counts.values())),
            "conflicting_duplicate_groups": conflicting}


# ----------------------------------------------------------------------------- datasets
class EmojiDataset(Dataset):
    """Tensors from the uint8 cache. Augmentation never interpolates: horizontal flip and integer
    translation (pad with white, random crop) keep anti-aliased outlines and gradients intact,
    because those are the provider's signature we want the network to learn."""

    def __init__(self, pixels: np.ndarray, targets: np.ndarray | None = None, meta: np.ndarray | None = None,
                 augment: bool = False, max_shift: int = 4, hflip: bool = True,
                 imagenet_norm: bool = False, out_size: int | None = None):
        self.pixels = pixels
        self.targets = None if targets is None else torch.as_tensor(targets, dtype=torch.long)
        self.meta = None if meta is None else torch.as_tensor(meta, dtype=torch.float32)
        self.augment, self.max_shift, self.hflip = augment, max_shift, hflip
        self.out_size = out_size
        mean = torch.tensor(IMAGENET_MEAN if imagenet_norm else (0.5, 0.5, 0.5)).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD if imagenet_norm else (0.5, 0.5, 0.5)).view(3, 1, 1)
        self.mean, self.std = mean, std

    def __len__(self) -> int:
        return len(self.pixels)

    def __getitem__(self, i: int):
        x = torch.from_numpy(self.pixels[i]).permute(2, 0, 1).float() / 255.0  # (3, H, W)
        if self.augment:
            if self.hflip and torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[2])
            if self.max_shift > 0:
                s = self.max_shift
                padded = torch.nn.functional.pad(x, (s, s, s, s), value=1.0)  # white border
                dx, dy = torch.randint(0, 2 * s + 1, (2,)).tolist()
                x = padded[:, dy:dy + x.shape[1], dx:dx + x.shape[2]]
        if self.out_size is not None and self.out_size != x.shape[-1]:
            x = torch.nn.functional.interpolate(x[None], size=(self.out_size, self.out_size),
                                                mode="bicubic", align_corners=False)[0].clamp(0, 1)
        x = (x - self.mean) / self.std
        items = [x]
        if self.meta is not None:
            items.append(self.meta[i])
        if self.targets is not None:
            items.append(self.targets[i])
        return tuple(items)
