"""Fast unit tests on synthetic images (no dataset needed):  python -m pytest tests -q"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emoji_provider import data as D  # noqa: E402
from emoji_provider.models import EmojiCNNv2, MetaFusion  # noqa: E402
from emoji_provider.submit import exact_duplicate_lookup, write_submission  # noqa: E402


def _png(tmp_path: Path, name: str, size=(72, 68), under=(70, 96, 99), mode="RGBA") -> Path:
    """A transparent PNG with a coloured disc and a 'junk' colour stored under the transparent pixels."""
    w, h = size
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., :3] = under
    yy, xx = np.mgrid[:h, :w]
    disc = (xx - w / 2) ** 2 + (yy - h / 2) ** 2 < (min(w, h) / 3) ** 2
    arr[disc] = (200, 30, 30, 255)
    im = Image.fromarray(arr, "RGBA")
    if mode == "P":
        im = im.quantize(colors=16)
    p = tmp_path / name
    im.save(p)
    return p


def test_white_compositing_removes_under_colour_and_pads_to_square(tmp_path):
    p = _png(tmp_path, "a.png", size=(72, 68))
    x = D.load_rgb_white(p, 96)
    assert x.shape == (96, 96, 3) and x.dtype == np.uint8
    assert (x[0, 0] == 255).all() and (x[-1, -1] == 255).all()          # corners are white, not the junk colour
    assert not np.any(np.all(x == (70, 96, 99), axis=-1))              # junk teal never appears
    assert (x == 255).all(axis=-1).mean() > 0.5                         # mostly white canvas


def test_metadata_scan_reads_native_format(tmp_path):
    p = _png(tmp_path, "s.png", size=(72, 72), mode="P")
    m = D.scan_native_metadata(p)
    assert (m["width"], m["height"], m["mode"]) == (72, 72, "P")
    feats = D.metadata_features(pd.DataFrame([m]), level="format")
    assert feats.shape == (1, D.META_DIM)


def test_group_split_keeps_duplicates_together():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 7, size=300)
    groups = [f"g{i // 3}" for i in range(300)]                          # triplets of identical images
    tr, va = D.make_group_split(y, groups, seed=1)
    g = np.asarray(groups)
    assert set(g[tr]).isdisjoint(g[va]) and len(tr) + len(va) == 300


def test_augmentation_is_exact_pixel_and_shapes_match():
    px = np.full((4, 96, 96, 3), 255, dtype=np.uint8)
    px[:, 40:56, 40:56] = (10, 200, 30)
    ds = D.EmojiDataset(px, np.zeros(4, dtype=np.int64), augment=True, max_shift=4)
    x, y = ds[0]
    assert x.shape == (3, 96, 96)
    vals = torch.unique(((x * 0.5 + 0.5) * 255).round())
    assert set(vals.tolist()) <= {10.0, 30.0, 200.0, 255.0}             # no interpolated values


def test_models_forward():
    m = EmojiCNNv2(7)
    assert m(torch.zeros(2, 3, 96, 96)).shape == (2, 7)
    f = MetaFusion(EmojiCNNv2(7), meta_dim=D.META_DIM, n_classes=7)
    assert f(torch.zeros(2, 3, 96, 96), torch.zeros(2, D.META_DIM)).shape == (2, 7)


def test_submission_and_duplicate_rule(tmp_path):
    ids = ["10001", "10002", "10003"]
    proba = np.array([[.6, .4, 0, 0, 0, 0, 0], [.1, .9, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]])
    sub = write_submission(ids, proba, ["apple", "facebook", "google", "messenger", "mozilla", "samsung", "whatsapp"],
                           tmp_path / "s.csv", set(ids))
    assert sub["Label"].tolist() == ["apple", "facebook", "google"]
    adj, changed = exact_duplicate_lookup(["h1", "h2", "h3"], ["h1", "hx"], np.array([1, 0]), proba, 7)
    assert changed == 1 and adj[0].argmax() == 1
