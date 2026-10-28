import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.config import CLASSES
from src.data import preprocess, split_indices
from src.models import build_model
from src.submission import validate


def _transparent_img(hidden=(76, 105, 113)) -> Image.Image:
    arr = np.zeros((72, 72, 4), np.uint8)
    arr[..., :3] = hidden                      # junk colour under transparency
    arr[20:50, 20:50] = (255, 0, 0, 255)       # opaque red square
    return Image.fromarray(arr, "RGBA")


def test_white_mode_hides_junk_colour():
    out = preprocess(_transparent_img(), "white", 72)
    assert out.shape == (72, 72, 3)
    assert tuple(out[0, 0]) == (255, 255, 255)
    assert tuple(out[35, 35]) == (255, 0, 0)


def test_naive_mode_exposes_junk_colour():
    assert tuple(preprocess(_transparent_img(), "naive", 72)[0, 0]) == (76, 105, 113)


def test_rgba_mode_zeroes_transparent_rgb_and_keeps_alpha():
    out = preprocess(_transparent_img(), "rgba", 72)
    assert out.shape == (72, 72, 4)
    assert tuple(out[0, 0]) == (0, 0, 0, 0)


def test_preprocess_unifies_size_and_palette_modes():
    p = Image.new("P", (240, 240))
    assert preprocess(p, "white", 96).shape == (96, 96, 3)


def test_bad_mode_raises():
    with pytest.raises(ValueError):
        preprocess(_transparent_img(), "grey", 64)  # type: ignore[arg-type]


def test_split_is_stratified_and_deterministic():
    y = np.array([0] * 80 + [1] * 20)
    tr1, va1 = split_indices(y)
    tr2, va2 = split_indices(y)
    assert (va1 == va2).all() and len(set(tr1) & set(va1)) == 0
    assert (y[va1] == 1).sum() == 4


def _sub(n=9879, start=10001):
    return pd.DataFrame({"Id": [str(i) for i in range(start, start + n)], "Label": [CLASSES[0]] * n})


def test_validate_accepts_good_submission():
    validate(_sub())


@pytest.mark.parametrize("df", [_sub(n=100), _sub(start=1), _sub().rename(columns={"Label": "Target"}),
                                _sub().assign(Label="0")])
def test_validate_rejects_bad_submission(df):
    with pytest.raises(ValueError):
        validate(df)


def test_freeze_leaves_only_head_trainable():
    m = build_model("resnet18", 3, 7, pretrained=False, freeze=True)
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith("fc.") for n in trainable)


def test_simple_cnn_forward_shape():
    m = build_model("simple_cnn", 4, 7, pretrained=False, freeze=False)
    assert m(torch.zeros(2, 4, 96, 96)).shape == (2, 7)


def test_excel_export_handles_missing_metrics(tmp_path, monkeypatch):
    import src.results as R
    monkeypatch.setattr(R, "EXPERIMENTS_DIR", tmp_path)
    monkeypatch.setattr(R, "RESULTS_XLSX", tmp_path / "out.xlsx")
    R.ExperimentRecord(exp_id="X_full", model="m", method="m", epochs=1, train_acc=0.99, train_prec=0.99,
                       val_acc=None, val_prec=None, explanation="", preprocessing="", preprocessing_why="").save()
    assert R.export_excel() == 1


def test_nearest_upscale_keeps_pixel_grid():
    arr = np.zeros((2, 2, 4), np.uint8)
    arr[0, 0] = (255, 0, 0, 255)
    out = preprocess(Image.fromarray(arr, "RGBA"), "white", 4, resample="nearest")
    assert set(map(tuple, out.reshape(-1, 3))) == {(255, 0, 0), (255, 255, 255)}  # no blended colours
