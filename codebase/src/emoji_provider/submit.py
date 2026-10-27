"""Submission writer and the two optional, clearly-flagged post-processing rules."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_submission(test_ids: list[str], proba: np.ndarray, class_names: list[str], path: Path,
                     expected_ids: set[str] | None = None) -> pd.DataFrame:
    pred = proba.argmax(1)
    sub = pd.DataFrame({"Id": list(test_ids), "Label": [class_names[i] for i in pred]})
    assert sub.columns.tolist() == ["Id", "Label"]
    assert sub["Id"].is_unique and sub.notna().all().all()
    assert set(sub["Label"]).issubset(class_names)
    if expected_ids is not None:
        assert set(sub["Id"]) == set(expected_ids), "submission IDs do not match the test folder"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    back = pd.read_csv(path, dtype=str)
    pd.testing.assert_frame_equal(sub, back)
    return sub


def exact_duplicate_lookup(test_hashes: list[str], train_hashes: list[str], y_train: np.ndarray,
                           proba: np.ndarray, n_classes: int) -> tuple[np.ndarray, int]:
    """Optional rule: a test file byte-identical to a training file gets that training label.

    Only applied when every training copy agrees on the label. Returns the adjusted probabilities and
    the number of predictions that changed.
    """
    lut: dict[str, set[int]] = {}
    for h, label in zip(train_hashes, y_train):
        lut.setdefault(h, set()).add(int(label))
    adjusted = proba.copy()
    changed = 0
    for i, h in enumerate(test_hashes):
        labels = lut.get(h)
        if labels and len(labels) == 1:
            label = next(iter(labels))
            if adjusted[i].argmax() != label:
                changed += 1
            adjusted[i] = np.eye(n_classes)[label]
    return adjusted, changed


def consistency_within_test_duplicates(test_hashes: list[str], proba: np.ndarray) -> np.ndarray:
    """Optional rule: identical test images share one averaged prediction (removes contradictory labels)."""
    df = pd.DataFrame(proba)
    df["h"] = test_hashes
    mean = df.groupby("h").transform("mean")
    return mean.to_numpy()
