"""Write and validate Kaggle submission files (Id,Label)."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CLASSES, SUBMISSIONS_DIR
from src.data import test_ids

EXPECTED_ROWS = 9879


def write_submission(pred_idx: np.ndarray, name: str) -> Path:
    """pred_idx: class indices aligned with sorted test ids. Labels written as class names
    (same vocabulary as train_labels.csv)."""
    ids = test_ids()
    if len(pred_idx) != len(ids):
        raise ValueError(f"{len(pred_idx)} predictions for {len(ids)} test images")
    df = pd.DataFrame({"Id": ids, "Label": [CLASSES[int(i)] for i in pred_idx]})
    validate(df)
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    path = SUBMISSIONS_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def validate(df: pd.DataFrame) -> None:
    """Fail loudly on anything Kaggle would score as 0 (wrong header, ids, labels, row count)."""
    if list(df.columns) != ["Id", "Label"]:
        raise ValueError(f"Header must be Id,Label, got {list(df.columns)}")
    if len(df) != EXPECTED_ROWS or df["Id"].duplicated().any():
        raise ValueError(f"Expected {EXPECTED_ROWS} unique ids, got {len(df)} ({df['Id'].duplicated().sum()} dup)")
    expected = {str(i) for i in range(10001, 10001 + EXPECTED_ROWS)}
    if set(df["Id"].astype(str)) != expected:
        raise ValueError("Ids are not exactly 10001..19879")
    bad = set(df["Label"]) - set(CLASSES)
    if bad:
        raise ValueError(f"Unknown labels: {bad}")
