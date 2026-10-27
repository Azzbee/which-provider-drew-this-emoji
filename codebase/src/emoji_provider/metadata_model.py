"""Metadata-only diagnostic: how much provider information is in the file format alone (no pixels).

This is a measurement, not a competition model. It quantifies the shortcut that any pixel model could
pick up through resampling artefacts, and it motivates the white-compositing / uniform-canvas preprocessing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import metadata_features


def run_metadata_diagnostic(meta: pd.DataFrame, y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray,
                            seed: int = 42) -> dict[str, dict]:
    """Returns {variant: {'val_accuracy', 'val_proba', 'model'}} for size+mode only and size+mode+under-colour."""
    results = {}
    for variant, level in (("size+mode", "format"), ("size+mode+file stats", "file"), ("size+mode+file stats+under-colour", "under-colour")):
        X = metadata_features(meta, level=level)
        for name, clf in (("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))),
                          ("random_forest", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1))):
            clf.fit(X[train_idx], y[train_idx])
            proba = clf.predict_proba(X[val_idx])
            acc = float((proba.argmax(1) == y[val_idx]).mean())
            results[f"{variant} / {name}"] = {"val_accuracy": acc, "val_proba": proba, "model": clf,
                                              "n_features": X.shape[1]}
    return results


def size_family(width: int, height: int) -> str:
    """Coarse native-resolution family used for the interpretability tables."""
    m = max(width, height)
    if m <= 72:
        return "<=72"
    if m <= 108:
        return "108"
    if m <= 160:
        return "160"
    return "240"
