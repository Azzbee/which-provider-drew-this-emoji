"""Non-deep-learning experiments.

E1a  metadata-only baseline (file size / PIL mode / colour under transparent pixels, NO pixels)
     -> measures how much the dataset leaks the label through how the files were exported.
E1b  hand-crafted visual features (colour histogram + HOG) on the clean "white" images -> SVM.

Usage: python -m src.classical {meta|hog}
"""
import sys
import time

import numpy as np
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from src.config import SEED, TRAIN_DIR
from src.data import load_array, load_labels, raw_metadata, split_indices
from src.results import ExperimentRecord, scores


def meta_features(df_meta) -> np.ndarray:
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df_meta[["pil_mode"]])
    num = df_meta[["w", "h", "hid_r", "hid_g", "hid_b"]].to_numpy(float)
    return np.hstack([num, enc.transform(df_meta[["pil_mode"]])])


def hog_color_features(imgs: np.ndarray) -> np.ndarray:
    """imgs (N,H,W,3) uint8 -> [8x8x8 RGB histogram | HOG on grayscale]."""
    feats = []
    for im in imgs:
        q = (im // 32).reshape(-1, 3).astype(int)
        hist = np.bincount(q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2], minlength=512).astype(float)
        hist /= hist.sum()
        gray = im.mean(axis=2)
        h = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        feats.append(np.concatenate([hist, h]))
    return np.asarray(feats)


def run(kind: str) -> ExperimentRecord:
    df = load_labels()
    y = df["y"].to_numpy()
    tr, va = split_indices(y)
    t0 = time.time()
    if kind == "meta":
        X = meta_features(raw_metadata(TRAIN_DIR, df["Id"].tolist()))
        clf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    elif kind == "hog":
        X = hog_color_features(load_array("train", "white", 64))
        clf = make_pipeline(StandardScaler(), SVC(C=10, kernel="rbf", random_state=SEED))
    else:
        raise ValueError(kind)
    clf.fit(X[tr], y[tr])
    tr_acc, tr_prec = scores(y[tr], clf.predict(X[tr]))
    va_acc, va_prec = scores(y[va], clf.predict(X[va]))
    extra = {"seed": SEED, "n_features": int(X.shape[1]), "runtime_s": round(time.time() - t0, 1)}
    if kind == "meta":
        rec = ExperimentRecord(
            exp_id="E1a_metadata_only", model="Random Forest (300 trees)",
            method="Classifier on file metadata only, no pixel content", epochs=None,
            train_acc=tr_acc, train_prec=tr_prec, val_acc=va_acc, val_prec=va_prec,
            explanation="",
            preprocessing="Features: image width/height, PIL colour mode (RGBA/P/LA/RGB/L), "
                          "RGB value stored under fully transparent pixels",
            preprocessing_why="Diagnostic: EDA showed size, colour mode and the hidden colour under "
                              "transparent pixels differ strongly by vendor (export artefacts, not visual style). "
                              "This measures how much accuracy is reachable WITHOUT looking at the emoji.",
            extra=extra)
    else:
        rec = ExperimentRecord(
            exp_id="E1b_hog_color_svm", model="SVM (RBF, C=10)",
            method="Hand-crafted features: 512-bin RGB colour histogram + HOG (9 orient., 8x8 cells)",
            epochs=None, train_acc=tr_acc, train_prec=tr_prec, val_acc=va_acc, val_prec=va_prec,
            explanation="",
            preprocessing="Alpha-composited on white background, resized to 64x64 (bicubic)",
            preprocessing_why="White compositing removes the hidden-colour shortcut and gives every vendor "
                              "the same canvas; 64 px keeps HOG dimensionality manageable for an SVM.",
            extra=extra)
    return rec


if __name__ == "__main__":
    r = run(sys.argv[1])
    r.save()
    print(r.exp_id, f"train acc={r.train_acc:.4f}  val acc={r.val_acc:.4f}  val prec={r.val_prec:.4f}", r.extra)
