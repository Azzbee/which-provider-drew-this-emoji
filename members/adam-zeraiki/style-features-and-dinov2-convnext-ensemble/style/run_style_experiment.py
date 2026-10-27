#!/usr/bin/env python3
"""Bounded, train-partition-only style classifiers for emoji provider prediction."""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
from threadpoolctl import threadpool_limits

CLASSES = ["apple", "facebook", "google", "messenger", "mozilla", "samsung", "whatsapp"]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def normalized_hist(values, bins, weights=None):
    hist = np.histogram(values, bins=bins, range=(0, 1), weights=weights)[0].astype(np.float32)
    return hist / max(float(hist.sum()), 1e-12)

def feature_vector(path):
    # The original dimensions are used only to preserve aspect ratio during rendering.
    # They, filenames, file sizes and provider metadata are never features.
    with Image.open(path) as image:
        rgba = ImageOps.contain(image.convert("RGBA"), (64, 64), method=Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (64, 64), (255, 255, 255, 0))
        canvas.paste(rgba, ((64 - rgba.width) // 2, (64 - rgba.height) // 2))
        white = Image.new("RGBA", (64, 64), (255, 255, 255, 255))
        rgb_image = Image.alpha_composite(white, canvas).convert("RGB")
        alpha = np.asarray(canvas, dtype=np.float32)[..., 3] / 255
    rgb = np.asarray(rgb_image, dtype=np.float32) / 255
    hsv = np.asarray(rgb_image.convert("HSV"), dtype=np.float32) / 255
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    groups = []
    def add(name, array):
        groups.append((name, np.asarray(array, dtype=np.float32).ravel()))
    add("rgb16", np.asarray(rgb_image.resize((16, 16), Image.Resampling.BOX), dtype=np.float32) / 255)
    add("rgb8", np.asarray(rgb_image.resize((8, 8), Image.Resampling.BOX), dtype=np.float32) / 255)
    for channel in range(3):
        add(f"rgb_hist_{channel}", normalized_hist(rgb[..., channel], 16, alpha))
    add("hue_hist", normalized_hist(hsv[..., 0], 24, alpha * hsv[..., 1]))
    add("saturation_hist", normalized_hist(hsv[..., 1], 16, alpha))
    add("value_hist", normalized_hist(hsv[..., 2], 16, alpha))
    hs = np.histogram2d(hsv[..., 0].ravel(), hsv[..., 1].ravel(), bins=(8, 8),
                        range=((0, 1), (0, 1)), weights=alpha.ravel())[0].astype(np.float32)
    add("joint_hs", hs / max(float(hs.sum()), 1e-12))
    foreground = alpha > .1
    values = rgb[foreground] if foreground.any() else rgb.reshape(-1, 3)
    add("rgb_statistics", np.concatenate([values.mean(0), values.std(0), np.quantile(values, [.1, .25, .5, .75, .9], axis=0).ravel()]))
    add("alpha8", alpha.reshape(8, 8, 8, 8).mean(axis=(1, 3)))
    add("alpha_hist", normalized_hist(alpha, 16))
    add("alpha_summary", [alpha.mean(), alpha.std(), *(np.mean(alpha > x) for x in [.01, .1, .5, .9, .99])])

    gy, gx = np.gradient(gray)
    magnitude = np.sqrt(gx * gx + gy * gy)
    orientation = np.mod(np.arctan2(gy, gx), np.pi)
    bins = np.minimum((orientation * (8 / np.pi)).astype(np.int32), 7)
    hog = []
    energies = []
    for y in range(0, 64, 16):
        for x in range(0, 64, 16):
            cell_mag = magnitude[y:y+16, x:x+16]
            h = np.bincount(bins[y:y+16, x:x+16].ravel(), weights=cell_mag.ravel(), minlength=8)
            hog.extend(h / np.sqrt(np.square(h).sum() + 1e-8))
            energies.extend([cell_mag.mean(), cell_mag.std()])
    add("hog_4x4x8", hog)
    add("edge_cell_energy", energies)
    add("edge_quantiles", np.quantile(magnitude, [.5, .75, .9, .95, .99, 1]))
    lap = np.abs(gray[1:-1, 2:] + gray[1:-1, :-2] + gray[2:, 1:-1] + gray[:-2, 1:-1] - 4 * gray[1:-1, 1:-1])
    add("laplacian", [lap.mean(), lap.std(), *np.quantile(lap, [.5, .75, .9, .95, .99, 1])])

    center = gray[1:-1, 1:-1]
    offsets = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]
    bits = np.stack([gray[1+dy:63+dy, 1+dx:63+dx] >= center for dy, dx in offsets], axis=0)
    transitions = np.not_equal(bits, np.roll(bits, 1, axis=0)).sum(axis=0)
    lbp = np.where(transitions <= 2, bits.sum(axis=0), 9)
    texture_histograms = []
    for region in [lbp, lbp[:31,:31], lbp[:31,31:], lbp[31:,:31], lbp[31:,31:]]:
        h = np.bincount(region.ravel(), minlength=10).astype(np.float32)
        texture_histograms.extend(h / h.sum())
    add("uniform_lbp", texture_histograms)
    result = np.concatenate([a for _, a in groups])
    assert np.isfinite(result).all()
    slices = []
    start = 0
    for name, a in groups:
        slices.append({"name": name, "start": start, "end": start + len(a)})
        start += len(a)
    return result, slices

def extraction(paths, destination, schema_path):
    if destination.exists():
        result = np.load(destination, mmap_mode="r")
        if len(result) != len(paths):
            raise RuntimeError("Feature cache row count differs from requested IDs")
        return result
    started = time.monotonic()
    first, schema = feature_vector(paths[0])
    result = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float32, shape=(len(paths), len(first)))
    result[0] = first
    for i, path in enumerate(paths[1:], start=1):
        result[i], _ = feature_vector(path)
        if i % 1000 == 0:
            print(json.dumps({"stage": "features", "file": destination.name, "done": i, "total": len(paths), "seconds": round(time.monotonic() - started, 1)}), flush=True)
    result.flush()
    schema_path.write_text(json.dumps(schema, indent=2) + "\n")
    print(json.dumps({"stage": "features_complete", "file": destination.name, "shape": list(result.shape), "seconds": round(time.monotonic() - started, 1)}), flush=True)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dinov2-val", type=Path)
    parser.add_argument("--dinov2-test", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.split, newline="") as f:
        rows = list(csv.DictReader(f))
    with open(args.dataset / "train_labels.csv", newline="") as f:
        labels = {r["Id"].zfill(5): r["Label"] for r in csv.DictReader(f)}
    assert len(rows) == len(labels) == 9879
    assert all(labels[r["Id"].zfill(5)] == r["Label"] for r in rows)
    ids = [r["Id"].zfill(5) for r in rows]
    y = np.array([CLASSES.index(r["Label"]) for r in rows])
    train = np.array([r["split"] == "train" for r in rows])
    val = np.array([r["split"] == "validation" for r in rows])
    assert int(train.sum()) == 7903 and int(val.sum()) == 1976 and np.all(train ^ val)
    test_paths = sorted((args.dataset / "test").glob("*.png"), key=lambda p: int(p.stem))
    assert len(test_paths) == 9879
    test_ids = [p.stem for p in test_paths]
    (args.out / "train_ids.json").write_text(json.dumps(ids) + "\n")
    (args.out / "validation_ids.json").write_text(json.dumps(np.array(ids)[val].tolist()) + "\n")
    (args.out / "test_ids.json").write_text(json.dumps(test_ids) + "\n")
    (args.out / "classes.json").write_text(json.dumps(CLASSES) + "\n")
    np.save(args.out / "validation_targets.npy", y[val])

    x = extraction([args.dataset / "train" / f"{i}.png" for i in ids], args.out / "train_features.npy", args.out / "feature_schema.json")
    xt = extraction(test_paths, args.out / "test_features.npy", args.out / "feature_schema.json")
    schema = json.loads((args.out / "feature_schema.json").read_text())
    # Fixed settings chosen before observing validation scores; no search or refit.
    settings = {
        "extra_trees": {"n_estimators": 400, "max_features": .5, "min_samples_leaf": 2,
                        "random_state": 42, "n_jobs": 2},
        "hist_gradient_boosting": {"max_iter": 140, "learning_rate": .08, "max_leaf_nodes": 15,
                                  "max_bins": 64, "l2_regularization": 1., "min_samples_leaf": 20,
                                  "early_stopping": False, "random_state": 42}
    }
    provenance = {
        "task": "Emoji provider classification using rendered image style features",
        "classes": CLASSES, "train_rows": int(train.sum()), "validation_rows": int(val.sum()),
        "test_rows": len(test_paths), "all_label_refit": False, "kaggle_submitted": False,
        "split_sha256": sha256(args.split), "labels_sha256": sha256(args.dataset / "train_labels.csv"),
        "source_code_sha256": sha256(__file__), "settings": settings,
        "features": "White-composited, aspect-preserving 64x64 RGBA rendering; 16x16 and 8x8 RGB; foreground weighted RGB/HSV histograms; alpha rendering; HOG; edge energy; uniform LBP. No source dimensions, file size, filename, provider metadata or image matching used.",
        "numpy_version": np.__version__, "threads": 2,
        "note": "Validation is used only for reporting these two prespecified model attempts, not fitting or early stopping. Any subsequent model selection or blending on this split is exploratory."
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    metrics = {}
    for name, cls in [("extra_trees", ExtraTreesClassifier), ("hist_gradient_boosting", HistGradientBoostingClassifier)]:
        if (args.out / f"{name}_metrics.json").exists():
            metrics[name] = json.loads((args.out / f"{name}_metrics.json").read_text())
            continue
        selected = np.arange(x.shape[1]) if name == "extra_trees" else np.arange(schema[0]["end"], x.shape[1])
        started = time.monotonic()
        print(json.dumps({"stage": "fit_start", "model": name, "features": len(selected)}), flush=True)
        model = cls(**settings[name])
        with threadpool_limits(limits=2):
            model.fit(x[train][:, selected], y[train])
            assert np.array_equal(model.classes_, np.arange(7))
            val_prob = model.predict_proba(x[val][:, selected]).astype(np.float32)
            test_prob = model.predict_proba(xt[:, selected]).astype(np.float32)
        np.save(args.out / f"{name}_validation_probabilities.npy", val_prob)
        np.save(args.out / f"{name}_test_probabilities.npy", test_prob)
        report = {
            "accuracy": float(accuracy_score(y[val], val_prob.argmax(1))),
            "correct": int((y[val] == val_prob.argmax(1)).sum()), "total": int(val.sum()),
            "macro_f1": float(f1_score(y[val], val_prob.argmax(1), average="macro")),
            "log_loss": float(log_loss(y[val], val_prob)),
            "confusion_matrix": confusion_matrix(y[val], val_prob.argmax(1), labels=np.arange(7)).tolist(),
            "fit_and_inference_seconds": time.monotonic() - started,
            "feature_count": len(selected), "settings": settings[name],
            "all_label_refit": False,
            "validation_probability_sha256": sha256(args.out / f"{name}_validation_probabilities.npy"),
            "test_probability_sha256": sha256(args.out / f"{name}_test_probabilities.npy")
        }
        joblib.dump({"model": model, "feature_indices": selected, "classes": CLASSES,
                     "feature_schema": schema, "split_sha256": provenance["split_sha256"]},
                    args.out / f"{name}_model.joblib", compress=3)
        with open(args.out / f"submission_{name}_style.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Id", "Label"])
            writer.writerows((int(i), CLASSES[p]) for i, p in zip(test_ids, test_prob.argmax(1)))
        (args.out / f"{name}_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        metrics[name] = report
        print(json.dumps({"stage": "model_complete", "model": name, **report}), flush=True)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

if __name__ == "__main__":
    main()
