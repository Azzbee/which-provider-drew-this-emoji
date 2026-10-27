#!/usr/bin/env python3
"""Run an existing emoji style classifier without training or network access."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from run_style_experiment import CLASSES, feature_vector, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset root containing test/*.png")
    parser.add_argument("--model", type=Path, required=True, help="Trusted saved *_model.joblib")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for probabilities, IDs, CSV and receipt")
    parser.add_argument("--features", type=Path, help="Optional existing test feature .npy; requires matching test_ids.json beside it")
    args = parser.parse_args()
    started = time.monotonic()

    paths = list((args.dataset / "test").glob("*.png"))
    if any(not p.stem.isdigit() for p in paths):
        raise ValueError("Every test PNG filename must be a numeric competition ID")
    paths.sort(key=lambda p: int(p.stem))
    ids = [int(p.stem) for p in paths]
    if ids != list(range(10001, 19880)):
        raise ValueError("Expected each test ID 10001–19879 exactly once")

    saved = joblib.load(args.model)
    if saved.get("classes") != CLASSES:
        raise ValueError("Saved model provider order does not match the competition class order")
    model = saved["model"]
    if not np.array_equal(model.classes_, np.arange(len(CLASSES))):
        raise ValueError("Classifier probability columns do not match the saved provider order")
    if hasattr(model, "n_jobs"):
        model.n_jobs = 2
    first, schema = feature_vector(paths[0])
    if saved.get("feature_schema") != schema:
        raise ValueError("Feature extractor schema differs from the saved training schema")
    columns = np.asarray(saved["feature_indices"])
    if (columns.ndim != 1 or not np.issubdtype(columns.dtype, np.integer)
            or len(set(columns.tolist())) != len(columns)
            or (columns < 0).any() or (columns >= len(first)).any()
            or model.n_features_in_ != len(columns)):
        raise ValueError("Saved feature column selection is invalid")

    cache = None
    if args.features:
        sidecar = args.features.parent / "test_ids.json"
        if not sidecar.exists() or [int(i) for i in json.loads(sidecar.read_text())] != ids:
            raise ValueError("Feature cache requires test_ids.json with exactly matching ordered IDs")
        cache = np.load(args.features, mmap_mode="r", allow_pickle=False)
        if cache.shape != (len(ids), len(first)) or not np.issubdtype(cache.dtype, np.floating):
            raise ValueError("Feature cache shape or dtype is invalid")
        if not np.isfinite(cache).all():
            raise ValueError("Feature cache contains a nonfinite value")
        # Verify a bounded sample against current images as a cache mismatch check.
        for i in np.linspace(0, len(ids) - 1, 12, dtype=int):
            rendered, _ = feature_vector(paths[i])
            np.testing.assert_allclose(cache[i], rendered, rtol=0, atol=1e-7,
                                       err_msg=f"Cached features differ from test image {ids[i]}")

    probabilities = np.empty((len(ids), len(CLASSES)), dtype=np.float32)
    with threadpool_limits(limits=2):
        for start in range(0, len(paths), 512):
            stop = min(start + 512, len(paths))
            features = (cache[start:stop] if cache is not None else
                        np.stack([feature_vector(p)[0] for p in paths[start:stop]]))
            probabilities[start:stop] = model.predict_proba(features[:, columns])
            print(json.dumps({"predicted": stop, "total": len(ids)}), flush=True)
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(1), 1, rtol=0, atol=1e-6)):
        raise ValueError("Classifier produced invalid probabilities")

    args.out.mkdir(parents=True, exist_ok=True)
    probability_path = args.out / "test_probabilities.npy"
    csv_path = args.out / "submission_style.csv"
    np.save(probability_path, probabilities)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Label"])
        writer.writerows((i, CLASSES[p]) for i, p in zip(ids, probabilities.argmax(1)))
    (args.out / "test_ids.json").write_text(json.dumps([str(i) for i in ids]) + "\n")
    (args.out / "classes.json").write_text(json.dumps(CLASSES) + "\n")
    receipt = {
        "rows": len(ids), "classes": CLASSES, "model_sha256": sha256(args.model),
        "feature_source_sha256": sha256(Path(__file__).with_name("run_style_experiment.py")),
        "inference_source_sha256": sha256(__file__), "csv_sha256": sha256(csv_path),
        "probabilities_sha256": sha256(probability_path), "features_from_cache": cache is not None,
        "cached_feature_image_checks": 12 if cache is not None else None,
        "source_training_split_sha256": saved.get("split_sha256"),
        "elapsed_seconds": time.monotonic() - started, "threads": 2,
        "training_performed": False, "kaggle_uploaded": False,
    }
    (args.out / "inference_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
