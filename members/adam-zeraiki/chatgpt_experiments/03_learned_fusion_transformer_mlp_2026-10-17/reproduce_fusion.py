#!/usr/bin/env python3
"""Reproduce a saved emoji fusion submission directly from the supplied images.

This program performs inference only. It does not train, download weights,
contact Kaggle, or require the large intermediate prefix caches used in the
development experiments.

Expected portable bundle layout beside this file:
  heads.py, extract_features.py, selected_candidate.json
  base_models/dino_original/{train_meta.py,dinov2_vits14_pretrain.pth,
                             run_224/split.csv,run_224/selection/best.pt}
  base_models/convnext/{train_convnext.py,convnext_tiny-983f1562.pth,
                        run_224/selection/best.pt}
  base_models/style/{run_style_experiment.py,results/feature_schema.json}
  runs/{frozen_style,tuned_style}/{configuration.json,selection_summary.json,
       {mlp,transformer}/seed_{17,43,101}/refit/model.pt}

The data directory must contain the original train_labels.csv and test/*.png.
Training images are not read. Full mode requires exactly the 9,879 test IDs
10001 through 19879. A small replay can use --sample-ids; its CSV is deliberately
named sample_predictions.csv and is not a valid full competition submission.

Examples, after installing the bundle's requirements.txt:
  python reproduce_fusion.py --data ./dataset --out ./inference
  python reproduce_fusion.py --data ./dataset --out ./transformer \
      --candidate tuned_style_transformer
  python reproduce_fusion.py --data ./dataset --out ./sample \
      --candidate frozen_style_mlp --sample-ids 10001 10250 19879
  python reproduce_fusion.py --data ./dataset --out ./baseline --candidate baseline

Pinned development environment: Python 3.12, torch 2.8.0+cpu,
torchvision 0.23.0+cpu, timm 1.0.20, numpy 2.3.5, pandas 2.2.3,
Pillow 12.3.0, scikit-learn 1.8.0, scipy 1.17.0, joblib 1.5.3,
threadpoolctl 3.6.0. Use requirements.txt for installation. The receipt records
the actual installed versions. CPU bfloat16 encoder inference and the original
float16 prefix roundtrip reproduce the development preprocessing and arithmetic.
Exact floating-point replay can depend on the CPU and numerical-library build.

The four fusion candidates use the same three fixed seed checkpoints. The
selected_candidate.json default needs a top-level candidate string. Checkpoint
hashes, epoch counts and configurations come from the corresponding saved run;
saved absolute development paths are never used to locate bundle files.

AI assistance: OpenAI Codex wrote this reproducibility wrapper at the user's
request to implement and test a TRIBE-inspired fusion architecture. This code
implements that image-classification adaptation, not the TRIBE v2 brain model.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


CLASSES = ["apple", "facebook", "google", "messenger", "mozilla", "samsung", "whatsapp"]
SEEDS = [17, 43, 101]
EXPECTED_IDS = [str(i) for i in range(10001, 19880)]
DIMENSIONS = [768, 768, 1460]
BRANCH_NAMES = ["dino", "convnext", "style"]
LABELS_SHA256 = "37385b8fa192a03230d170654d255351a1cc3f42dfcd94b55f98f03abcfd55fb"
BASELINE = "baseline_dinov2_convnext_equal"
FUSION_CANDIDATES = {
    "frozen_style_mlp": ("frozen_style", "mlp", "pretrained"),
    "frozen_style_transformer": ("frozen_style", "transformer", "pretrained"),
    "tuned_style_mlp": ("tuned_style", "mlp", "tuned"),
    "tuned_style_transformer": ("tuned_style", "transformer", "tuned"),
}
SOURCE_HASHES = {
    "base_models/dino_original/train_meta.py": "1cfd165616fd1ff3dff4d246ba655a0485b0ca18775e648430ab7242fc14c810",
    "base_models/dino_original/dinov2_vits14_pretrain.pth": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
    "base_models/dino_original/run_224/split.csv": "32461c8bab69a90b659e5d8903cd8d43f8a5bf2262062ba02776333537889e5e",
    "base_models/dino_original/run_224/selection/best.pt": "731f53ccaf235be2747bb64b13c0846ed07e01f28633e72ce97c1a6074a211fc",
    "base_models/convnext/train_convnext.py": "6ebf2e328cb8d293e71d63d0499256dcbb974bd13c5fee09debee320b481b71f",
    "base_models/convnext/convnext_tiny-983f1562.pth": "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d",
    "base_models/convnext/run_224/selection/best.pt": "f8c081ca533695a702412952dc2f9e4ca450814ed993e3f60adffeadbeb8a9cf",
    "base_models/style/run_style_experiment.py": "f6824b4d1e6efa68f4b4cb455c3d4ba75e7a377c4d360c4df84ef2c62a161523",
    "base_models/style/results/feature_schema.json": "81d2d8937a978264ea10ee6b6ba6e0eb9a056da81a7a43583f10976bbbf504b0",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    require(specification is not None and specification.loader is not None, f"Cannot import {path}")
    result = importlib.util.module_from_spec(specification)
    sys.modules[name] = result
    specification.loader.exec_module(result)
    return result


def emit(event: str, **values: Any) -> None:
    print(json.dumps({"event": event, **values}), flush=True)


def resolve_candidate(requested: str | None, bundle: Path) -> tuple[str, dict[str, Any]]:
    selection: dict[str, Any] = {}
    if requested is None:
        path = bundle / "selected_candidate.json"
        require(path.is_file(), "selected_candidate.json is missing; specify --candidate explicitly")
        selection = json.loads(path.read_text())
        requested = selection.get("candidate")
        require(isinstance(requested, str), "selected_candidate.json must contain a candidate string")
    if requested in {"baseline", "baseline_equal", "dinov2_convnext_equal"}:
        requested = BASELINE
    require(requested == BASELINE or requested in FUSION_CANDIDATES, f"Unsupported candidate: {requested}")
    return requested, selection


def verify_sources(bundle: Path, include_style: bool) -> dict[str, str]:
    checked = {}
    for relative, expected in SOURCE_HASHES.items():
        if not include_style and relative.startswith("base_models/style/"):
            continue
        path = bundle / relative
        require(path.is_file(), f"Required bundle file is missing: {relative}")
        actual = sha256(path)
        require(actual == expected, f"Bundle source hash mismatch: {relative}")
        checked[relative] = actual
    for relative in ("heads.py", "extract_features.py"):
        require((bundle / relative).is_file(), f"Required bundle file is missing: {relative}")
        checked[relative] = sha256(bundle / relative)
    return checked


def choose_ids(data: Path, requested: list[str] | None) -> tuple[list[str], bool]:
    require((data / "test").is_dir(), "--data must contain a test directory")
    require((data / "train_labels.csv").is_file(), "--data must contain the original train_labels.csv")
    require(sha256(data / "train_labels.csv") == LABELS_SHA256,
            "train_labels.csv differs from the original competition file used by this bundle")
    if requested is None:
        paths = list((data / "test").glob("*.png"))
        ids = sorted(path.stem for path in paths)
        require(ids == EXPECTED_IDS, "Full mode requires exactly test IDs 10001 through 19879, each once")
        return ids, False
    ids = [part.strip() for value in requested for part in value.split(",") if part.strip()]
    require(bool(ids) and len(ids) == len(set(ids)), "--sample-ids must be a nonempty list of unique IDs")
    expected = set(EXPECTED_IDS)
    require(all(identifier in expected for identifier in ids), "A sample ID is outside the competition test range")
    require(all((data / "test" / f"{identifier}.png").is_file() for identifier in ids), "A requested test image is missing")
    return ids, True


def protect_input_cache(data: Path, output: Path, ids: list[str], sample: bool) -> str:
    """Prevent cache reuse after changing image contents, IDs, order or mode."""
    record = {
        "format_version": 1, "mode": "sample" if sample else "full",
        "ids": ids, "labels_sha256": LABELS_SHA256,
        "test_image_sha256": [sha256(data / "test" / f"{identifier}.png") for identifier in ids],
    }
    path = output / "input_manifest.json"
    if path.exists():
        require(json.loads(path.read_text()) == record,
                "Output directory contains features for different images or IDs; use a different --out")
    else:
        write_json(path, record)
    return sha256(path)


def verify_head_source(bundle: Path, executed_hash: str, current_hash: str) -> dict[str, Any]:
    """Accept the executed source or its one recorded module-docstring correction."""
    if executed_hash == current_hash:
        return {"status": "exact_executed_source", "source_sha256": current_hash}
    provenance = bundle / "runs/provenance"
    patch_path = provenance / "documentation_patch.json"
    executed_path = provenance / "heads_executed.py"
    require(patch_path.is_file() and executed_path.is_file(),
            "heads.py differs from the fitted run and no recorded documentation-only correction exists")
    patch = json.loads(patch_path.read_text())
    require(patch.get("executed_source") == "heads_executed.py", "Unexpected executed-source provenance filename")
    require(patch.get("executed_source_sha256") == executed_hash == sha256(executed_path),
            "Recorded executed heads source does not match the fitted run")
    require(patch.get("current_source_sha256") == current_hash == sha256(bundle / "heads.py"),
            "Current heads source does not match the recorded documentation correction")
    original = executed_path.read_text()
    current = (bundle / "heads.py").read_text()
    old_phrase, new_phrase = "No competition labels", "No hidden test labels"
    require(original.count(old_phrase) == 1 and original.replace(old_phrase, new_phrase, 1) == current,
            "heads.py contains changes beyond the single recorded documentation correction")

    def executable_ast(source: str) -> str:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                body = node.body
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                    del body[0]
        return ast.dump(tree, include_attributes=False)

    old_ast, new_ast = executable_ast(original), executable_ast(current)
    require(old_ast == new_ast, "The documentation correction changed executable Python syntax")
    ast_hash = hashlib.sha256(old_ast.encode()).hexdigest()
    require(patch.get("executable_ast_identical") is True and patch.get("executable_ast_sha256") == ast_hash,
            "Executable AST proof differs from the documentation correction record")
    return {
        "status": "verified_recorded_documentation_only_correction",
        "executed_source_sha256": executed_hash, "current_source_sha256": current_hash,
        "documentation_patch_sha256": sha256(patch_path), "executable_ast_sha256": ast_hash,
        "exact_single_phrase_replacement_verified": True,
    }


def locked_heads(bundle: Path, candidate: str, source_hashes: dict[str, str]) -> dict[str, Any] | None:
    if candidate == BASELINE:
        return None
    condition, family, representation = FUSION_CANDIDATES[candidate]
    run = bundle / "runs" / condition
    summary = json.loads((run / "selection_summary.json").read_text())
    configuration = json.loads((run / "configuration.json").read_text())
    require(summary.get("refit_completed") is True, f"Refits are not complete for {condition}")
    source_verification = verify_head_source(bundle, configuration["script_sha256"], source_hashes["heads.py"])
    require(configuration["class_order"] == CLASSES, "Run class order differs from the competition")
    require(configuration["branch_names"] == BRANCH_NAMES, "Run feature branch order differs")
    require(configuration["configuration"]["seeds"] == SEEDS, "Run does not contain the three fixed experiment seeds")
    require(summary["configuration_sha256"] == sha256(run / "configuration.json"), "Saved run configuration hash mismatch")
    records = summary["families"][family]["refit_per_seed"]
    require([entry["seed"] for entry in records] == SEEDS, "Refit seed order differs from the locked experiment")
    checkpoints = []
    for entry in records:
        seed = entry["seed"]
        path = run / family / f"seed_{seed}" / "refit" / "model.pt"
        require(entry["training_rows"] == 7903, "Unexpected refit training partition")
        require(path.is_file(), f"Refit checkpoint is missing: {path.relative_to(bundle)}")
        actual = sha256(path)
        require(actual == entry["checkpoint_sha256"], f"Refit checkpoint hash mismatch for seed {seed}")
        checkpoints.append({
            "path": path, "bundle_path": str(path.relative_to(bundle)), "sha256": actual,
            "seed": seed, "epochs": entry["selected_epochs"],
        })
    return {
        "condition": condition, "family": family, "representation": representation,
        "configuration": configuration["configuration"], "checkpoints": checkpoints,
        "configuration_sha256": sha256(run / "configuration.json"),
        "selection_summary_sha256": sha256(run / "selection_summary.json"),
        "head_source_verification": source_verification,
    }


def complete_arrays(directory: Path, dimensions: dict[str, int], count: int) -> dict[str, Any]:
    done = np.load(directory / "test_complete.npy", allow_pickle=False)
    require(done.shape == (count,) and done.dtype == np.bool_ and bool(done.all()),
            f"Feature extraction is incomplete: {directory}")
    result = {}
    for name, width in dimensions.items():
        path = directory / f"test_{name}.npy"
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        require(values.shape == (count, width) and values.dtype == np.float32, f"Wrong feature shape/dtype: {path}")
        require(bool(np.isfinite(values).all()), f"Nonfinite features: {path}")
        result[name] = {"shape": list(values.shape), "dtype": str(values.dtype), "sha256": sha256(path)}
    return result


def direct_convnext(extractor: Any, arguments: Any, ids: list[str]) -> dict[str, Any]:
    """Compute prefix in memory and retain only pooled features/probabilities."""
    root = arguments.package / "convnext"
    module = load_module(root / "train_convnext.py", "portable_original_convnext")
    pretrained = root / "convnext_tiny-983f1562.pth"
    checkpoint = root / "run_224/selection/best.pt"
    signature = extractor.signature(arguments, "test", ids, {
        "model": "ConvNeXt-Tiny", "pretrained_sha256": sha256(pretrained),
        "fine_tuned_checkpoint_sha256": sha256(checkpoint),
        "feature": "Normalized pooled final-stage 768 features",
        "prefix_source": "Fresh images; in-memory prefix; float16 storage roundtrip; no prefix cache",
    })
    directory = arguments.out / "convnext"
    dimensions = {"pretrained": 768, "tuned": 768, "tuned_probabilities": 7}
    arrays, done = extractor.init_arrays(directory, "test", dimensions, len(ids), signature)
    if not bool(done.all()):
        backbone = module.load_backbone(SimpleNamespace(checkpoint=pretrained))
        original = module.TailClassifier(backbone).eval().requires_grad_(False)
        tuned = module.TailClassifier(backbone).eval().requires_grad_(False)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        require(saved["class_names"] == CLASSES, "ConvNeXt checkpoint class order mismatch")
        tuned.load_state_dict(saved["model_state_dict"], strict=True)
        del saved
        pending = np.flatnonzero(~done)
        loader = DataLoader(
            module.ImageDataset(arguments.data, "test", ids, resolution=224, indices=pending),
            batch_size=arguments.batch_size, shuffle=False, num_workers=0,
        )
        started = last = time.monotonic()
        with torch.inference_mode():
            for images, indices, _ in loader:
                with extractor.autocast():
                    prefix = module.prefix_forward(backbone, images)
                prefix = prefix.float().to(torch.float16).float()
                with extractor.autocast():
                    original_features = original.features(prefix)
                    tuned_features = tuned.features(prefix)
                    logits = tuned.head(tuned_features)
                values = {
                    "pretrained": original_features.float().numpy(),
                    "tuned": tuned_features.float().numpy(),
                    "tuned_probabilities": logits.float().softmax(1).numpy(),
                }
                rows = indices.numpy()
                for name, value in values.items():
                    require(bool(np.isfinite(value).all()), f"ConvNeXt produced nonfinite {name}")
                    arrays[name][rows] = value
                for array in arrays.values():
                    array.flush()
                done[rows] = True
                done.flush()
                now = time.monotonic()
                if now - last >= 20 or bool(done.all()):
                    emit("portable_convnext_features", done=int(done.sum()), total=len(ids), seconds=round(now - started, 1))
                    last = now
        del original, tuned, backbone
    result = complete_arrays(directory, dimensions, len(ids))
    write_json(directory / "test_receipt.json", {
        "complete": True, "rows": len(ids), "features": result,
        "prefix_cache_created": False, "training_performed": False,
    })
    return result


def direct_style(extractor: Any, arguments: Any, ids: list[str]) -> dict[str, Any]:
    root = arguments.package / "style"
    module = load_module(root / "run_style_experiment.py", "portable_style_features")
    schema_path = root / "results/feature_schema.json"
    expected_schema = json.loads(schema_path.read_text())
    require(module.CLASSES == CLASSES, "Style extractor class order mismatch")
    signature = extractor.signature(arguments, "test", ids, {
        "model": "Fixed 1460-dimensional pixel-style descriptor",
        "extractor_sha256": sha256(root / "run_style_experiment.py"),
        "feature_schema_sha256": sha256(schema_path),
        "compute": "Original NumPy float32 pixel-style extractor",
        "resolution": 64, "prefix_roundtrip": None,
        "preprocessing": "Original RGBA style rendering; identical schema to training",
    })
    directory = arguments.out / "style"
    arrays, done = extractor.init_arrays(directory, "test", {"features": 1460}, len(ids), signature)
    pending = np.flatnonzero(~done)
    started = last = time.monotonic()
    for start in range(0, len(pending), 128):
        rows = pending[start:start + 128]
        for index in rows:
            vector, schema = module.feature_vector(arguments.data / "test" / f"{ids[int(index)]}.png")
            require(schema == expected_schema, "Style feature schema differs from training")
            require(vector.shape == (1460,) and bool(np.isfinite(vector).all()), "Invalid style descriptor")
            arrays["features"][index] = vector
        arrays["features"].flush()
        done[rows] = True
        done.flush()
        now = time.monotonic()
        if now - last >= 20 or bool(done.all()):
            emit("portable_style_features", done=int(done.sum()), total=len(ids), seconds=round(now - started, 1))
            last = now
    result = complete_arrays(directory, {"features": 1460}, len(ids))
    write_json(directory / "test_receipt.json", {"complete": True, "rows": len(ids), "features": result, "training_performed": False})
    return result


def predict_fusion(heads: Any, specification: dict[str, Any], features_root: Path,
                   ids: list[str], output: Path, batch_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    representation = specification["representation"]
    paths = {
        "dino": features_root / "dino" / f"test_{representation}.npy",
        "convnext": features_root / "convnext" / f"test_{representation}.npy",
        "style": features_root / "style" / "test_features.npy",
    }
    features, feature_records = heads.load_features(paths, BRANCH_NAMES, len(ids), DIMENSIONS)
    indices = np.arange(len(ids), dtype=np.int64)
    arrays, records = [], []
    for entry in specification["checkpoints"]:
        model, metadata = heads.load_checkpoint(entry["path"])
        require(metadata["branch_names"] == BRANCH_NAMES and metadata["class_order"] == CLASSES, "Head metadata order mismatch")
        require(metadata["dimensions"] == DIMENSIONS, "Head feature dimensions differ")
        require(metadata["family"] == specification["family"] and metadata["seed"] == entry["seed"], "Head family or seed mismatch")
        require(metadata["stage"] == "original_training_refit" and metadata["epoch"] == entry["epochs"], "Unexpected head training stage or epoch")
        require(metadata["configuration"] == specification["configuration"], "Head configuration differs from its locked run")
        probabilities = heads.predict_probabilities(model, features, indices, batch_size)
        path = output / "seed_probabilities" / f"seed_{entry['seed']}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, probabilities)
        arrays.append(probabilities)
        records.append({
            "seed": entry["seed"], "epochs": entry["epochs"],
            "checkpoint": entry["bundle_path"], "checkpoint_sha256": entry["sha256"],
            "probabilities_sha256": sha256(path),
        })
        del model
    average = heads.mean_probabilities(arrays)
    return average, {
        "condition": specification["condition"], "family": specification["family"],
        "rule": "Equal fixed-seed mean, accumulated in float64 then stored as float32",
        "seeds": records, "equal_seed_weights": [1 / len(records)] * len(records),
        "configuration_sha256": specification["configuration_sha256"],
        "selection_summary_sha256": specification["selection_summary_sha256"],
        "head_source_verification": specification["head_source_verification"],
        "feature_files": feature_records,
    }


def versions() -> dict[str, str]:
    result = {"python": platform.python_version(), "torch": str(torch.__version__)}
    for distribution in ("torchvision", "timm", "numpy", "pandas", "Pillow", "scikit-learn", "scipy", "joblib", "threadpoolctl"):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = "not installed"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="Original dataset root containing train_labels.csv and test/*.png")
    parser.add_argument("--out", type=Path, required=True, help="Output directory; use a different directory for different sample IDs")
    parser.add_argument("--candidate", choices=[*FUSION_CANDIDATES, BASELINE, "baseline", "baseline_equal", "dinov2_convnext_equal"])
    parser.add_argument("--sample-ids", nargs="+", help="Small explicit test-ID list; spaces or commas are accepted")
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent, help="Defaults to the directory containing this script")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16, help="Encoder batch size used in the original feature extraction")
    parser.add_argument("--head-batch-size", type=int, default=256)
    args = parser.parse_args()
    require(args.threads >= 1 and args.batch_size >= 1 and args.head_batch_size >= 1, "Thread and batch sizes must be positive")
    started = time.monotonic()
    bundle, data, output = args.bundle.resolve(), args.data.resolve(), args.out.resolve()
    candidate, default_selection = resolve_candidate(args.candidate, bundle)
    source_hashes = verify_sources(bundle, include_style=candidate != BASELINE)
    specification = locked_heads(bundle, candidate, source_hashes)
    ids, sample = choose_ids(data, args.sample_ids)
    output.mkdir(parents=True, exist_ok=True)
    input_hash = protect_input_cache(data, output, ids, sample)
    heads = load_module(bundle / "heads.py", "portable_fusion_heads")
    extractor = load_module(bundle / "extract_features.py", "portable_feature_extractor")
    heads.configure_cpu(args.threads)
    torch.manual_seed(42)
    feature_args = SimpleNamespace(package=bundle / "base_models", data=data, out=output / "features", batch_size=args.batch_size)
    emit("portable_inference_started", candidate=candidate, rows=len(ids), sample_mode=sample)
    extractor.run_dino(feature_args, ids, "test")
    dino_record = complete_arrays(feature_args.out / "dino", {"pretrained": 768, "tuned": 768, "tuned_probabilities": 7}, len(ids))
    conv_record = direct_convnext(extractor, feature_args, ids)
    feature_records: dict[str, Any] = {"dino": dino_record, "convnext": conv_record}
    if specification is None:
        dino = np.load(feature_args.out / "dino/test_tuned_probabilities.npy", allow_pickle=False)
        convnext = np.load(feature_args.out / "convnext/test_tuned_probabilities.npy", allow_pickle=False)
        heads.validate_probabilities(dino, len(ids), 7)
        heads.validate_probabilities(convnext, len(ids), 7)
        # This exact float32 arithmetic was used by the previous selected ensemble.
        probabilities = np.float32(0.5) * dino + np.float32(0.5) * convnext
        predictor_record = {"rule": "Float32 50:50 DINOv2/ConvNeXt probability average", "weights": [0.5, 0.5]}
    else:
        feature_records["style"] = direct_style(extractor, feature_args, ids)
        probabilities, predictor_record = predict_fusion(heads, specification, feature_args.out, ids, output, args.head_batch_size)
    heads.validate_probabilities(probabilities, len(ids), 7)
    require(probabilities.dtype == np.float32, "Final probabilities must retain the original float32 storage")
    stem = "sample" if sample else "test"
    probability_path = output / f"{stem}_probabilities.npy"
    csv_path = output / ("sample_predictions.csv" if sample else "submission.csv")
    np.save(probability_path, probabilities)
    heads.write_submission(csv_path, ids, probabilities, CLASSES)
    write_json(output / f"{stem}_ids.json", ids)
    write_json(output / "class_order.json", CLASSES)
    # Validate the serialized upload schema independently of the writer.
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ["Id", "Label"], "Serialized CSV header differs from Id,Label")
        rows = list(reader)
    require([row["Id"] for row in rows] == ids, "Serialized CSV ID order differs")
    require([row["Label"] for row in rows] == [CLASSES[i] for i in probabilities.argmax(axis=1)], "Serialized CSV labels differ from saved probabilities")
    receipt = {
        "candidate": candidate, "rows": len(ids), "mode": "sample" if sample else "full",
        "complete_kaggle_submission": not sample, "class_order": CLASSES,
        "input_manifest_sha256": input_hash, "csv": csv_path.name, "csv_sha256": sha256(csv_path),
        "probabilities": probability_path.name, "probabilities_sha256": sha256(probability_path),
        "source_files": source_hashes, "wrapper_sha256": sha256(Path(__file__)),
        "features": feature_records, "predictor": predictor_record,
        "default_selection_used": bool(default_selection),
        "selected_candidate_file_sha256": sha256(bundle / "selected_candidate.json") if default_selection else None,
        "environment": versions(), "threads": args.threads,
        "encoder_batch_size": args.batch_size, "head_batch_size": args.head_batch_size,
        "elapsed_seconds": time.monotonic() - started,
        "training_performed": False, "large_prefix_cache_required": False,
        "network_access_performed": False, "kaggle_uploaded": False,
    }
    write_json(output / "inference_receipt.json", receipt)
    emit("portable_inference_complete", candidate=candidate, rows=len(ids), sample_mode=sample,
         csv=str(csv_path), csv_sha256=receipt["csv_sha256"], seconds=round(receipt["elapsed_seconds"], 1))


if __name__ == "__main__":
    main()
