#!/usr/bin/env python3
"""Train and replay small three-expert image-feature fusion heads on CPU.

AI assistance: this implementation was written with ChatGPT at the student's
request: "ok perfect, then do it. Also test it with a jev like model and we'll
see if it gives interesting results". It implements a TRIBE-inspired expert
fusion experiment, not the TRIBE v2 neuroscience model. No competition labels
or leaderboard scores enter training.

The input manifest is JSON. Required keys:
  train_paths: {branch_name: path_to_N_by_D_float_npy, ...} (exactly 3 branches)
  ids, labels, groups: arrays in labelled-row order, or paths to JSON/NPY arrays
  train_indices, validation_indices: disjoint integer row-index arrays or paths
  class_order: list of class labels (label arrays may use names or class indices)
Optional keys:
  test_paths: the same branch names mapped to unlabelled feature arrays
  test_ids: IDs in the test feature row order
  inner_train_indices, inner_validation_indices: fixed grouped inner split;
      these must partition train_indices and share no duplicate groups
  name, provenance: descriptive, copied to output records
Paths inside a manifest resolve relative to that manifest's directory.

Example:
  python heads.py fit --manifest features.json --output results --threads 2
  python heads.py predict --run results --manifest features.json --output replay

All supervised normalization and head selection use the inner training split.
Epoch counts and the family recommendation use ONLY inner validation. Each seed
then refits on the original training partition for its selected number of epochs.
The outer validation labels are deliberately never scored by this script.
It saves probabilities for all labelled rows, letting a separate locked-protocol
evaluator compare the models after all experiments have been fixed.

The default families share projection width64, standardization, input clipping,
optimizer, dropout and seed set. The MLP 192->160->classes has a similar fusion
parameter count to the one-layer Transformer (width64, 4heads, feedforward128).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch import nn


DEFAULTS = {
    "width": 64,
    "heads": 4,
    "feedforward_multiplier": 2,
    "transformer_layers": 1,
    "mlp_hidden": 160,
    "dropout": 0.15,
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "batch_size": 256,
    "max_epochs": 60,
    "patience": 8,
    "clip_input": 8.0,
    "std_floor_absolute": 1e-5,
    "std_floor_relative": 1e-3,
    "max_gradient_norm": 5.0,
    "inner_folds": 5,
    "inner_split_seed": 271828,
    "seeds": [17, 43, 101],
    "families": ["mlp", "transformer"],
    "scheduler": "cosine_fixed_max_epoch_horizon",
    "minimum_lr_fraction": 0.05,
    "label_smoothing": 0.0,
}


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_sha256(values: list[Any]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def read_array(value: Any, base: Path) -> list[Any]:
    if isinstance(value, str):
        path = resolve_path(value, base)
        if path.suffix.lower() == ".npy":
            result = np.load(path, allow_pickle=False).tolist()
        else:
            result = json.loads(path.read_text())
    else:
        result = value
    if not isinstance(result, list):
        raise ValueError("A manifest array must be a list or a path to a JSON/NPY list")
    return result


def configure_cpu(threads: int) -> None:
    if threads < 1:
        raise ValueError("threads must be >=1")
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_manifest(path: Path, *, require_split: bool = True) -> dict[str, Any]:
    source = json.loads(path.read_text())
    base = path.resolve().parent
    names = list(source["train_paths"])
    if len(names) != 3 or len(set(names)) != 3:
        raise ValueError("The predeclared experiment requires exactly three named branches")
    class_order = [str(x) for x in source["class_order"]]
    if len(class_order) < 2 or len(set(class_order)) != len(class_order):
        raise ValueError("class_order must contain unique class names")
    ids = [str(x) for x in read_array(source["ids"], base)]
    if len(ids) != len(set(ids)):
        raise ValueError("Labelled IDs must be unique")
    label_values = read_array(source["labels"], base)
    lookup = {name: index for index, name in enumerate(class_order)}
    labels = np.asarray([
        lookup[str(x)] if str(x) in lookup else int(x) for x in label_values
    ], dtype=np.int64)
    if labels.shape != (len(ids),) or (labels < 0).any() or (labels >= len(class_order)).any():
        raise ValueError("Invalid label array or class mapping")
    group_field = source.get("groups", source.get("group_ids"))
    if group_field is None:
        raise ValueError("groups or group_ids is required to protect duplicate groups")
    groups = np.asarray([str(x) for x in read_array(group_field, base)])
    if groups.shape != (len(ids),):
        raise ValueError("Group count must match labelled row count")
    result: dict[str, Any] = {
        "source": source,
        "path": path.resolve(),
        "names": names,
        "class_order": class_order,
        "ids": ids,
        "labels": labels,
        "groups": groups,
        "train_paths": {name: resolve_path(source["train_paths"][name], base) for name in names},
        "test_paths": {},
        "test_ids": [],
    }
    if source.get("test_paths"):
        if set(source["test_paths"]) != set(names):
            raise ValueError("Train/test branch names must match exactly")
        result["test_paths"] = {
            name: resolve_path(source["test_paths"][name], base) for name in names
        }
        result["test_ids"] = [str(x) for x in read_array(source["test_ids"], base)]
        if len(result["test_ids"]) != len(set(result["test_ids"])):
            raise ValueError("Test IDs must be unique")
    for key in ("train_indices", "validation_indices", "inner_train_indices", "inner_validation_indices"):
        if key in source:
            raw_values = read_array(source[key], base)
            if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in raw_values):
                raise ValueError(f"{key} must contain integer row indices")
            array = np.asarray(raw_values, dtype=np.int64)
            if array.ndim != 1 or len(array) != len(set(array.tolist())):
                raise ValueError(f"{key} must be a unique, one-dimensional list")
            if len(array) == 0 or array.min() < 0 or array.max() >= len(ids):
                raise ValueError(f"{key} contains an out-of-range index or is empty")
            result[key] = array
    if require_split:
        train = result["train_indices"]
        outer = result["validation_indices"]
        verify_partition(train, outer, np.arange(len(ids)), groups, "outer")
    return result


def verify_partition(first: np.ndarray, second: np.ndarray, parent: np.ndarray,
                     groups: np.ndarray, name: str) -> None:
    if set(first.tolist()) & set(second.tolist()):
        raise ValueError(f"{name}: row overlap")
    if set(first.tolist()) | set(second.tolist()) != set(parent.tolist()):
        raise ValueError(f"{name}: indices must partition the parent rows")
    if set(groups[first].tolist()) & set(groups[second].tolist()):
        raise ValueError(f"{name}: exact-duplicate group overlap")


def get_inner_split(manifest: dict[str, Any], config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    train = manifest["train_indices"]
    supplied = ("inner_train_indices" in manifest, "inner_validation_indices" in manifest)
    if any(supplied) and not all(supplied):
        raise ValueError("Both inner split index arrays must be supplied together")
    if all(supplied):
        first = manifest["inner_train_indices"]
        second = manifest["inner_validation_indices"]
    else:
        splitter = StratifiedGroupKFold(
            n_splits=config["inner_folds"], shuffle=True,
            random_state=config["inner_split_seed"],
        )
        first_local, second_local = next(splitter.split(
            np.zeros(len(train)), manifest["labels"][train], manifest["groups"][train]
        ))
        first, second = train[first_local], train[second_local]
    verify_partition(first, second, train, manifest["groups"], "inner")
    n_classes = len(manifest["class_order"])
    if len(np.unique(manifest["labels"][first])) != n_classes:
        raise ValueError("Inner training must contain every class")
    if len(np.unique(manifest["labels"][second])) != n_classes:
        raise ValueError("Inner validation must contain every class")
    return first, second


def load_features(paths: dict[str, Path], names: list[str], n_rows: int,
                  expected_dimensions: list[int] | None = None) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    features: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    for number, name in enumerate(names):
        path = paths[name]
        source = np.load(path, mmap_mode="r", allow_pickle=False)
        if source.ndim != 2 or source.shape[0] != n_rows or source.shape[1] < 1:
            raise ValueError(f"{name}: expected {n_rows} rows and a positive feature dimension, got {source.shape}")
        if not np.issubdtype(source.dtype, np.floating):
            raise ValueError(f"{name}: features must be floating point")
        if expected_dimensions is not None and source.shape[1] != expected_dimensions[number]:
            raise ValueError(f"{name}: feature dimension differs from fitted checkpoint")
        for start in range(0, n_rows, 1024):
            if not np.isfinite(source[start:start + 1024]).all():
                raise ValueError(f"{name}: non-finite feature value")
        features.append(torch.from_numpy(np.array(source, dtype=np.float32, copy=True)))
        records.append({
            "branch": name, "path": str(path), "shape": list(source.shape),
            "source_dtype": str(source.dtype), "sha256": sha256(path),
        })
        del source
    return features, records


class FusionHead(nn.Module):
    def __init__(self, dimensions: list[int], classes: int, family: str,
                 config: dict[str, Any]):
        super().__init__()
        if family not in {"mlp", "transformer"}:
            raise ValueError(f"Unknown fusion family {family}")
        self.family = family
        self.dimensions = list(dimensions)
        self.config = dict(config)
        width = config["width"]
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(dimension, width), nn.LayerNorm(width), nn.GELU())
            for dimension in dimensions
        ])
        for number, dimension in enumerate(dimensions):
            self.register_buffer(f"mean_{number}", torch.zeros(dimension))
            self.register_buffer(f"std_{number}", torch.ones(dimension))
        self.expert_identity = nn.Parameter(torch.empty(len(dimensions), width))
        nn.init.normal_(self.expert_identity, std=0.02)
        self.input_dropout = nn.Dropout(config["dropout"])
        if family == "transformer":
            self.cls_token = nn.Parameter(torch.empty(1, 1, width))
            nn.init.normal_(self.cls_token, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=width,
                nhead=config["heads"],
                dim_feedforward=config["feedforward_multiplier"] * width,
                dropout=config["dropout"],
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.fusion = nn.TransformerEncoder(
                layer, num_layers=config["transformer_layers"], enable_nested_tensor=False,
            )
            self.output_norm = nn.LayerNorm(width)
            self.classifier = nn.Linear(width, classes)
        else:
            self.fusion = nn.Sequential(
                nn.Linear(len(dimensions) * width, config["mlp_hidden"]),
                nn.LayerNorm(config["mlp_hidden"]),
                nn.GELU(),
                nn.Dropout(config["dropout"]),
            )
            self.classifier = nn.Linear(config["mlp_hidden"], classes)

    def fit_standardization(self, features: list[torch.Tensor], indices: np.ndarray) -> list[dict[str, Any]]:
        records = []
        tensor_indices = torch.as_tensor(indices, dtype=torch.long)
        with torch.no_grad():
            for number, feature in enumerate(features):
                rows = feature.index_select(0, tensor_indices)
                mean = rows.mean(dim=0)
                std = rows.std(dim=0, unbiased=False)
                floor = max(
                    self.config["std_floor_absolute"],
                    float(torch.median(std)) * self.config["std_floor_relative"],
                )
                replaced = int((std < floor).sum())
                std = std.clamp_min(floor)
                getattr(self, f"mean_{number}").copy_(mean)
                getattr(self, f"std_{number}").copy_(std)
                records.append({
                    "branch_index": number, "fit_rows": len(indices),
                    "minimum_std": floor, "floored_dimensions": replaced,
                    "mean_min": float(mean.min()), "mean_max": float(mean.max()),
                    "std_min": float(std.min()), "std_max": float(std.max()),
                })
                del rows
        return records

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        projected = []
        for number, (values, projection) in enumerate(zip(features, self.projections)):
            standardized = (values - getattr(self, f"mean_{number}")) / getattr(self, f"std_{number}")
            standardized = standardized.clamp(-self.config["clip_input"], self.config["clip_input"])
            projected.append(projection(standardized))
        tokens = torch.stack(projected, dim=1) + self.expert_identity.unsqueeze(0)
        tokens = self.input_dropout(tokens)
        if self.family == "transformer":
            cls = self.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = self.fusion(torch.cat([cls, tokens], dim=1))
            representation = self.output_norm(tokens[:, 0])
        else:
            representation = self.fusion(tokens.flatten(start_dim=1))
        return self.classifier(representation)


def parameter_counts(model: FusionHead) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    projections = sum(parameter.numel() for parameter in model.projections.parameters() if parameter.requires_grad)
    return {"total_trainable": total, "shared_projection_design": projections, "fusion_and_classifier": total - projections}


def get_batches(features: list[torch.Tensor], indices: torch.Tensor) -> list[torch.Tensor]:
    return [feature.index_select(0, indices) for feature in features]


@torch.no_grad()
def predict_probabilities(model: FusionHead, features: list[torch.Tensor], indices: np.ndarray,
                          batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(indices), batch_size):
        batch_indices = torch.as_tensor(indices[start:start + batch_size], dtype=torch.long)
        logits = model(get_batches(features, batch_indices))
        outputs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    result = np.concatenate(outputs, axis=0).astype(np.float32, copy=False)
    validate_probabilities(result, len(indices), model.classifier.out_features)
    return result


def validate_probabilities(probabilities: np.ndarray, rows: int, classes: int) -> None:
    if probabilities.shape != (rows, classes) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid probability array shape or values")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Probabilities outside [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=2e-6, rtol=0):
        raise ValueError("Probability rows do not sum to one")


def classification_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    # This helper is called only on INTERNAL validation labels during fitting.
    prediction = probabilities.argmax(axis=1)
    correct = int(np.sum(prediction == labels))
    chosen = probabilities[np.arange(len(labels)), labels].astype(np.float64)
    return {
        "rows": len(labels), "correct": correct, "accuracy": correct / len(labels),
        "log_loss": float(-np.log(np.clip(chosen, 1e-12, 1)).mean()),
        "macro_f1": float(f1_score(labels, prediction, labels=np.arange(probabilities.shape[1]), average="macro", zero_division=0)),
    }


def gradient_check(model: FusionHead) -> dict[str, Any]:
    report = {}
    sections: dict[str, Any] = {
        f"projection_{number}": list(module.parameters())
        for number, module in enumerate(model.projections)
    }
    sections["fusion"] = list(model.fusion.parameters())
    sections["classifier"] = list(model.classifier.parameters())
    for name, parameters in sections.items():
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        finite = bool(gradients) and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        norm = math.sqrt(sum(float((gradient.detach().double() ** 2).sum()) for gradient in gradients))
        report[name] = {"finite": finite, "gradient_l2": norm}
        if not finite or norm <= 0:
            raise RuntimeError(f"Missing, zero or nonfinite training gradients in {name}")
    return report


def checkpoint_payload(model: FusionHead, manifest: dict[str, Any], config: dict[str, Any],
                       family: str, seed: int, epoch: int, stage: str) -> dict[str, Any]:
    return {
        "format_version": 1,
        "family": family,
        "seed": seed,
        "epoch": epoch,
        "stage": stage,
        "dimensions": model.dimensions,
        "branch_names": manifest["names"],
        "class_order": manifest["class_order"],
        "configuration": config,
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "provenance": manifest["source"].get("provenance", {}),
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def load_checkpoint(path: Path) -> tuple[FusionHead, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = FusionHead(payload["dimensions"], len(payload["class_order"]), payload["family"], payload["configuration"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def train_seed(manifest: dict[str, Any], features: list[torch.Tensor], config: dict[str, Any],
               family: str, seed: int, train_indices: np.ndarray,
               validation_indices: np.ndarray | None, epochs: int, output: Path,
               stage: str) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    dimensions = [feature.shape[1] for feature in features]
    model = FusionHead(dimensions, len(manifest["class_order"]), family, config)
    scaler_record = model.fit_standardization(features, train_indices)
    labels = torch.as_tensor(manifest["labels"], dtype=torch.long)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["max_epochs"],
        eta_min=config["learning_rate"] * config["minimum_lr_fraction"],
    )
    loss_function = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, Any]] = []
    best_key = (-1, -float("inf"))
    best_epoch = 0
    best_metrics = None
    best_probe = None
    no_improvement = 0
    checks = None
    checkpoint = output / ("best.pt" if validation_indices is not None else "model.pt")
    training_start = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_indices), generator=generator).numpy()
        order = train_indices[permutation]
        loss_sum = 0.0
        correct = 0
        epoch_start = time.monotonic()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        for start in range(0, len(order), config["batch_size"]):
            indices = torch.as_tensor(order[start:start + config["batch_size"]], dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(get_batches(features, indices))
            batch_labels = labels.index_select(0, indices)
            loss = loss_function(logits, batch_labels)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Training produced a nonfinite loss")
            loss.backward()
            if checks is None:
                checks = gradient_check(model)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_gradient_norm"], error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
            correct += int((logits.detach().argmax(dim=1) == batch_labels).sum())
        row = {
            "epoch": epoch, "train_loss": loss_sum / len(order),
            "train_accuracy": correct / len(order), "learning_rate": learning_rate,
        }
        if validation_indices is not None:
            probabilities = predict_probabilities(model, features, validation_indices, config["batch_size"])
            metrics = classification_metrics(probabilities, manifest["labels"][validation_indices])
            row.update({"inner_accuracy": metrics["accuracy"], "inner_log_loss": metrics["log_loss"], "inner_macro_f1": metrics["macro_f1"]})
            key = (metrics["correct"], -metrics["log_loss"])
            if key > best_key:
                best_key, best_epoch, best_metrics, no_improvement = key, epoch, metrics, 0
                payload = checkpoint_payload(model, manifest, config, family, seed, epoch, stage)
                payload["internal_validation_metrics"] = metrics
                save_checkpoint(checkpoint, payload)
                best_probe = predict_probabilities(
                    model, features, validation_indices[:min(16, len(validation_indices))],
                    config["batch_size"],
                )
            else:
                no_improvement += 1
        row["seconds"] = time.monotonic() - epoch_start
        history.append(row)
        scheduler.step()
        # The streamed log contains internal metrics only. Outer labels stay unused.
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs or no_improvement >= config["patience"]:
            print(json.dumps({"family": family, "seed": seed, "stage": stage, **row}), flush=True)
        if validation_indices is not None and no_improvement >= config["patience"]:
            break
    if validation_indices is None:
        best_epoch = epochs
        save_checkpoint(checkpoint, checkpoint_payload(model, manifest, config, family, seed, epochs, stage))
    else:
        model, _ = load_checkpoint(checkpoint)
    write_history(output / "history.csv", history)
    # Compare an in-memory selected/refitted model with an independent strict reload.
    probe_indices = (validation_indices if validation_indices is not None else train_indices)[:min(16, len(train_indices))]
    expected = best_probe if validation_indices is not None else predict_probabilities(model, features, probe_indices, config["batch_size"])
    reloaded, _ = load_checkpoint(checkpoint)
    replay = predict_probabilities(reloaded, features, probe_indices, config["batch_size"])
    if not np.array_equal(expected, replay):
        raise RuntimeError("Checkpoint replay differs from selected/refitted model")
    result = {
        "family": family, "seed": seed, "stage": stage,
        "training_rows": len(train_indices), "completed_epochs": len(history),
        "selected_epochs": best_epoch,
        "internal_validation_metrics": best_metrics,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "parameter_counts": parameter_counts(model), "standardization": scaler_record,
        "gradient_check": checks, "checkpoint_reload_exact": True,
        "first_epoch_training_loss": history[0]["train_loss"],
        "last_epoch_training_loss": history[-1]["train_loss"],
        "training_seconds": time.monotonic() - training_start,
    }
    dump_json(output / "training_record.json", result)
    return result


def write_submission(path: Path, ids: list[str], probabilities: np.ndarray, classes: list[str]) -> None:
    validate_probabilities(probabilities, len(ids), len(classes))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Id", "Label"])
        writer.writerows((identifier, classes[int(label)]) for identifier, label in zip(ids, probabilities.argmax(axis=1)))


def mean_probabilities(arrays: list[np.ndarray]) -> np.ndarray:
    # Fixed equal weights; no seed selection based on validation performance.
    result = np.stack(arrays, axis=0).mean(axis=0, dtype=np.float64).astype(np.float32)
    validate_probabilities(result, arrays[0].shape[0], arrays[0].shape[1])
    return result


def fit(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    if (output / "selection_summary.json").exists():
        raise FileExistsError("This output already contains a completed fit; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(Path(args.manifest))
    config = dict(DEFAULTS)
    config.update({
        "seeds": [int(x) for x in args.seeds.split(",")],
        "families": args.families.split(","),
        "max_epochs": args.max_epochs, "patience": args.patience,
        "batch_size": args.batch_size,
    })
    if len(config["seeds"]) != len(set(config["seeds"])) or not config["seeds"]:
        raise ValueError("Seed list must be nonempty and unique")
    if set(config["families"]) - {"mlp", "transformer"} or not config["families"]:
        raise ValueError("Families must be mlp and/or transformer")
    if config["max_epochs"] < 1 or config["patience"] < 1 or config["batch_size"] < 1:
        raise ValueError("Epoch, patience and batch sizes must be positive")
    inner_train, inner_validation = get_inner_split(manifest, config)
    features, feature_records = load_features(manifest["train_paths"], manifest["names"], len(manifest["ids"]))
    configuration = {
        "configuration": config,
        "manifest_path": str(manifest["path"]), "manifest_sha256": sha256(manifest["path"]),
        "manifest": manifest["source"], "feature_files": feature_records,
        "script_sha256": sha256(Path(__file__)),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "threads": args.threads},
        "selection_rule": "Per-seed epochs: inner correct count, then lower log loss. Family: equal-seed averaged inner probability accuracy, then lower log loss, then fewer parameters.",
        "outer_evaluation": "Deferred: this script never calculates metrics using outer validation labels.",
        "normalization": "Mean and standard deviation fitted on inner training for selection, original training for refit; per-dimension std floor then input clipping to +/-8.",
        "class_order": manifest["class_order"], "branch_names": manifest["names"],
    }
    dump_json(output / "configuration.json", configuration)
    split_record = {
        "train_indices": manifest["train_indices"].tolist(),
        "outer_validation_indices": manifest["validation_indices"].tolist(),
        "inner_train_indices": inner_train.tolist(), "inner_validation_indices": inner_validation.tolist(),
        "inner_train_ids": [manifest["ids"][int(x)] for x in inner_train],
        "inner_validation_ids": [manifest["ids"][int(x)] for x in inner_validation],
        "outer_validation_ids": [manifest["ids"][int(x)] for x in manifest["validation_indices"]],
        "inner_group_overlap": 0, "outer_group_overlap": 0,
        "labelled_ids_sha256": sequence_sha256(manifest["ids"]),
    }
    dump_json(output / "split.json", split_record)
    dump_json(output / "labeled_ids.json", manifest["ids"])
    dump_json(output / "class_order.json", manifest["class_order"])
    if manifest["test_ids"]:
        dump_json(output / "test_ids.json", manifest["test_ids"])
    all_indices = np.arange(len(manifest["ids"]), dtype=np.int64)
    families: dict[str, Any] = {}
    # Finish ALL internal family/seed selection before any outer predictions.
    for family in config["families"]:
        records = []
        inner_arrays = []
        for seed in config["seeds"]:
            seed_output = output / family / f"seed_{seed}" / "selection"
            record = train_seed(manifest, features, config, family, seed, inner_train,
                                inner_validation, config["max_epochs"], seed_output, "internal_selection")
            model, _ = load_checkpoint(Path(record["checkpoint"]))
            probabilities = predict_probabilities(model, features, inner_validation, config["batch_size"])
            np.save(seed_output / "internal_validation_probabilities.npy", probabilities)
            inner_arrays.append(probabilities)
            records.append(record)
            del model
        average = mean_probabilities(inner_arrays)
        np.save(output / family / "internal_validation_probabilities.npy", average)
        metrics = classification_metrics(average, manifest["labels"][inner_validation])
        families[family] = {
            "family": family, "internal_ensemble_metrics": metrics,
            "per_seed": records,
            "internal_seed_accuracy_mean": float(np.mean([r["internal_validation_metrics"]["accuracy"] for r in records])),
            "internal_seed_accuracy_sample_sd": float(np.std([r["internal_validation_metrics"]["accuracy"] for r in records], ddof=1)) if len(records) > 1 else 0.0,
            "parameter_counts_per_seed": records[0]["parameter_counts"],
        }
    selected_family = max(families, key=lambda name: (
        families[name]["internal_ensemble_metrics"]["correct"],
        -families[name]["internal_ensemble_metrics"]["log_loss"],
        -families[name]["parameter_counts_per_seed"]["total_trainable"],
    ))
    summary = {
        "selected_family_by_internal_validation": selected_family,
        "families": families,
        "configuration_sha256": sha256(output / "configuration.json"),
        "split_sha256": sha256(output / "split.json"),
        "outer_metrics_calculated": False,
        "refit_completed": False,
        "test_prediction_completed": False,
        "selection_locked_before_refit": True,
    }
    # Save a pre-refit lock that remains untouched as evidence of internal choices.
    dump_json(output / "selection_lock.json", summary)
    print(json.dumps({"event": "internal_selection_locked", "selected_family": selected_family,
                      "metrics": {name: value["internal_ensemble_metrics"] for name, value in families.items()}}), flush=True)
    if not args.skip_refit:
        for family in config["families"]:
            arrays = []
            refit_records = []
            for selection in families[family]["per_seed"]:
                seed = selection["seed"]
                seed_output = output / family / f"seed_{seed}" / "refit"
                record = train_seed(manifest, features, config, family, seed, manifest["train_indices"],
                                    None, selection["selected_epochs"], seed_output, "original_training_refit")
                model, _ = load_checkpoint(Path(record["checkpoint"]))
                probabilities = predict_probabilities(model, features, all_indices, config["batch_size"])
                np.save(seed_output / "labeled_probabilities.npy", probabilities)
                arrays.append(probabilities)
                refit_records.append(record)
                del model
            average = mean_probabilities(arrays)
            np.save(output / family / "labeled_probabilities.npy", average)
            np.save(output / family / "outer_validation_probabilities.npy", average[manifest["validation_indices"]])
            families[family]["refit_per_seed"] = refit_records
            families[family]["labeled_probabilities_sha256"] = sha256(output / family / "labeled_probabilities.npy")
            dump_json(output / family / "record.json", families[family])
        summary["refit_completed"] = True
    dump_json(output / "selection_summary.json", summary)
    # Free labelled feature memory before reading test features.
    del features
    if not args.skip_test and not args.skip_refit and manifest["test_paths"]:
        predict_run(output, manifest, output, args.batch_size, "test", emit_submission=True)
        summary["test_prediction_completed"] = True
        dump_json(output / "selection_summary.json", summary)
    print(json.dumps({"event": "fit_complete", "output": str(output),
                      "selected_family": selected_family, "outer_metrics_calculated": False}), flush=True)


def predict_run(run: Path, manifest: dict[str, Any], output: Path, batch_size: int,
                partition: str, emit_submission: bool = True) -> None:
    summary = json.loads((run / "selection_summary.json").read_text())
    if not summary.get("refit_completed"):
        raise ValueError("Prediction requires completed refitted checkpoints")
    configuration = json.loads((run / "configuration.json").read_text())
    if configuration["branch_names"] != manifest["names"]:
        raise ValueError("Feature branch order does not match the training run")
    if configuration["class_order"] != manifest["class_order"]:
        raise ValueError("Class order does not match the training run")
    if partition == "test":
        if not manifest["test_paths"]:
            raise ValueError("Test feature paths are missing")
        paths, ids = manifest["test_paths"], manifest["test_ids"]
    else:
        paths, ids = manifest["train_paths"], manifest["ids"]
    first_family = next(iter(summary["families"]))
    first_seed = summary["families"][first_family]["refit_per_seed"][0]["seed"]
    model, payload = load_checkpoint(run / first_family / f"seed_{first_seed}" / "refit" / "model.pt")
    features, feature_records = load_features(paths, manifest["names"], len(ids), payload["dimensions"])
    del model
    indices = np.arange(len(ids), dtype=np.int64)
    records = {}
    for family, details in summary["families"].items():
        arrays = []
        for seed_record in details["refit_per_seed"]:
            seed = seed_record["seed"]
            checkpoint = run / family / f"seed_{seed}" / "refit" / "model.pt"
            model, metadata = load_checkpoint(checkpoint)
            if metadata["branch_names"] != manifest["names"] or metadata["class_order"] != manifest["class_order"]:
                raise ValueError("Checkpoint metadata mismatch")
            probabilities = predict_probabilities(model, features, indices, batch_size)
            seed_output = output / family / f"seed_{seed}" / "refit"
            seed_output.mkdir(parents=True, exist_ok=True)
            np.save(seed_output / f"{partition}_probabilities.npy", probabilities)
            arrays.append(probabilities)
            del model
        average = mean_probabilities(arrays)
        family_output = output / family
        family_output.mkdir(parents=True, exist_ok=True)
        probability_path = family_output / f"{partition}_probabilities.npy"
        np.save(probability_path, average)
        record = {"rows": len(ids), "probabilities_sha256": sha256(probability_path),
                  "equal_seed_weights": [1 / len(arrays)] * len(arrays)}
        if partition == "test" and emit_submission:
            submission = family_output / "submission.csv"
            write_submission(submission, ids, average, manifest["class_order"])
            record["submission_sha256"] = sha256(submission)
        records[family] = record
    dump_json(output / f"{partition}_ids.json", ids)
    dump_json(output / f"{partition}_prediction_record.json", {
        "source_run": str(run), "partition": partition,
        "manifest_sha256": sha256(manifest["path"]),
        "feature_files": feature_records, "class_order": manifest["class_order"],
        "ids_sha256": sequence_sha256(ids), "families": records,
        "outer_metrics_calculated": False,
    })
    print(json.dumps({"event": "prediction_complete", "partition": partition, "rows": len(ids), "output": str(output)}), flush=True)


def predict(args: argparse.Namespace) -> None:
    manifest = load_manifest(Path(args.manifest), require_split=False)
    predict_run(Path(args.run).resolve(), manifest, Path(args.output).resolve(),
                args.batch_size, args.partition, emit_submission=not args.no_submission)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("fit", help="Select on grouped internal validation and refit fixed heads")
    fit_parser.add_argument("--manifest", required=True)
    fit_parser.add_argument("--output", required=True)
    fit_parser.add_argument("--families", default=",".join(DEFAULTS["families"]))
    fit_parser.add_argument("--seeds", default=",".join(map(str, DEFAULTS["seeds"])))
    fit_parser.add_argument("--max-epochs", type=int, default=DEFAULTS["max_epochs"])
    fit_parser.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    fit_parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    fit_parser.add_argument("--threads", type=int, default=2)
    fit_parser.add_argument("--skip-refit", action="store_true")
    fit_parser.add_argument("--skip-test", action="store_true")
    fit_parser.set_defaults(function=fit)
    predict_parser = commands.add_parser("predict", help="Replay refitted heads on cached features")
    predict_parser.add_argument("--run", required=True)
    predict_parser.add_argument("--manifest", required=True)
    predict_parser.add_argument("--output", required=True)
    predict_parser.add_argument("--partition", choices=["test", "labeled"], default="test")
    predict_parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    predict_parser.add_argument("--threads", type=int, default=2)
    predict_parser.add_argument("--no-submission", action="store_true")
    predict_parser.set_defaults(function=predict)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_cpu(args.threads)
    args.function(args)


if __name__ == "__main__":
    main()
