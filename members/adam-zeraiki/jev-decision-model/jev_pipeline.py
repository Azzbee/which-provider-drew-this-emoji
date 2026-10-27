#!/usr/bin/env python3
"""Reproducible two-stage, budget-capped hosted Jev emoji experiment.

`prepare` and `dry-run` are offline. Actual API access lives only in jev_client.py.
The old 1,976-image development set has been reused for model selection.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageOps

CLASSES = ["apple", "facebook", "google", "messenger", "mozilla", "samsung", "whatsapp"]
MODEL = "jev-1.14.0"
SEED = 20261018
NLL_FLOOR = 1e-12
VERSION = 1

EXPERT_INSTRUCTIONS = (
    "Identify which provider rendered the query emoji image. Classify the visual "
    "rendering style/provider, not the emoji's meaning. The seven labelled reference "
    "images are examples from the training data; they may depict different emoji "
    "contents and are not exhaustive descriptions of a provider. Facebook and "
    "Messenger are distinct dataset labels. Select exactly one allowed provider."
)
FINAL_INSTRUCTIONS = (
    "Make the final provider classification for the query emoji image. Classify its "
    "visual rendering style/provider, not its meaning. The state includes the full "
    "seven-class probability distributions of DINOv2, ConvNeXt, and a separate "
    "image-only Jev pass. Treat all three as fallible evidence together with the "
    "image and labelled training references; select the provider best supported "
    "by that evidence. Reference images may depict different emoji contents. "
    "Facebook and Messenger are distinct dataset labels. Select exactly one "
    "allowed provider."
)


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def probability_matrix(value, rows=None):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 7 or (rows is not None and len(value) != rows):
        raise ValueError(f"Expected {rows or 'N'} x 7 probabilities; got {value.shape}")
    if not np.isfinite(value).all() or (value < 0).any() or (value > 1).any():
        raise ValueError("Probability matrix contains invalid values")
    if not np.allclose(value.sum(1), 1.0, atol=1e-5, rtol=0):
        raise ValueError("Probability rows do not sum to one")
    return value / value.sum(1, keepdims=True)


def metrics(probabilities, labels):
    p = probability_matrix(probabilities, len(labels))
    y = np.asarray(labels, dtype=np.int64)
    pred = p.argmax(1)
    return {
        "rows": len(y), "correct": int(np.sum(pred == y)),
        "accuracy": float(np.mean(pred == y)),
        "nll": float(-np.log(np.clip(p[np.arange(len(y)), y], NLL_FLOOR, 1)).mean()),
        "nll_definition": "float64; normalize each row; true-class probability clipped to [1e-12,1]",
    }


def comparison(candidate, baseline, labels):
    cm, bm = metrics(candidate, labels), metrics(baseline, labels)
    y = np.asarray(labels)
    c, b = np.asarray(candidate).argmax(1), np.asarray(baseline).argmax(1)
    fixed = int(np.sum((b != y) & (c == y)))
    broken = int(np.sum((b == y) & (c != y)))
    n = fixed + broken
    # Exact two-sided paired sign/McNemar binomial p, descriptive only.
    p = min(1., 2 * sum(math.comb(n, i) for i in range(min(fixed, broken) + 1)) / 2**n) if n else 1.
    return {**cm, "errors_corrected": fixed, "new_errors": broken,
            "net_additional_correct": fixed - broken,
            "prediction_changes": int(np.sum(c != b)),
            "paired_exact_p_descriptive": p,
            "passes_frozen_exploratory_gate": bool(fixed - broken >= 8 and cm["nll"] <= bm["nll"])}


def transform_image(source):
    """Composite alpha onto white and letterbox to RGB 280x280; no crop/augmentation."""
    with Image.open(source) as src:
        rgba = ImageOps.exif_transpose(src).convert("RGBA")
        rgb = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba).convert("RGB")
        small = ImageOps.contain(rgb, (280, 280), Image.Resampling.LANCZOS)
        out = Image.new("RGB", (280, 280), "white")
        out.paste(small, ((280 - small.width) // 2, (280 - small.height) // 2))
        stream = io.BytesIO()
        out.save(stream, format="PNG", optimize=False)
        return stream.getvalue()


def materialize_image(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(transform_image(source))
    return {"original_sha256": sha256(source), "prepared_sha256": sha256(target)}


def load_client_module(path=None):
    if path:
        p = Path(path)
        p = p / "jev_client.py" if p.is_dir() else p
    else:
        p = Path(__file__).with_name("jev_client.py")
    if not p.is_file():
        raise RuntimeError("Place jev_client.py beside this script or pass --client-module PATH")
    spec = importlib.util.spec_from_file_location("jev_client", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def prepare(args):
    out, bundle, dataset = Path(args.run_dir), Path(args.bundle), Path(args.dataset_root)
    if (out / "protocol.json").exists():
        raise ValueError("Prepared protocol already exists; use another run directory to avoid changing a frozen experiment")
    md = bundle / "metadata"
    ids, labels, groups, train, val, classes, test_ids = [read_json(md / f"{n}.json") for n in
        ("ids", "labels", "groups", "train_indices", "validation_indices", "class_order", "test_ids")]
    if classes != CLASSES or len(ids) != 9879 or len(val) != 1976 or len(train) != 7903:
        raise ValueError("Unexpected dataset or class order")
    if len(set(ids)) != len(ids) or set(train) & set(val) or set(train) | set(val) != set(range(len(ids))):
        raise ValueError("Training/development split is not a disjoint full partition")
    y, g, train, val = np.asarray(labels), np.asarray(groups), np.asarray(train), np.asarray(val)
    if set(g[train]) & set(g[val]):
        raise ValueError("Duplicate-group leakage between original train and development sets")
    source_files = {str(p.relative_to(bundle)): sha256(p) for p in md.glob("*.json")}
    base = {}
    for model in ("dino", "convnext"):
        fp = bundle / "features" / model / "train_tuned_probabilities.npy"
        fm = read_json(fp.with_name("train_metadata.json"))
        if fm["ids"] != ids:
            raise ValueError(f"{model} feature/metadata row alignment mismatch")
        base[model] = probability_matrix(np.load(fp), len(ids))
        source_files[str(fp.relative_to(bundle))] = sha256(fp)
        source_files[str(fp.with_name("train_metadata.json").relative_to(bundle))] = sha256(fp.with_name("train_metadata.json"))
    feature_path = bundle / "features/dino/train_pretrained.npy"
    if sha256(feature_path) != "9b1c51df098ff6a61bb42af6c29fa89ed5f8230aeaa772883e423fa8e7365293":
        raise ValueError("Original public pretrained DINO reference features have changed")
    features = np.asarray(np.load(feature_path), dtype=np.float64)
    if features.shape != (9879, 768) or not np.isfinite(features).all():
        raise ValueError("Invalid original public pretrained DINO features")
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    source_files[str(feature_path.relative_to(bundle))] = sha256(feature_path)
    references, ref_groups = [], set()
    for k, cls in enumerate(CLASSES):
        candidates = np.array(sorted(train[y[train] == k], key=lambda i: int(ids[i])))
        mean = features[candidates].mean(0)
        mean /= max(np.linalg.norm(mean), 1e-12)
        scores = features[candidates] @ mean
        order = sorted(range(len(candidates)), key=lambda j: (-float(scores[j]), int(ids[candidates[j]])))
        row = next(int(candidates[j]) for j in order if int(g[candidates[j]]) not in ref_groups)
        ref_groups.add(int(g[row]))
        references.append({"class": cls, "row": row, "id": ids[row], "group": int(g[row]),
                           "image": f"images/references/{cls}.png"})
    rng = np.random.default_rng(SEED)
    blocked = set(g[val]) | ref_groups
    pilot, pilot_groups = [], set()
    for k in range(7):
        candidates = sorted((int(i) for i in train if y[i] == k and int(g[i]) not in blocked), key=lambda i: int(ids[i]))
        for i in rng.permutation(candidates):
            i = int(i)
            if int(g[i]) not in pilot_groups:
                pilot.append(i); pilot_groups.add(int(g[i]))
                if sum(y[j] == k for j in pilot) == 7:
                    break
    if len(pilot) != 49:
        raise ValueError("Insufficient distinct training groups for balanced pilot")
    extra = sorted((int(i) for i in train if int(g[i]) not in blocked | pilot_groups), key=lambda i: int(ids[i]))
    pilot.append(int(rng.choice(extra)))
    pilot = list(map(int, rng.permutation(pilot)))
    disagreements = val[base["dino"][val].argmax(1) != base["convnext"][val].argmax(1)]
    remaining = val[base["dino"][val].argmax(1) == base["convnext"][val].argmax(1)]
    execution = list(map(int, rng.permutation(disagreements))) + list(map(int, rng.permutation(remaining)))
    out.mkdir(parents=True, exist_ok=True)
    for r in references:
        r.update(materialize_image(dataset / "train" / f"{r['id']}.png", out / r["image"]))
    partitions = {}
    for name, rows in (("pilot", pilot), ("development", execution)):
        entries = []
        for row in rows:
            entry = {"row": row, "id": ids[row], "label": int(y[row]), "group": int(g[row]),
                     "image": f"images/{name}/{ids[row]}.png"}
            entry.update(materialize_image(dataset / "train" / f"{ids[row]}.png", out / entry["image"]))
            entries.append(entry)
        for model in ("dino", "convnext"):
            p = out / f"{name}_{model}.npy"
            np.save(p, base[model][rows])
        partitions[name] = {"rows": entries, "dino_file": f"{name}_dino.npy", "convnext_file": f"{name}_convnext.npy",
                            "dino_sha256": sha256(out / f"{name}_dino.npy"),
                            "convnext_sha256": sha256(out / f"{name}_convnext.npy")}
    deployment = None
    if args.deployment_root:
        dep = Path(args.deployment_root)
        if read_json(dep / "test_ids.json") != test_ids or test_ids != [str(i) for i in range(10001, 19880)]:
            raise ValueError("All-label deployment test IDs differ from original test IDs")
        deployment = {"ids": test_ids, "labels_used_to_train_base_models": 9879,
                      "has_independent_validation_score": False}
        for model in ("dino", "convnext"):
            p = dep / model / "test_probabilities.npy"
            probability_matrix(np.load(p), 9879)
            shutil.copyfile(p, out / f"test_{model}.npy")
            deployment[f"{model}_file"] = f"test_{model}.npy"
            deployment[f"{model}_sha256"] = sha256(out / f"test_{model}.npy")
    protocol = {
        "format_version": VERSION, "created_utc": utcnow(), "model": MODEL, "class_order": CLASSES,
        "selection_seed": SEED, "image_preprocessing": "EXIF transpose; RGBA white composite; RGB; Lanczos contain; white 280x280 centered letterbox; PNG",
        "reference_selection": "Per-class medoid: original public-pretrained DINO features, float64 row L2 normalization, original training-class normalized centroid, maximum cosine, numeric-ID tie-break; duplicate reference groups excluded",
        "references": references, "partitions": partitions, "deployment": deployment,
        "prompts": {"expert": EXPERT_INSTRUCTIONS, "final": FINAL_INSTRUCTIONS},
        "base_weights": {"dino": 0.5, "convnext": 0.5},
        "fixed_third_expert_weights": {"dino": 0.4, "convnext": 0.4, "jev": 0.2},
        "router": "Use final Jev only where DINOv2 and ConvNeXt top labels disagree; otherwise exact baseline probabilities",
        "development_order": "All original model disagreements first, then agreements; each segment deterministically shuffled without labels",
        "gate": {"minimum_net_additional_correct": 8, "maximum_nll": "recomputed float64 baseline NLL", "candidate_priority": ["third_expert_blend", "final_jev", "disagreement_router"]},
        "evaluation_warning": "REUSED development set used in earlier model selection; results are exploratory and not an unbiased final-test score. Pilot is training data and is for integration/cost checks only.",
        "nll_floor": NLL_FLOOR, "budget_usd": "5", "paired_p": "exact two-sided binomial, descriptive only; no significance claim",
        "source_hashes": source_files, "pipeline_sha256": sha256(__file__),
        "ai_usage": "AI generated experiment code and prompt text. Students must write their own course report and disclose prompts/code assistance.",
    }
    write_json(out / "protocol.json", protocol)
    (out / "protocol.sha256").write_text(digest(protocol) + "\n")
    development = partitions["development"]
    d, c = np.load(out / development["dino_file"]), np.load(out / development["convnext_file"])
    baseline = metrics((d + c) / 2, [x["label"] for x in development["rows"]])
    if baseline["correct"] != 1919:
        raise ValueError("Original baseline did not reproduce 1919/1976")
    receipt = {"prepared_utc": utcnow(), "protocol_sha256": digest(protocol),
               "reference_ids": {r["class"]: r["id"] for r in references},
               "pilot_rows": len(pilot), "development_rows": len(val),
               "development_disagreements": len(disagreements), "baseline": baseline,
               "historical_float32_clipped_nll": 0.1376239145136072,
               "nll_note": "This experiment recomputes every NLL with the same float64/floor1e-12 convention; do not compare its NLL gate to historical float32-clipped NLL.",
               "api_calls": 0, "api_spend_usd": "0", "jev_results_available": False}
    write_json(out / "prepared_receipt.json", receipt)
    return receipt


def load_protocol(run_dir, check_code=True):
    root = Path(run_dir)
    p = read_json(root / "protocol.json")
    if digest(p) != (root / "protocol.sha256").read_text().strip():
        raise ValueError("Frozen protocol hash mismatch")
    if p["model"] != MODEL or p["class_order"] != CLASSES:
        raise ValueError("Unexpected frozen model/classes")
    if check_code and sha256(__file__) != p["pipeline_sha256"]:
        raise ValueError("Pipeline code changed after freezing; prepare a new run before any API call")
    return p


def load_partition(root, protocol, partition):
    p = protocol["partitions"][partition]
    matrices = []
    for model in ("dino", "convnext"):
        file = Path(root) / p[f"{model}_file"]
        if sha256(file) != p[f"{model}_sha256"]:
            raise ValueError("Frozen base probabilities changed")
        matrices.append(probability_matrix(np.load(file), len(p["rows"])))
    return p["rows"], *matrices


def image_asset(root, entry):
    p = Path(root) / entry["image"]
    if sha256(p) != entry["prepared_sha256"]:
        raise ValueError("Prepared image changed")
    with Image.open(p) as img:
        if img.size != (280, 280) or img.mode != "RGB" or img.format != "PNG":
            raise ValueError("Prepared image violates frozen encoding")
    return {"type": "image", "format": "png", "data_base64": base64.b64encode(p.read_bytes()).decode("ascii")}


def payload_for(root, protocol, row, dino=None, convnext=None, jev=None):
    final = jev is not None
    query_ref = "asset_q_6b379f"
    assets = {query_ref: image_asset(root, row)}
    references = []
    for k, r in enumerate(protocol["references"]):
        ref = f"asset_r_{k}_791ec2"
        assets[ref] = image_asset(root, r)
        references.append({"provider": r["class"], "image": ref})
    # Explicit allowlist. Never copy a local row/metadata object into an API request.
    state = {"query_image": query_ref, "training_references": references}
    if final:
        if dino is None or convnext is None:
            raise ValueError("Final Jev requires all three expert distributions")
        state["expert_probabilities"] = {
            name: {cls: float(v) for cls, v in zip(CLASSES, probability_matrix([p])[0])}
            for name, p in (("dinov2", dino), ("convnext", convnext), ("jev_image_only", jev))}
    question = {"type": "choice", "instructions": protocol["prompts"]["final" if final else "expert"],
                "criteria": {cls: f"The query emoji was rendered by the {cls} provider." for cls in CLASSES}}
    return {"model": MODEL, "state": state, "assets": assets, "questions": {"provider": question}}


def validate_payload_shape(payload, final):
    if set(payload) != {"model", "state", "assets", "questions"} or payload["model"] != MODEL:
        raise ValueError("Unexpected outbound payload fields")
    allowed = {"query_image", "training_references"} | ({"expert_probabilities"} if final else set())
    if set(payload["state"]) != allowed or len(payload["assets"]) != 8:
        raise ValueError("Unexpected state fields/image count")
    if set(payload["questions"]) != {"provider"} or list(payload["questions"]["provider"]["criteria"]) != CLASSES:
        raise ValueError("Choice question must contain exactly seven fixed classes")
    if final:
        experts = payload["state"]["expert_probabilities"]
        if set(experts) != {"dinov2", "convnext", "jev_image_only"}:
            raise ValueError("Final decision must include all three experts")
        for p in experts.values():
            if list(p) != CLASSES:
                raise ValueError("Expert class order changed")
            probability_matrix([list(p.values())])


def dry_run(args):
    root = Path(args.run_dir)
    p = load_protocol(root)
    count = 0
    h = hashlib.sha256()
    for partition in ("pilot", "development"):
        rows, dino, conv = load_partition(root, p, partition)
        for i, row in enumerate(rows):
            expert = payload_for(root, p, row)
            final = payload_for(root, p, row, dino[i], conv[i], np.full(7, 1/7))
            validate_payload_shape(expert, False)
            validate_payload_shape(final, True)
            h.update(canonical({"expert": digest(expert), "final_schema_only": digest(final)}))
            count += 2
    receipt = {"checked_utc": utcnow(), "protocol_sha256": digest(p), "request_shapes_checked": count,
               "request_shapes_sha256": h.hexdigest(), "api_calls": 0, "api_spend_usd": "0",
               "status": "OFFLINE_READY_AWAITING_ACTUAL_JEV_ACCESS", "jev_metrics": None,
               "final_stage_note": "Dry-run uses a uniform synthetic Jev vector solely to validate request structure. It is never submitted or counted as a Jev result."}
    write_json(root / "dry_run_receipt.json", receipt)
    return receipt


def parsed_vector(mod, response):
    parsed = mod.parse_choice(response, "provider", CLASSES)
    return probability_matrix([[parsed["probabilities"][c] for c in CLASSES]])[0]


def outcome_path(root, partition, row):
    return Path(root) / "results" / partition / f"{row['id']}.json"


def read_outcome(root, protocol, partition, row, dino, conv, mod, require_final=True):
    path = outcome_path(root, partition, row)
    if not path.exists():
        return None
    item = read_json(path)
    if item["protocol_sha256"] != digest(protocol) or item["id"] != row["id"] or item["partition"] != partition:
        raise ValueError("Outcome provenance does not match frozen experiment")
    expert_payload = payload_for(root, protocol, row)
    if item["expert"]["payload_sha256"] != digest(expert_payload):
        raise ValueError("Expert outcome request hash mismatch")
    if digest(item["expert"]["response"]) != item["expert"]["response_sha256"]:
        raise ValueError("Expert response hash mismatch")
    j = parsed_vector(mod, item["expert"]["response"])
    f = None
    if item.get("final"):
        final_payload = payload_for(root, protocol, row, dino, conv, j)
        if item["final"]["payload_sha256"] != digest(final_payload):
            raise ValueError("Final outcome request hash mismatch")
        if digest(item["final"]["response"]) != item["final"]["response_sha256"]:
            raise ValueError("Final response hash mismatch")
        f = parsed_vector(mod, item["final"]["response"])
    if require_final and f is None:
        return None
    return item, j, f


def run_one(root, protocol, partition, row, dino, conv, client, mod, final_needed=True):
    existing = read_outcome(root, protocol, partition, row, dino, conv, mod, require_final=final_needed)
    if existing is not None:
        return existing
    key_prefix = f"{digest(protocol)[:16]}:{partition}:{row['id']}"
    payload = payload_for(root, protocol, row)
    validate_payload_shape(payload, False)
    context = {"protocol_sha256": digest(protocol), "partition": partition,
               "image_id": row["id"], "query_image_sha256": row["prepared_sha256"], "stage": "expert"}
    raw = client.run(payload, request_key=key_prefix + ":expert", context=context)
    j = parsed_vector(mod, raw)
    item = {"protocol_sha256": digest(protocol), "partition": partition, "id": row["id"],
            "expert": {"payload_sha256": digest(payload), "response_sha256": digest(raw), "response": raw},
            "final": None, "updated_utc": utcnow()}
    write_json(outcome_path(root, partition, row), item)
    f = None
    if final_needed:
        payload = payload_for(root, protocol, row, dino, conv, j)
        validate_payload_shape(payload, True)
        context["stage"] = "final"
        raw = client.run(payload, request_key=key_prefix + ":final", context=context)
        f = parsed_vector(mod, raw)
        item["final"] = {"payload_sha256": digest(payload), "response_sha256": digest(raw), "response": raw}
        item["updated_utc"] = utcnow()
        write_json(outcome_path(root, partition, row), item)
    return item, j, f


def run_partition(args):
    root, partition = Path(args.run_dir), args.partition
    protocol = load_protocol(root)
    mod = load_client_module(args.client_module)
    client = mod.BudgetedJevClient(root / "api_cache", budget_usd=protocol["budget_usd"])
    rows, dino, conv = load_partition(root, protocol, partition)
    if partition == "development":
        pilot = root / "pilot_run_receipt.json"
        if not pilot.exists() or not read_json(pilot).get("complete"):
            raise ValueError("Complete the 50-image technical pilot before development API calls")
    requested = min(args.limit or len(rows), len(rows))
    complete, stop = 0, None
    try:
        for i, row in enumerate(rows[:requested]):
            run_one(root, protocol, partition, row, dino[i], conv[i], client, mod)
            complete += 1
            if complete % 10 == 0:
                print(json.dumps({"partition": partition, "completed_this_pass": complete,
                                  "budget": client.budget_summary()}), flush=True)
    except mod.JevClientError as exc:
        stop = {"type": type(exc).__name__, "message": str(exc)}
    receipt = {"updated_utc": utcnow(), "protocol_sha256": digest(protocol), "partition": partition,
               "requested_rows": requested, "completed_this_pass": complete,
               "complete": complete == len(rows), "stop": stop, "budget": client.budget_summary(),
               "warning": protocol["evaluation_warning"]}
    write_json(root / f"{partition}_run_receipt.json", receipt)
    if partition == "pilot" and receipt["complete"]:
        tokens = {"expert": [], "final": []}
        for i, row in enumerate(rows):
            outcome = read_outcome(root, protocol, partition, row, dino[i], conv[i], mod)[0]
            for stage in tokens:
                tokens[stage].append(outcome[stage]["response"]["usage"]["input_tokens"])
        receipt["observed_input_tokens"] = {k: {"min": min(v), "max": max(v), "mean": sum(v)/len(v)} for k, v in tokens.items()}
        receipt["estimated_full_development_usd_at_pilot_mean"] = sum(sum(v)/len(v) for v in tokens.values()) * 1976 * .042 / 1_000_000
        receipt["estimate_is_not_a_guarantee"] = True
        write_json(root / "pilot_run_receipt.json", receipt)
    return receipt


def evaluate(args):
    root = Path(args.run_dir)
    protocol = load_protocol(root)
    mod = load_client_module(args.client_module)
    rows, dino, conv = load_partition(root, protocol, "development")
    y = np.array([x["label"] for x in rows])
    baseline = (dino + conv) / 2
    mask = dino.argmax(1) != conv.argmax(1)
    third = np.full_like(baseline, np.nan)
    final = np.full_like(baseline, np.nan)
    available = np.zeros(len(rows), dtype=bool)
    for i, row in enumerate(rows):
        outcome = read_outcome(root, protocol, "development", row, dino[i], conv[i], mod)
        if outcome is not None:
            _, third[i], final[i] = outcome
            available[i] = True
    candidates = {}
    if np.all(available):
        candidates["jev_image_only_diagnostic"] = comparison(third, baseline, y)
        candidates["jev_image_only_diagnostic"]["eligible_for_deployment"] = False
        candidates["third_expert_blend"] = comparison(.4*dino + .4*conv + .2*third, baseline, y)
        candidates["final_jev"] = comparison(final, baseline, y)
    if np.all(available[mask]):
        routed = baseline.copy()
        routed[mask] = final[mask]
        candidates["disagreement_router"] = comparison(routed, baseline, y)
    for name, result in candidates.items():
        if name != "jev_image_only_diagnostic":
            result["eligible_for_deployment"] = result["passes_frozen_exploratory_gate"]
    priority = protocol["gate"]["candidate_priority"]
    selected = next((n for n in priority if candidates.get(n, {}).get("eligible_for_deployment")), None)
    receipt = {"evaluated_utc": utcnow(), "protocol_sha256": digest(protocol),
               "warning": protocol["evaluation_warning"], "baseline": metrics(baseline, y),
               "complete_two_stage_rows": int(available.sum()), "required_rows_for_all_candidates": len(rows),
               "routed_rows_completed": int(available[mask].sum()), "routed_rows_required": int(mask.sum()),
               "candidates": candidates, "selected_deployment_candidate": selected,
               "selection_rule": "First passing candidate in preregistered priority, not highest observed accuracy",
               "missing_rows_are_not_filled_as_Jev_predictions": True,
               "nll_note": "Float64 normalization/floor1e-12 shared by all candidates and baseline; historical float32 NLL differs slightly."}
    write_json(root / "evaluation.json", receipt)
    return receipt


def budget(args):
    p = load_protocol(args.run_dir)
    mod = load_client_module(args.client_module)
    client = mod.BudgetedJevClient(Path(args.run_dir) / "api_cache", budget_usd=p["budget_usd"])
    return client.budget_summary()


def deployment_inputs(root, protocol):
    dep = protocol.get("deployment")
    if not dep:
        raise ValueError("prepare did not include all-label deployment probabilities")
    arrays = []
    for model in ("dino", "convnext"):
        p = Path(root) / dep[f"{model}_file"]
        if sha256(p) != dep[f"{model}_sha256"]:
            raise ValueError("Deployment probability hash changed")
        arrays.append(probability_matrix(np.load(p), len(dep["ids"])))
    return dep["ids"], *arrays


def test_rows(root, ids, needed, dataset_root=None):
    path = Path(root) / "test_images_manifest.json"
    manifest = read_json(path) if path.exists() else {}
    for i in needed:
        item_id = ids[i]
        if item_id not in manifest:
            if dataset_root is None:
                raise ValueError("Test image manifest incomplete; supply --dataset-root to deploy")
            entry = {"id": item_id, "image": f"images/test/{item_id}.png"}
            entry.update(materialize_image(Path(dataset_root) / "test" / f"{item_id}.png", Path(root) / entry["image"]))
            manifest[item_id] = entry
        else:
            image_asset(root, manifest[item_id])
    write_json(path, manifest)
    return manifest


def make_submission(root, protocol, candidate, ids, dino, conv, entries, mod):
    baseline = (dino + conv) / 2
    pred = baseline.copy()
    mask = dino.argmax(1) != conv.argmax(1)
    needed = np.flatnonzero(mask) if candidate == "disagreement_router" else np.arange(len(ids))
    for i in needed:
        item = read_outcome(root, protocol, "test", entries[ids[i]], dino[i], conv[i], mod,
                            require_final=candidate != "third_expert_blend")
        if item is None:
            raise ValueError(f"Missing actual Jev result for test image {ids[i]}; no CSV will be generated")
        _, j, f = item
        pred[i] = .4*dino[i] + .4*conv[i] + .2*j if candidate == "third_expert_blend" else f
    probability_matrix(pred, 9879)
    if ids != [str(i) for i in range(10001, 19880)]:
        raise ValueError("Submission IDs/order differ from competition")
    target = Path(root) / f"submission_jev_{candidate}.csv"
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["Id", "Label"])
    writer.writerows((item_id, CLASSES[int(k)]) for item_id, k in zip(ids, pred.argmax(1)))
    target.write_text(buffer.getvalue())
    np.save(Path(root) / f"test_probabilities_jev_{candidate}.npy", pred)
    return {"csv": target.name, "csv_sha256": sha256(target), "rows": len(ids),
            "jev_queried_rows": len(needed), "prediction_changes_vs_all_label_baseline": int(np.sum(pred.argmax(1) != baseline.argmax(1))),
            "has_independent_validation_score": False, "kaggle_submission_performed": False}


def deploy(args):
    root = Path(args.run_dir)
    protocol = load_protocol(root)
    ev = evaluate(args)
    candidate = ev["selected_deployment_candidate"]
    if candidate is None:
        raise ValueError("No completed candidate passes the frozen exploratory development gate; deployment is blocked")
    mod = load_client_module(args.client_module)
    client = mod.BudgetedJevClient(root / "api_cache", budget_usd=protocol["budget_usd"])
    ids, dino, conv = deployment_inputs(root, protocol)
    needed = np.flatnonzero(dino.argmax(1) != conv.argmax(1)) if candidate == "disagreement_router" else np.arange(len(ids))
    entries = test_rows(root, ids, needed, args.dataset_root)
    complete, stop = 0, None
    try:
        for i in needed:
            run_one(root, protocol, "test", entries[ids[i]], dino[i], conv[i], client, mod,
                    final_needed=candidate != "third_expert_blend")
            complete += 1
            if complete % 25 == 0:
                print(json.dumps({"candidate": candidate, "completed": complete, "required": len(needed),
                                  "budget": client.budget_summary()}), flush=True)
    except mod.JevClientError as exc:
        stop = {"type": type(exc).__name__, "message": str(exc)}
    receipt = {"updated_utc": utcnow(), "protocol_sha256": digest(protocol), "candidate": candidate,
               "required_jev_rows": len(needed), "completed_this_pass": complete,
               "complete": complete == len(needed), "stop": stop, "budget": client.budget_summary(),
               "deployment_transfer_warning": "Base encoders now use all9879training labels. Old development gate evaluates original7903trained base models and is not an independent score of this all-label deployment."}
    if receipt["complete"]:
        receipt.update(make_submission(root, protocol, candidate, ids, dino, conv, entries, mod))
    write_json(root / "deployment_receipt.json", receipt)
    return receipt


def replay(args):
    root = Path(args.run_dir)
    protocol = load_protocol(root)
    ev = evaluate(args)
    path = root / "deployment_receipt.json"
    if not path.exists() or not read_json(path).get("complete"):
        return {"api_calls": 0, "deployment_available": False, "evaluation": ev}
    old = read_json(path)
    candidate = old["candidate"]
    if not ev["candidates"].get(candidate, {}).get("eligible_for_deployment"):
        raise ValueError("Cached deployment lacks a still-verifiable development gate")
    mod = load_client_module(args.client_module)
    ids, dino, conv = deployment_inputs(root, protocol)
    entries = read_json(root / "test_images_manifest.json")
    receipt = make_submission(root, protocol, candidate, ids, dino, conv, entries, mod)
    if receipt["csv_sha256"] != old["csv_sha256"]:
        raise ValueError("Cached replay CSV is not identical to completed deployment")
    return {"api_calls": 0, "bit_identical_csv": True, **receipt}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "dry-run", "run", "evaluate", "budget", "deploy", "replay"):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", required=True, help="Portable prepared experiment directory")
        p.add_argument("--client-module", help="jev_client.py path; defaults beside this script")
        if name == "prepare":
            p.add_argument("--bundle", required=True)
            p.add_argument("--dataset-root", required=True)
            p.add_argument("--deployment-root")
        if name == "run":
            p.add_argument("--partition", choices=("pilot", "development"), required=True)
            p.add_argument("--limit", type=int)
        if name == "deploy":
            p.add_argument("--dataset-root", required=True)
    args = parser.parse_args()
    if getattr(args, "limit", None) is not None and args.limit < 1:
        parser.error("--limit must be positive")
    commands = {"prepare": prepare, "dry-run": dry_run, "run": run_partition,
                "evaluate": evaluate, "budget": budget, "deploy": deploy, "replay": replay}
    try:
        result = commands[args.command](args)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 2 if result.get("stop") else 0
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
