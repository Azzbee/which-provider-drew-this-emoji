#!/usr/bin/env python3
"""Report two prespecified exploratory blends with the prior DINOv2 model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--results", type=Path, default=ROOT / "results")
parser.add_argument("--prior", type=Path, default=ROOT.parent / "cv_next_recovered" / "experiment" / "DINOv2_Emoji_Experiment" / "run_224")
args = parser.parse_args()
RESULTS = args.results
PRIOR = args.prior
CLASSES = json.loads((RESULTS / "classes.json").read_text())
VAL_IDS = json.loads((RESULTS / "validation_ids.json").read_text())
TEST_IDS = json.loads((RESULTS / "test_ids.json").read_text())
Y = np.load(RESULTS / "validation_targets.npy")
with open(PRIOR / "selection" / "validation_predictions.csv", newline="") as f:
    rows = list(csv.DictReader(f))
lookup = {row["Id"].zfill(5): row for row in rows}
assert set(lookup) == set(VAL_IDS) and len(lookup) == len(VAL_IDS)
assert all(lookup[i]["true_label"] == CLASSES[y] for i, y in zip(VAL_IDS, Y))
DINO = np.array([[float(lookup[i][f"p_{c}"]) for c in CLASSES] for i in VAL_IDS], dtype=np.float64)
DINO /= DINO.sum(1, keepdims=True)
DINO_TEST = np.load(PRIOR / "test_probabilities.npy").astype(np.float64)
with open(PRIOR / "submission_dinov2_finetuned.csv", newline="") as f:
    dino_test_rows = list(csv.DictReader(f))
assert [int(r["Id"]) for r in dino_test_rows] == [int(i) for i in TEST_IDS]
assert np.all(DINO_TEST.argmax(1) == np.array([CLASSES.index(r["Label"]) for r in dino_test_rows]))
DINO_TEST /= DINO_TEST.sum(1, keepdims=True)

def scores(prob):
    pred = prob.argmax(1)
    prior_correct = DINO.argmax(1) == Y
    correct = pred == Y
    return {"accuracy": float(accuracy_score(Y, pred)), "correct": int(correct.sum()),
            "total": len(Y), "macro_f1": float(f1_score(Y, pred, average="macro")),
            "log_loss": float(log_loss(Y, prob)),
            "rescued_dino_errors": int((correct & ~prior_correct).sum()),
            "lost_dino_correct": int((~correct & prior_correct).sum()),
            "disagreed_with_dino": int((pred != DINO.argmax(1)).sum())}

report = {"baseline_dinov2": scores(DINO), "attempts": {},
          "status": "Exploratory holdout blend assessment; weights were prespecified at 0.1 and 0.25 for each style model. This shared split also selected the previous DINOv2 epoch, so it is not an independent final generalization estimate.",
          "all_label_refit": False, "kaggle_submission": False}
for name in ["extra_trees", "hist_gradient_boosting"]:
    path = RESULTS / f"{name}_validation_probabilities.npy"
    if not path.exists():
        continue
    val_prob = np.load(path).astype(np.float64)
    test_prob = np.load(RESULTS / f"{name}_test_probabilities.npy").astype(np.float64)
    assert val_prob.shape == DINO.shape and test_prob.shape == DINO_TEST.shape
    assert np.isfinite(val_prob).all() and np.isfinite(test_prob).all()
    val_prob /= val_prob.sum(1, keepdims=True)
    test_prob /= test_prob.sum(1, keepdims=True)
    report["attempts"][name] = {"style_only": scores(val_prob), "blends": {}}
    for style_weight in [.1, .25]:
        blend = (1 - style_weight) * DINO + style_weight * val_prob
        test_blend = (1 - style_weight) * DINO_TEST + style_weight * test_prob
        key = f"dinov2_{name}_style{int(100*style_weight)}"
        metrics = scores(blend)
        metrics["test_changed_from_dino"] = int((test_blend.argmax(1) != DINO_TEST.argmax(1)).sum())
        metrics["style_weight"] = style_weight
        np.save(RESULTS / f"{key}_validation_probabilities.npy", blend.astype(np.float32))
        np.save(RESULTS / f"{key}_test_probabilities.npy", test_blend.astype(np.float32))
        with open(RESULTS / f"submission_{key}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Id", "Label"])
            writer.writerows((int(i), CLASSES[p]) for i, p in zip(TEST_IDS, test_blend.argmax(1)))
        report["attempts"][name]["blends"][key] = metrics
(RESULTS / "exploratory_blends.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
