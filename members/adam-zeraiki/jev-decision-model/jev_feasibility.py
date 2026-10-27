#!/usr/bin/env python3
"""Certify whether the frozen Jev candidates can still meet the accuracy gate.

Reads completed cached responses and frozen local inputs only. It never creates
an API client, calls a network method, reads credentials, or changes the ledger.
The sole output is RUN_DIR/feasibility_receipt.json when invoked successfully.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import jev_client
import jev_pipeline as pipeline


def subset_comparison(candidate, baseline, labels):
    """Observed subset statistics, without applying a full-development gate."""
    y = np.asarray(labels)
    c = np.asarray(candidate).argmax(1)
    b = np.asarray(baseline).argmax(1)
    fixed = int(np.sum((b != y) & (c == y)))
    broken = int(np.sum((b == y) & (c != y)))
    return {
        **pipeline.metrics(candidate, labels),
        "errors_corrected_vs_baseline_on_same_rows": fixed,
        "new_errors_vs_baseline_on_same_rows": broken,
        "net_additional_correct_on_same_rows": fixed - broken,
    }


def assess(run_dir):
    root = Path(run_dir)
    protocol = pipeline.load_protocol(root)
    rows, dino, convnext = pipeline.load_partition(root, protocol, "development")
    if len(rows) != 1976:
        raise ValueError("Expected the frozen 1,976-row reused development set")
    if protocol["fixed_third_expert_weights"] != {"dino": 0.4, "convnext": 0.4, "jev": 0.2}:
        raise ValueError("This certificate applies only to the frozen 0.4/0.4/0.2 blend")
    y = np.asarray([row["label"] for row in rows], dtype=np.int64)
    baseline = (dino + convnext) / 2
    baseline_prediction = baseline.argmax(1)
    routed = dino.argmax(1) != convnext.argmax(1)
    if int(routed.sum()) != 112:
        raise ValueError("Expected exactly 112 original model disagreements")

    available = np.zeros(len(rows), dtype=bool)
    image_jev = np.full_like(baseline, np.nan)
    final_jev = np.full_like(baseline, np.nan)
    outcome_hashes = []
    for i, row in enumerate(rows):
        # read_outcome validates protocol, request and raw-response hashes,
        # exact model/classes, and both completed stage distributions.
        result = pipeline.read_outcome(
            root, protocol, "development", row, dino[i], convnext[i], jev_client
        )
        if result is not None:
            _, image_jev[i], final_jev[i] = result
            available[i] = True
            outcome_hashes.append({
                "id": row["id"],
                "outcome_sha256": pipeline.sha256(pipeline.outcome_path(root, "development", row)),
            })
    if not np.all(available[routed]):
        raise ValueError("Every one of the 112 routed cases must have both cached Jev responses")

    remaining = ~available
    observed_blend = 0.4*dino[available] + 0.4*convnext[available] + 0.2*image_jev[available]
    known = {
        "baseline": pipeline.metrics(baseline[available], y[available]),
        "third_expert_blend": subset_comparison(observed_blend, baseline[available], y[available]),
        "final_jev": subset_comparison(final_jev[available], baseline[available], y[available]),
        "jev_image_only_diagnostic": subset_comparison(image_jev[available], baseline[available], y[available]),
    }
    baseline_metrics = pipeline.metrics(baseline, y)
    gate = baseline_metrics["correct"] + int(protocol["gate"]["minimum_net_additional_correct"])

    # For any future Jev probability vector j, the fixed blend equals
    # 0.8*b + 0.2*j. Giving probability one to the true class and zero to
    # every other class is the best possible j for its classification.
    # A current error can therefore be fixed only if
    #   max_{k != y}(b_k) - b_y <= 0.2/0.8 = 0.25.
    # Include exact/near ties conservatively even if the fixed argmax tie
    # order would still choose the wrong class. This overestimates, never
    # underestimates, the number of possible future correct predictions.
    baseline_correct_remaining = remaining & (baseline_prediction == y)
    baseline_wrong_remaining = remaining & (baseline_prediction != y)
    competitor = baseline.copy()
    competitor[np.arange(len(y)), y] = -np.inf
    gap = competitor.max(1) - baseline[np.arange(len(y)), y]
    potentially_fixable = baseline_wrong_remaining & (gap <= 0.25 + 1e-12)
    unfixable = baseline_wrong_remaining & ~potentially_fixable
    blend_upper = (
        known["third_expert_blend"]["correct"]
        + int(baseline_correct_remaining.sum())
        + int(potentially_fixable.sum())
    )
    direct_upper = known["final_jev"]["correct"] + int(remaining.sum())
    independent_upper = known["jev_image_only_diagnostic"]["correct"] + int(remaining.sum())

    router = baseline.copy()
    router[routed] = final_jev[routed]
    router_result = pipeline.comparison(router, baseline, y)
    bounds = {
        "third_expert_blend": {
            "observed_correct": known["third_expert_blend"]["correct"],
            "remaining_baseline_correct_granted_preserved": int(baseline_correct_remaining.sum()),
            "remaining_baseline_errors": int(baseline_wrong_remaining.sum()),
            "remaining_errors_potentially_fixable_with_any_jev_distribution": int(potentially_fixable.sum()),
            "remaining_errors_impossible_to_fix_at_weight_0_2": int(unfixable.sum()),
            "maximum_possible_full_development_correct": int(blend_upper),
            "can_reach_accuracy_gate": bool(blend_upper >= gate),
            "true_class_probability_gap_threshold": 0.25,
            "conservative_tie_tolerance": 1e-12,
            "optimistically_fixable_image_ids": [rows[i]["id"] for i in np.flatnonzero(potentially_fixable)],
            "proof": "For baseline b=(DINO+ConvNeXt)/2, blend=0.8b+0.2j. The best possible j assigns 1 to the true class and 0 elsewhere. A baseline error is fixable only when max_other(b)-b_true<=0.25. Count ties within 1e-12 as potentially fixable and grant that every remaining baseline-correct image stays correct. Add these optimistic counts to the already-observed blend correct predictions.",
        },
        "final_jev": {
            "observed_correct": known["final_jev"]["correct"],
            "remaining_rows_granted_all_correct": int(remaining.sum()),
            "maximum_possible_full_development_correct": int(direct_upper),
            "can_reach_accuracy_gate": bool(direct_upper >= gate),
        },
        "jev_image_only_diagnostic": {
            "observed_correct": known["jev_image_only_diagnostic"]["correct"],
            "remaining_rows_granted_all_correct": int(remaining.sum()),
            "maximum_possible_full_development_correct": int(independent_upper),
            "can_reach_accuracy_gate": bool(independent_upper >= gate),
            "eligible_for_deployment_under_frozen_protocol": False,
        },
    }
    stopped = (
        not bounds["third_expert_blend"]["can_reach_accuracy_gate"]
        and not bounds["final_jev"]["can_reach_accuracy_gate"]
        and not router_result["passes_frozen_exploratory_gate"]
    )
    return {
        "created_utc": pipeline.utcnow(),
        "status": "NO_FROZEN_CANDIDATE_CAN_PASS" if stopped else "AT_LEAST_ONE_CANDIDATE_NOT_EXCLUDED",
        "protocol_sha256": pipeline.digest(protocol),
        "pipeline_sha256": protocol["pipeline_sha256"],
        "feasibility_script_sha256": pipeline.sha256(__file__),
        "validated_outcome_hashes_sha256": pipeline.digest(outcome_hashes),
        "completed_two_stage_development_rows": int(available.sum()),
        "remaining_unqueried_development_rows": int(remaining.sum()),
        "completed_routed_rows": int(available[routed].sum()),
        "required_routed_rows": int(routed.sum()),
        "baseline_full_development": baseline_metrics,
        "gate_minimum_correct": gate,
        "gate_additional_requirement": "Candidate NLL must also be no worse than the baseline; an accuracy upper bound below the gate suffices to exclude a candidate regardless of future NLL.",
        "observed_cached_subset_comparisons": known,
        "complete_full_development_router": router_result,
        "optimistic_full_development_upper_bounds": bounds,
        "conclusion": "Further requests cannot make the frozen blend or direct final Jev meet the accuracy gate; the completed router also fails." if stopped else "Consult candidate-specific bounds; this receipt does not certify universal futility.",
        "interpretation": "Upper bounds are mathematical best cases for unobserved responses, not measured full-development scores. Observed subset scores cover completed rows only. The original development split was reused for earlier model selection; no unbiased test-set score is claimed.",
        "api_calls_made_by_this_script": 0,
        "credentials_read_by_this_script": False,
        "ledger_modified_by_this_script": False,
        "only_written_artifact": "feasibility_receipt.json",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(Path(__file__).parent / "prepared"))
    args = parser.parse_args()
    receipt = assess(args.run_dir)
    pipeline.write_json(Path(args.run_dir) / "feasibility_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
