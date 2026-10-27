#!/usr/bin/env python3
"""Independent, paired validation comparison for the fixed emoji-provider split.

No training or model inference. Intervals and p-values are exploratory because
the holdout has already been used for model selection. The bootstrap samples
whole exact-image groups from the supplied image_hashes.csv, not separate rows.
AI assistance: OpenAI Codex wrote this audit at the user's request to explore
improvements; retain this source and its generated receipts with the codebase.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


CLASSES = ['apple', 'facebook', 'google', 'messenger', 'mozilla', 'samsung', 'whatsapp']
EXPECTED_SPLIT_SHA256 = '32461c8bab69a90b659e5d8903cd8d43f8a5bf2262062ba02776333537889e5e'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def check_probabilities(probabilities, rows, name):
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (rows, len(CLASSES)):
        raise ValueError(f'{name}: expected {(rows, len(CLASSES))}, got {probabilities.shape}')
    if not np.issubdtype(probabilities.dtype, np.floating):
        raise ValueError(f'{name}: expected floating-point probabilities')
    if not np.isfinite(probabilities).all():
        raise ValueError(f'{name}: nonfinite probability')
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError(f'{name}: probability outside [0, 1]')
    if not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-5):
        raise ValueError(f'{name}: rows do not sum to one')
    return probabilities.astype(np.float64, copy=False)


def mcnemar_exact_two_sided(baseline_wrong_candidate_correct,
                            baseline_correct_candidate_wrong):
    """Exact binomial test of discordant pairs, with stable log arithmetic."""
    b = int(baseline_wrong_candidate_correct)
    c = int(baseline_correct_candidate_wrong)
    if min(b, c) < 0:
        raise ValueError('Discordant counts must be nonnegative')
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            - n * math.log(2) for i in range(k + 1)]
    maximum = max(logs)
    tail = math.exp(maximum) * math.fsum(math.exp(x - maximum) for x in logs)
    return min(1.0, 2 * tail)


def grouped_bootstrap_delta(y, baseline, candidate, groups, replicates=10000, seed=20261017):
    """Percentile interval for image-weighted accuracy delta, resampling groups.

    Each replicate samples G groups with replacement. All images in each drawn
    group are retained, so the denominator can vary with duplicated group sizes.
    A chunked implementation avoids a large replicates-by-groups allocation.
    """
    if replicates < 100:
        raise ValueError('Use at least 100 bootstrap replicates')
    y, baseline, candidate, groups = map(np.asarray, (y, baseline, candidate, groups))
    if y.ndim != 1 or not (y.shape == baseline.shape == candidate.shape == groups.shape):
        raise ValueError('Bootstrap inputs must have identical one-dimensional shapes')
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    if len(unique_groups) == 0:
        raise ValueError('Cannot bootstrap an empty dataset')
    change = (candidate == y).astype(np.int64) - (baseline == y).astype(np.int64)
    group_change = np.bincount(inverse, weights=change, minlength=len(unique_groups))
    group_size = np.bincount(inverse, minlength=len(unique_groups))
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 128):
        stop = min(start + 128, replicates)
        sampled = rng.integers(0, len(unique_groups), size=(stop - start, len(unique_groups)))
        estimates[start:stop] = group_change[sampled].sum(axis=1) / group_size[sampled].sum(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        'method': 'Percentile paired cluster bootstrap; resample whole RGB96 pixel-hash groups',
        'statistic': 'Image-weighted candidate accuracy minus reference accuracy',
        'replicates': int(replicates),
        'seed': int(seed),
        'unique_validation_groups': int(len(unique_groups)),
        'validation_images': int(len(y)),
        'interval_95_fraction': [float(low), float(high)],
        'interval_95_percentage_points': [float(100 * low), float(100 * high)],
        'exploratory_after_model_selection': True,
        'limitation': 'Accounts for exact RGB96 image grouping only; does not correct repeated holdout selection, near-duplicate dependence, or other distribution shifts.',
    }


def probability_metrics(y, probabilities):
    prediction = probabilities.argmax(axis=1)
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    np.add.at(confusion, (y, prediction), 1)
    true_count = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    denominator = true_count + predicted_count
    f1 = np.divide(2 * true_positive, denominator,
                   out=np.zeros(len(CLASSES), dtype=float), where=denominator != 0)
    recall = np.divide(true_positive, true_count,
                       out=np.zeros(len(CLASSES), dtype=float), where=true_count != 0)
    return {
        'correct': int((prediction == y).sum()),
        'total': int(len(y)),
        'accuracy': float((prediction == y).mean()),
        'macro_f1': float(f1.mean()),
        'log_loss_clip_1e_minus_12': float(-np.log(np.maximum(probabilities[np.arange(len(y)), y], 1e-12)).mean()),
        'confusion_matrix_rows_true_columns_predicted': confusion.tolist(),
        'per_class': {name: {'support': int(true_count[i]), 'correct': int(true_positive[i]),
                              'recall': float(recall[i]), 'f1': float(f1[i])}
                      for i, name in enumerate(CLASSES)},
    }


def load_reference(experiment):
    experiment = Path(experiment)
    split_path = experiment / 'run_224/split.csv'
    if sha(split_path) != EXPECTED_SPLIT_SHA256:
        raise ValueError('This is not the exact predeclared validation split')
    split = pd.read_csv(split_path, dtype=str, keep_default_na=False)
    if (split.columns.tolist() != ['Id', 'Label', 'split'] or len(split) != 9879
            or not split.Id.is_unique or split.Id.tolist() != sorted(split.Id)):
        raise ValueError('Invalid or unsorted split manifest')
    if split['split'].value_counts().to_dict() != {'train': 7903, 'validation': 1976}:
        raise ValueError('Unexpected train/validation counts')
    if sorted(split.Label.unique()) != CLASSES:
        raise ValueError('Unexpected provider classes')
    hashes_path = experiment / 'image_hashes.csv'
    hashes = pd.read_csv(hashes_path, dtype=str, keep_default_na=False)
    hashes = hashes.loc[hashes['split'].eq('train')].set_index('Id')
    if not hashes.index.is_unique or set(hashes.index) != set(split.Id):
        raise ValueError('Hash manifest must cover every labeled image exactly once')
    hashes = hashes.loc[split.Id]
    if hashes.Label.tolist() != split.Label.tolist():
        raise ValueError('Hash-manifest labels disagree with the fixed split')
    if not hashes.pixel_hash.str.fullmatch(r'[0-9a-f]{64}').all():
        raise ValueError('Invalid pixel hashes')
    group_frame = pd.DataFrame({'group': hashes.pixel_hash.to_numpy(),
                               'partition': split['split'].to_numpy(),
                               'label': split.Label.to_numpy()})
    if (group_frame.groupby('group').partition.nunique() != 1).any():
        raise ValueError('Exact pixel group crosses train/validation')
    if (group_frame.groupby('group').label.nunique() != 1).any():
        raise ValueError('Exact pixel group has contradictory labels')
    validation = split.loc[split['split'].eq('validation')]
    ids = validation.Id.tolist()
    y = validation.Label.map({name: i for i, name in enumerate(CLASSES)}).to_numpy(dtype=np.int64)
    baseline_path = experiment / 'run_224/selection/validation_predictions.csv'
    baseline_frame = pd.read_csv(baseline_path, dtype={'Id': str})
    if baseline_frame.Id.tolist() != ids or baseline_frame.true_label.tolist() != validation.Label.tolist():
        raise ValueError('Reference predictions differ from the fixed validation IDs or labels')
    baseline = check_probabilities(baseline_frame[[f'p_{name}' for name in CLASSES]].to_numpy(), len(ids), 'reference')
    predicted_names = [CLASSES[i] for i in baseline.argmax(axis=1)]
    if baseline_frame.predicted_label.tolist() != predicted_names:
        raise ValueError('Reference labels differ from probability argmax')
    if int((baseline.argmax(axis=1) == y).sum()) != 1895:
        raise ValueError('Reference is not the archived 1,895/1,976 result')
    provenance = {
        'split_sha256': sha(split_path),
        'image_hash_manifest_sha256': sha(hashes_path),
        'baseline_predictions_sha256': sha(baseline_path),
        'train_validation_exact_pixel_group_overlap': 0,
        'conflicting_label_pixel_groups': 0,
        'validation_id_order_sha256': hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
    }
    return ids, y, baseline, hashes.loc[ids, 'pixel_hash'].to_numpy(), provenance


def compare_candidate(experiment, candidate_probabilities, candidate_ids, candidate_classes,
                      name='candidate', replicates=10000, seed=20261017):
    ids, y, baseline, groups, provenance = load_reference(experiment)
    if list(candidate_ids) != ids:
        raise ValueError('Candidate IDs must exactly match fixed validation IDs and order; no silent sorting')
    if len(candidate_classes) != len(CLASSES) or set(candidate_classes) != set(CLASSES):
        raise ValueError('Candidate class order must contain the seven provider names exactly once')
    candidate = check_probabilities(candidate_probabilities, len(ids), name)
    candidate = candidate[:, [list(candidate_classes).index(c) for c in CLASSES]]
    baseline_prediction = baseline.argmax(axis=1)
    candidate_prediction = candidate.argmax(axis=1)
    baseline_correct, candidate_correct = baseline_prediction == y, candidate_prediction == y
    improved = int((~baseline_correct & candidate_correct).sum())
    regressed = int((baseline_correct & ~candidate_correct).sum())
    return {
        'candidate_name': name,
        'class_order_used_for_metrics': CLASSES,
        'candidate_supplied_class_order': list(candidate_classes),
        'provenance': provenance,
        'reference': probability_metrics(y, baseline),
        'candidate': probability_metrics(y, candidate),
        'paired': {
            'both_correct': int((baseline_correct & candidate_correct).sum()),
            'both_wrong': int((~baseline_correct & ~candidate_correct).sum()),
            'baseline_wrong_candidate_correct': improved,
            'baseline_correct_candidate_wrong': regressed,
            'net_additional_correct': improved - regressed,
            'accuracy_delta_percentage_points': 100 * (improved - regressed) / len(y),
            'predicted_class_changes': int((baseline_prediction != candidate_prediction).sum()),
            'mcnemar_exact_two_sided_image_level_p': mcnemar_exact_two_sided(improved, regressed),
            'mcnemar_caveat': 'Descriptive only: image-level calculation ignores repeated-image dependence and is uncorrected for multiple candidate selection; use grouped bootstrap for the paired interval.',
        },
        'group_bootstrap': grouped_bootstrap_delta(y, baseline_prediction, candidate_prediction, groups, replicates, seed),
        'interpretation': 'Exploratory comparison on an already reused model-selection holdout. This is not an unbiased estimate of the selected winner and says nothing measured about hidden Kaggle test accuracy.',
        'kaggle_submitted': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--candidate-probabilities', type=Path, required=True)
    parser.add_argument('--candidate-ids', type=Path, required=True, help='JSON array of ordered, zero-padded string validation IDs')
    parser.add_argument('--class-order', nargs=7, required=True, help='Explicit column order in candidate probabilities')
    parser.add_argument('--name', required=True)
    parser.add_argument('--replicates', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20261017)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    probabilities = np.load(args.candidate_probabilities, allow_pickle=False)
    ids = json.loads(args.candidate_ids.read_text())
    result = compare_candidate(args.experiment, probabilities, ids, args.class_order,
                               args.name, args.replicates, args.seed)
    result['provenance'].update({
        'candidate_probabilities_sha256': sha(args.candidate_probabilities),
        'candidate_ids_sha256': sha(args.candidate_ids),
        'comparison_script_sha256': sha(__file__),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'name': args.name, 'reference_correct': result['reference']['correct'],
                      'candidate_correct': result['candidate']['correct'], 'paired': result['paired'],
                      'bootstrap_95_percentage_points': result['group_bootstrap']['interval_95_percentage_points'],
                      'output': str(args.out)}), flush=True)


if __name__ == '__main__':
    main()
