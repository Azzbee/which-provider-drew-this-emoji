#!/usr/bin/env python3
"""Reproduce the selected emoji-provider CSV using packaged trained models.

Run from a fresh computer after installing the bundle's pinned dependencies:
    python reproduce_submission.py --data ./dataset --out ./reproduced --threads 8

The full dataset archive is required because the model scripts check the
original training labels, image IDs, and held-out split before inference.
This program does not train, refit, download weights, or upload to Kaggle.
AI assistance: OpenAI Codex authored this wrapper and the experiment pipelines.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CLASSES = ['apple', 'facebook', 'google', 'messenger', 'mozilla', 'samsung', 'whatsapp']
EXPECTED_IDS = [str(i) for i in range(10001, 19880)]
PRETRAINED_HASHES = {
    'dinov2': 'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9',
    'convnext': '983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d',
}


def emit(event, **fields):
    print(json.dumps({'event': event, **fields}, default=str), flush=True)


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(value, torch.Tensor):
            raise ValueError(f'Unexpected non-tensor in model state: {name}')
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_checkpoint(component, directory, pretrained, split, selection):
    """Verify immutable origin and the saved result's final pretrained tensors."""
    checkpoint = directory / 'run_224/selection/best.pt'
    metadata_file = directory / 'run_224/selection/metrics.json'
    require(checkpoint.is_file() and pretrained.is_file() and metadata_file.is_file(),
            f'{component}: missing packaged checkpoint, pretrained weights, or selection metrics')
    checkpoint_sha, pretrained_sha = sha(checkpoint), sha(pretrained)
    require(pretrained_sha == PRETRAINED_HASHES[component], f'{component}: official pretrained SHA256 mismatch')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    cfg = saved['configuration']
    require(saved['class_names'] == CLASSES, f'{component}: checkpoint class order mismatch')
    require(cfg['checkpoint_sha256'] == pretrained_sha, f'{component}: pretrained provenance mismatch')
    require(cfg['split_manifest_sha256'] == sha(split), f'{component}: split provenance mismatch')
    require(cfg['resolution'] == 224 and cfg['compute_precision'] == 'bf16',
            f'{component}: this bundle requires the selected 224px BF16 inference recipe')
    require(not cfg.get('all_label_refit', False) and cfg.get('mode', 'selection') != 'all_data_refit',
            f'{component}: unexpected all-label refit checkpoint')
    expected_model = ('vit_small_patch14_dinov2.lvd142m' if component == 'dinov2'
                      else 'torchvision_convnext_tiny_imagenet1k_v1')
    require(cfg['model'] == expected_model, f'{component}: model architecture mismatch')
    if component == 'dinov2':
        require(cfg['trainable_last_blocks'] == 2, 'DINOv2 must use the validated final-two-block model')
    model_state = saved['model_state_dict']
    whole_state_sha = state_hash(model_state)
    suffix_state_sha = state_hash({key: value for key, value in model_state.items() if not key.startswith('head.')})
    head_state_sha = state_hash({key[len('head.'):]: value for key, value in model_state.items() if key.startswith('head.')})
    metrics = read_json(metadata_file)
    require(metrics['selected_epoch'] == saved['epoch'], f'{component}: selected epoch differs from metrics')
    require(metrics['final_pretrained_suffix_sha256'] == suffix_state_sha,
            f'{component}: saved pretrained tensors differ from the validated model')
    require(metrics['true_backbone_fine_tuning'] is True, f'{component}: missing evidence of actual fine-tuning')
    if 'final_classifier_sha256' in metrics:
        require(metrics['final_classifier_sha256'] == head_state_sha, f'{component}: classifier tensors differ from metrics')
    expected_state = selection.get('model_state_sha256', {}).get(component)
    if expected_state:
        require(expected_state == whole_state_sha, f'{component}: selected full model tensor hash mismatch')
    packaging_file = directory / 'checkpoint_packaging.json'
    packaging_verified = False
    if packaging_file.exists():
        packaging = read_json(packaging_file)
        require(packaging['portable_inference_checkpoint_sha256'] == checkpoint_sha,
                f'{component}: packaged checkpoint file hash mismatch')
        require(packaging['all_model_tensors_exactly_equal'] is True,
                f'{component}: checkpoint packaging did not preserve every model tensor')
        if 'pretrained_checkpoint_sha256' in packaging:
            require(packaging['pretrained_checkpoint_sha256'] == pretrained_sha,
                    f'{component}: packaging pretrained hash mismatch')
        packaging_verified = True
    require(packaging_verified or bool(expected_state) or metrics.get('selected_checkpoint_sha256') == checkpoint_sha,
            f'{component}: include checkpoint_packaging.json or a selected model_state_sha256 to verify the complete checkpoint')
    result = {'component': component, 'checkpoint_sha256': checkpoint_sha,
              'model_state_sha256': whole_state_sha, 'pretrained_suffix_state_sha256': suffix_state_sha,
              'classifier_state_sha256': head_state_sha, 'pretrained_checkpoint_sha256': pretrained_sha,
              'split_manifest_sha256': sha(split), 'selected_epoch': saved['epoch'],
              'packaging_verified': packaging_verified, 'all_label_refit': False}
    del saved, model_state
    return result


def run_script(command):
    environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    subprocess.run(command, check=True, env=environment)


def validate_probabilities(path, ids, component):
    require(ids == EXPECTED_IDS and len(set(ids)) == 9879, f'{component}: test IDs/order mismatch')
    probabilities = np.load(path, allow_pickle=False)
    require(probabilities.shape == (9879, 7), f'{component}: wrong probability shape')
    require(probabilities.dtype == np.float32, f'{component}: expected original float32 probabilities')
    require(np.isfinite(probabilities).all(), f'{component}: nonfinite probabilities')
    require((probabilities >= 0).all() and (probabilities <= 1).all(), f'{component}: probabilities outside [0,1]')
    require(np.allclose(probabilities.sum(axis=1), 1., atol=1e-5), f'{component}: probability rows are not normalized')
    return probabilities


def reproduce_dino(args, package, selection):
    experiment = package / 'dino_original'
    split = experiment / 'run_224/split.csv'
    details = validate_checkpoint('dinov2', experiment, experiment / 'dinov2_vits14_pretrain.pth', split, selection)
    output = args.out / 'dino'
    emit('model_inference_started', model='dinov2', selected_epoch=details['selected_epoch'])
    run_script([sys.executable, str(package / 'dino_tta/evaluate_tta.py'),
                '--experiment', str(experiment), '--data', str(args.data), '--out', str(output),
                '--split', 'test', '--views', 'original', '--threads', str(args.threads)])
    receipt = read_json(output / 'test_receipt.json')
    require(receipt['class_names'] == CLASSES and receipt['split'] == 'test' and receipt['views'] == ['original'],
            'DINOv2 inference receipt classes, split, or views differ')
    require(receipt['model_checkpoint_sha256'] == details['checkpoint_sha256'], 'DINOv2 inference used a different checkpoint')
    require(receipt['pretrained_checkpoint_sha256'] == details['pretrained_checkpoint_sha256'], 'DINOv2 inference pretrained hash differs')
    require(receipt['split_manifest_sha256'] == details['split_manifest_sha256'], 'DINOv2 inference split hash differs')
    require(receipt['kaggle_uploaded'] is False, 'Unexpected Kaggle upload flag in DINOv2 receipt')
    probability_file = output / 'test_original_probabilities.npy'
    probabilities = validate_probabilities(probability_file, read_json(output / 'test_ids.json'), 'dinov2')
    details['probabilities_sha256'] = sha(probability_file)
    details['inference_receipt_sha256'] = sha(output / 'test_receipt.json')
    return probabilities, details


def reproduce_convnext(args, package, selection):
    experiment = package / 'convnext'
    split = package / 'dino_original/run_224/split.csv'
    pretrained = experiment / 'convnext_tiny-983f1562.pth'
    details = validate_checkpoint('convnext', experiment, pretrained, split, selection)
    output = args.out / 'convnext'
    emit('model_inference_started', model='convnext', selected_epoch=details['selected_epoch'])
    run_script([sys.executable, str(experiment / 'train_convnext.py'), '--stage', 'predict', '--method', 'finetune',
                '--data', str(args.data), '--split', str(split),
                '--hashes', str(package / 'dino_original/image_hashes.csv'), '--checkpoint', str(pretrained),
                '--model-file', str(experiment / 'run_224/selection/best.pt'),
                '--out', str(output), '--threads', str(args.threads)])
    receipt_path = output / 'submission_receipt_convnext_finetuned.json'
    receipt = read_json(receipt_path)
    require(receipt['class_names'] == CLASSES and receipt['rows'] == 9879, 'ConvNeXt receipt class order or row count differs')
    require(receipt['model_sha256'] == details['checkpoint_sha256'], 'ConvNeXt inference used a different checkpoint')
    require(receipt['configuration']['checkpoint_sha256'] == details['pretrained_checkpoint_sha256'],
            'ConvNeXt inference pretrained hash differs')
    require(receipt['configuration']['split_manifest_sha256'] == details['split_manifest_sha256'],
            'ConvNeXt inference split hash differs')
    require(receipt['final_pretrained_suffix_sha256'] == details['pretrained_suffix_state_sha256'],
            'ConvNeXt inference suffix tensor hash differs')
    require(receipt['all_label_refit'] is False and receipt['test_submission_performed'] is False,
            'Unexpected refit or upload flag in ConvNeXt receipt')
    csv_path = output / 'submission_convnext_finetuned.csv'
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    require(frame.columns.tolist() == ['Id', 'Label'], 'ConvNeXt intermediate CSV columns differ')
    probability_path = output / 'test_probabilities_convnext_finetuned.npy'
    probabilities = validate_probabilities(probability_path, frame.Id.tolist(), 'convnext')
    require(frame.Label.tolist() == [CLASSES[i] for i in probabilities.argmax(axis=1)],
            'ConvNeXt CSV labels differ from saved probabilities')
    require(sha(csv_path) == receipt['submission_sha256'], 'ConvNeXt intermediate CSV hash differs from its receipt')
    details['probabilities_sha256'] = sha(probability_path)
    details['inference_receipt_sha256'] = sha(receipt_path)
    return probabilities, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Extracted full competition dataset')
    parser.add_argument('--out', type=Path, default=Path('./reproduced'))
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    require(args.threads >= 1, '--threads must be positive')
    args.data, args.out = args.data.resolve(), args.out.resolve()
    package = Path(__file__).resolve().parent
    selection_path = package / 'selected_ensemble.json'
    selection = read_json(selection_path)
    require(selection['class_names'] == CLASSES, 'Selected ensemble class order differs')
    require(isinstance(selection['name'], str) and bool(selection['name'].strip()), 'Selection must have a model name')
    weight = selection['dinov2_weight']
    require(isinstance(weight, (int, float)) and not isinstance(weight, bool) and math.isfinite(weight) and 0 <= weight <= 1,
            'dinov2_weight must be a finite number between zero and one')
    require((args.data / 'train_labels.csv').is_file() and (args.data / 'train').is_dir(),
            'Provide the complete dataset archive, including training labels and training images')
    require(sorted(p.stem for p in (args.data / 'test').glob('*.png')) == EXPECTED_IDS,
            'Dataset must contain precisely test images 10001 through 19879')
    args.out.mkdir(parents=True, exist_ok=True)
    details, dino, convnext = [], None, None
    if weight > 0:
        dino, report = reproduce_dino(args, package, selection)
        details.append(report)
    if weight < 1:
        convnext, report = reproduce_convnext(args, package, selection)
        details.append(report)
    if weight == 1:
        probabilities = dino.copy()
    elif weight == 0:
        probabilities = convnext.copy()
    else:
        # Preserve the exact float32 arithmetic used for fixed-weight model comparison.
        probabilities = np.float32(weight) * dino + np.float32(1. - weight) * convnext
    require(probabilities.dtype == np.float32 and np.isfinite(probabilities).all(), 'Invalid final probability array')
    require(np.allclose(probabilities.sum(axis=1), 1., atol=1e-5), 'Final probability rows are not normalized')
    frame = pd.DataFrame({'Id': EXPECTED_IDS, 'Label': [CLASSES[i] for i in probabilities.argmax(axis=1)]})
    csv_path, probability_path = args.out / 'submission.csv', args.out / 'test_probabilities.npy'
    frame.to_csv(csv_path, index=False)
    np.save(probability_path, probabilities)
    pd.testing.assert_frame_equal(frame, pd.read_csv(csv_path, dtype=str, keep_default_na=False))
    receipt = {'name': selection['name'], 'dinov2_weight': weight, 'convnext_weight': 1. - weight,
               'probability_rule': 'Float32 weighted probability average; no extra normalization or test-time augmentation',
               'class_names': CLASSES, 'rows': len(frame), 'first_id': EXPECTED_IDS[0], 'last_id': EXPECTED_IDS[-1],
               'selection_file_sha256': sha(selection_path), 'submission_sha256': sha(csv_path),
               'probabilities_sha256': sha(probability_path), 'counts': frame.Label.value_counts().to_dict(),
               'component_checks': details, 'training_performed': False, 'all_label_refit': False,
               'kaggle_uploaded': False, 'python_version': sys.version, 'torch_version': str(torch.__version__),
               'numpy_version': np.__version__, 'threads': args.threads}
    if selection.get('reference_submission_sha256'):
        receipt['matches_reference_submission_sha256'] = receipt['submission_sha256'] == selection['reference_submission_sha256']
    (args.out / 'reproduction_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    emit('reproduction_finished', submission=str(csv_path), submission_sha256=receipt['submission_sha256'],
         rows=len(frame), name=selection['name'], dinov2_weight=weight)


if __name__ == '__main__':
    main()
