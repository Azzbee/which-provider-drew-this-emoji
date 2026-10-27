#!/usr/bin/env python3
"""Complementary ConvNeXt-Tiny emoji-provider experiment.

AI assistance: OpenAI Codex wrote this pipeline at the user's request to try
additional models after exhausting the team's daily Kaggle submission quota.
It reuses the earlier DINOv2 split and identical image preprocessing. No hidden
test labels, Kaggle submissions, label lookups, or all-label refitting are used.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import convnext_tiny
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

CLASSES = ['apple', 'facebook', 'google', 'messenger', 'mozilla', 'samsung', 'whatsapp']
MODEL_NAME = 'torchvision_convnext_tiny_imagenet1k_v1'
SOURCE_URL = 'https://download.pytorch.org/models/convnext_tiny-983f1562.pth'
SOURCE_SHA256 = '983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d'
DEFAULT_ROOT = Path('/workspace/scratch/9c7afe2f05a2/tmp')
EXPERIMENT = DEFAULT_ROOT / 'cv_next_recovered/experiment/DINOv2_Emoji_Experiment'


def emit(event, **fields):
    print(json.dumps({'event': event, **fields}, default=str), flush=True)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def json_write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, default=str) + '\n')
    os.replace(temporary, path)


def torch_write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(data, temporary)
    os.replace(temporary, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def autocast(args):
    return torch.autocast('cpu', dtype=torch.bfloat16, enabled=args.precision == 'bf16')


def prepare_data(args):
    labels = pd.read_csv(args.data / 'train_labels.csv', dtype=str, keep_default_na=False)
    if labels.columns.tolist() != ['Id', 'Label']:
        raise ValueError('Expected exactly Id,Label in train_labels.csv')
    labels = labels.sort_values('Id').reset_index(drop=True)
    ids = sorted(p.stem for p in (args.data / 'train').glob('*.png'))
    test_ids = sorted(p.stem for p in (args.data / 'test').glob('*.png'))
    if len(labels) != 9879 or not labels.Id.is_unique or labels.Id.tolist() != ids:
        raise ValueError('Training image count, ID uniqueness, or label alignment differs')
    if sorted(labels.Label.unique()) != CLASSES or test_ids != [str(i) for i in range(10001, 19880)]:
        raise ValueError('Unexpected provider classes or test IDs')
    if set(ids) & set(test_ids):
        raise ValueError('Train and test IDs overlap')
    split = pd.read_csv(args.split, dtype=str).sort_values('Id').reset_index(drop=True)
    pd.testing.assert_frame_equal(labels, split[['Id', 'Label']])
    tr = np.flatnonzero(split['split'].eq('train'))
    va = np.flatnonzero(split['split'].eq('validation'))
    if len(tr) != 7903 or len(va) != 1976 or set(split['split']) != {'train', 'validation'}:
        raise ValueError('Must preserve the exact 7903/1976 DINOv2 split')
    hashes = pd.read_csv(args.hashes, dtype=str, keep_default_na=False)
    hashes = hashes.loc[hashes['split'].eq('train')].set_index('Id')
    if not hashes.index.is_unique or set(hashes.index) != set(ids):
        raise ValueError('Hash manifest must cover each labeled image exactly once')
    groups = hashes.loc[ids, 'pixel_hash'].to_numpy()
    if set(groups[tr]) & set(groups[va]):
        raise ValueError('An exact RGB96 image group crosses the held-out split')
    for indices in (tr, va):
        if set(labels.iloc[indices].Label) != set(CLASSES):
            raise ValueError('Each split must contain all classes')
    split_path = args.out / 'split.csv'
    if split_path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(split_path, dtype=str), split)
    else:
        split.to_csv(split_path, index=False)
    audit = {'train_count': len(tr), 'validation_count': len(va), 'test_count': len(test_ids),
             'class_names': CLASSES, 'split_sha256': sha(split_path),
             'labels_sha256': sha(args.data / 'train_labels.csv'),
             'image_hash_manifest_sha256': sha(args.hashes),
             'source_split': str(args.split), 'same_holdout_as_dinov2': True,
             'test_label_usage': 'None', 'all_label_refit': False}
    json_write(args.out / 'data_audit.json', audit)
    emit('data_validated', **audit)
    return labels, test_ids, tr, va, groups


class ImageDataset(Dataset):
    # Identical preprocessing and byte-hash definition to the DINOv2 experiment.
    def __init__(self, data, split, ids, resolution=224, indices=None):
        self.root, self.ids, self.resolution = data / split, ids, resolution
        self.indices = np.arange(len(ids)) if indices is None else np.asarray(indices)
        self.mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index = int(self.indices[index])
        with Image.open(self.root / (self.ids[index] + '.png')) as original:
            rgba = original.convert('RGBA')
            rgb = Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert('RGB')
            fitted = ImageOps.contain(rgb, (self.resolution, self.resolution), Image.Resampling.LANCZOS)
            square = Image.new('RGB', (self.resolution, self.resolution), 'white')
            square.paste(fitted, ((self.resolution - fitted.width) // 2,
                                  (self.resolution - fitted.height) // 2))
            pixels = np.array(square, dtype=np.uint8)
        digest = hashlib.sha256(pixels.tobytes()).hexdigest()
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).float().div(255.)
        return (tensor - self.mean) / self.std, index, digest


def load_backbone(args):
    if sha(args.checkpoint) != SOURCE_SHA256:
        raise ValueError('Pretrained checkpoint SHA256 differs from the verified official weights')
    model = convnext_tiny(weights=None, stochastic_depth_prob=0.1)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.eval().requires_grad_(False)
    return model


def prefix_forward(model, images):
    for layer in list(model.features)[:6]:
        images = layer(images)
    return images


class TailClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.tail = nn.Sequential(*copy.deepcopy(list(backbone.features)[6:]))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.norm = copy.deepcopy(backbone.classifier[0])
        self.head = nn.Linear(768, len(CLASSES))
        nn.init.trunc_normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
        self.requires_grad_(True)

    def features(self, tokens):
        return self.norm(self.avgpool(self.tail(tokens))).flatten(1)

    def forward(self, tokens):
        return self.head(self.features(tokens))


def signature(args, ids, split):
    return {'format_version': 1, 'model': MODEL_NAME, 'checkpoint_sha256': SOURCE_SHA256,
            'split': split, 'ids_sha256': hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
            'split_manifest_sha256': sha(args.out / 'split.csv'),
            'image_hash_manifest_sha256': sha(args.hashes), 'resolution': 224,
            'frozen_prefix': 'torchvision features[0:6], always eval',
            'compute_precision': args.precision, 'storage_dtype': 'float16',
            'shape': [len(ids), 384, 14, 14],
            'preprocessing': 'RGBA on white; Lanczos contain; centered square; ImageNet mean/std',
            'versions': {p: importlib.metadata.version(p) for p in ['torch', 'torchvision', 'Pillow']}}


def cache_images(args, backbone, ids, split):
    directory = args.out / 'cache'
    directory.mkdir(exist_ok=True)
    metadata = signature(args, ids, split)
    paths = [directory / f'{split}_{suffix}' for suffix in
             ['metadata.json', 'tokens.npy', 'complete.npy', 'pixel_sha256.npy']]
    exists = [p.exists() for p in paths]
    if any(exists) and not all(exists):
        raise RuntimeError('Partial cache files; move the incomplete cache aside before retrying')
    if all(exists):
        if json.loads(paths[0].read_text()) != metadata:
            raise ValueError('Cache provenance differs; use a fresh --out directory')
        tokens, done, hashes = [np.load(p, mmap_mode='r+') for p in paths[1:]]
        if list(tokens.shape) != metadata['shape'] or tokens.dtype != np.float16 or done.shape != (len(ids),):
            raise ValueError('Unexpected cache shape or dtype')
    else:
        tokens = np.lib.format.open_memmap(paths[1], mode='w+', dtype=np.float16, shape=tuple(metadata['shape']))
        done = np.lib.format.open_memmap(paths[2], mode='w+', dtype=bool, shape=(len(ids),))
        hashes = np.lib.format.open_memmap(paths[3], mode='w+', dtype='S64', shape=(len(ids),))
        done[:] = False
        done.flush()
        json_write(paths[0], metadata)
        pd.DataFrame({'Id': ids}).to_csv(directory / f'{split}_ids.csv', index=False)
    pending = np.flatnonzero(~done)
    if not len(pending):
        emit('cache_already_complete', split=split, count=len(ids))
        return
    loader = DataLoader(ImageDataset(args.data, split, ids, indices=pending),
                        batch_size=args.cache_batch_size, shuffle=False, num_workers=args.workers)
    started, previous, new_count = time.perf_counter(), time.perf_counter(), 0
    backbone.eval()
    with torch.inference_mode():
        for images, indices, digests in loader:
            with autocast(args):
                output = prefix_forward(backbone, images)
            values = output.float().numpy().astype(np.float16)
            if not np.isfinite(values).all():
                raise FloatingPointError('Nonfinite frozen-prefix features')
            ix = indices.numpy()
            tokens[ix], hashes[ix] = values, np.asarray(digests, dtype='S64')
            tokens.flush()
            hashes.flush()
            done[ix] = True
            done.flush()
            new_count += len(ix)
            now = time.perf_counter()
            if now - previous >= 15 or new_count == len(pending):
                emit('cache_progress', split=split, completed=int(done.sum()), total=len(ids),
                     images_per_second=round(new_count / (now - started), 2),
                     remaining_seconds=round((len(pending) - new_count) * (now - started) / new_count, 1))
                previous = now
    emit('cache_finished', split=split, seconds=round(time.perf_counter() - started, 2))


def open_cache(args, ids, split):
    directory = args.out / 'cache'
    if json.loads((directory / f'{split}_metadata.json').read_text()) != signature(args, ids, split):
        raise ValueError('Cache provenance mismatch')
    if not bool(np.load(directory / f'{split}_complete.npy', mmap_mode='r').all()):
        raise ValueError('Prefix cache is not complete')
    tokens = np.load(directory / f'{split}_tokens.npy', mmap_mode='r')
    if tokens.shape != (len(ids), 384, 14, 14) or tokens.dtype != np.float16:
        raise ValueError('Invalid cached feature shape or dtype')
    return tokens


def check_pixel_split(args, tr, va):
    hashes = np.load(args.out / 'cache/train_pixel_sha256.npy', mmap_mode='r')
    overlap = set(hashes[tr].tolist()) & set(hashes[va].tolist())
    if overlap:
        raise ValueError('Exact 224px input duplicates cross the train/validation split')
    json_write(args.out / 'actual224_split_check.json',
               {'cross_split_exact_224_pixel_groups': len(overlap), 'passed': True})


class TokenDataset(Dataset):
    def __init__(self, cache, indices, targets=None):
        self.cache, self.indices, self.targets = cache, np.asarray(indices), targets

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])
        return (torch.from_numpy(np.array(self.cache[index], dtype=np.float32)),
                int(self.targets[index]) if self.targets is not None else -1, index)


def token_loader(args, cache, indices, targets=None, shuffle=False, epoch=0):
    return DataLoader(TokenDataset(cache, indices, targets), batch_size=args.batch_size,
                      shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(args.seed + epoch))


def metrics(probabilities, target):
    if (probabilities.shape != (len(target), len(CLASSES)) or not np.isfinite(probabilities).all()
            or not np.allclose(probabilities.sum(axis=1), 1., atol=1e-5)):
        raise ValueError('Invalid probability matrix')
    pred = probabilities.argmax(axis=1)
    return {'accuracy': float(np.mean(pred == target)), 'correct': int(np.sum(pred == target)),
            'total': len(target),
            'cross_entropy': float(-np.log(np.maximum(probabilities[np.arange(len(target)), target], 1e-12)).mean())}


def predict_tail(args, model, cache, indices):
    model.eval()
    output, order = [], []
    with torch.inference_mode():
        for tokens, _, batch_indices in token_loader(args, cache, indices):
            with autocast(args):
                logits = model(tokens)
            output.append(logits.float().softmax(dim=1).numpy())
            order.extend(batch_indices.tolist())
    if order != list(map(int, indices)):
        raise ValueError('Prediction order changed')
    probabilities = np.concatenate(output)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1, atol=1e-5):
        raise FloatingPointError('Invalid probabilities')
    return probabilities


def pretrained_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            if not name.startswith('head.')}


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def backbone_change(model, initial):
    current = pretrained_state(model)
    squared, count, maximum, changed = 0., 0, 0., 0
    for name, before in initial.items():
        delta = current[name].double() - before.double()
        squared += float(delta.square().sum())
        count += delta.numel()
        maximum = max(maximum, float(delta.abs().max()))
        changed += int(bool(delta.ne(0).any()))
    return {'initial_pretrained_suffix_sha256': state_hash(initial),
            'final_pretrained_suffix_sha256': state_hash(current),
            'pretrained_tensors_changed': changed, 'pretrained_parameter_count': count,
            'pretrained_parameter_delta_l2': math.sqrt(squared),
            'pretrained_parameter_delta_rms': math.sqrt(squared / count),
            'pretrained_parameter_delta_max_abs': maximum,
            'true_backbone_fine_tuning': changed > 0 and squared > 0}


def configuration(args, method):
    return {'method': method, 'model': MODEL_NAME, 'resolution': 224, 'class_names': CLASSES,
            'checkpoint_sha256': SOURCE_SHA256, 'source_url': SOURCE_URL,
            'split_manifest_sha256': sha(args.out / 'split.csv'),
            'compute_precision': args.precision, 'cache_dtype': 'float16', 'batch_size': args.batch_size,
            'seed': args.seed, 'preprocessing': 'White RGBA composite, Lanczos letterbox 224, ImageNet normalization',
            'image_augmentation': 'None', 'all_label_refit': False,
            'frozen_prefix': 'TorchVision features[0:6], eval mode',
            'trainable_suffix': 'features[6:8] plus original final LayerNorm and new linear 768-to-7 head',
            'stochastic_depth_schedule_max': 0.1,
            'stochastic_depth_note': 'TorchVision default per-block schedule; active only for the final stage during fine-tuning',
            'versions': {p: importlib.metadata.version(p) for p in
                         ['torch', 'torchvision', 'numpy', 'pandas', 'scikit-learn', 'Pillow']},
            'python_version': sys.version,
            'ai_usage': 'OpenAI Codex wrote this code; retain the conversation and prompt log for course transparency.'}


def save_validation(args, directory, labels, va, y, probabilities, summary):
    prediction = probabilities.argmax(axis=1)
    frame = pd.DataFrame({'Id': labels.iloc[va].Id.to_numpy(),
                          'true_label': labels.iloc[va].Label.to_numpy(),
                          'predicted_label': [CLASSES[i] for i in prediction]})
    for i, name in enumerate(CLASSES):
        frame[f'p_{name}'] = probabilities[:, i]
    frame.to_csv(directory / 'validation_predictions.csv', index=False)
    np.save(directory / 'validation_probabilities.npy', probabilities)
    summary.update(metrics(probabilities, y[va]))
    summary['confusion_matrix'] = confusion_matrix(y[va], prediction, labels=range(7)).tolist()
    summary['classification_report'] = classification_report(
        y[va], prediction, labels=range(7), target_names=CLASSES, output_dict=True, zero_division=0)
    summary['complexity_decision'] = {
        'reference': 'Earlier DINOv2 selected model on this exact held-out split',
        'reference_correct': args.reference_correct, 'minimum_additional_correct': args.min_correct_gain,
        'correct_gain': summary['correct'] - args.reference_correct,
        'passes_required_gain': summary['correct'] >= args.reference_correct + args.min_correct_gain,
        'note': 'A local model-choice threshold, not a Kaggle score or a statistical significance test.'}
    json_write(directory / 'metrics.json', summary)
    emit('validation_saved', directory=str(directory), accuracy=summary['accuracy'], correct=summary['correct'],
         total=summary['total'], complexity_decision=summary['complexity_decision'])


def fit_tail(args, backbone, labels, tr, va):
    cache = open_cache(args, labels.Id.tolist(), 'train')
    check_pixel_split(args, tr, va)
    y = labels.Label.map({c: i for i, c in enumerate(CLASSES)}).to_numpy(dtype=np.int64)
    seed_all(args.seed)
    model = TailClassifier(backbone)
    initial = pretrained_state(model)
    initial_head = {name: value.detach().clone() for name, value in model.head.state_dict().items()}
    initial_hash = state_hash(initial)
    directory = args.out / 'selection'
    directory.mkdir(exist_ok=True)
    config = {**configuration(args, 'finetune'), 'epochs_budget': args.epochs, 'patience': args.patience,
              'backbone_lr': args.tail_lr, 'classifier_lr': args.head_lr, 'weight_decay': args.weight_decay,
              'optimizer': 'AdamW', 'gradient_clip_norm': 1.0, 'label_smoothing': 0.,
              'lr_schedule': 'Epoch cosine multiplier from 1.0 toward 0.1 over the original epoch budget',
              'selection_rule': 'Highest held-out accuracy, then lowest cross-entropy'}
    config_path = directory / 'configuration.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Training settings changed; use a new --out directory')
    json_write(config_path, config)
    torch_write(directory / 'initial_pretrained_suffix.pt', initial)
    torch_write(directory / 'initial_classifier.pt', initial_head)
    pretrained_parameters = list(model.tail.parameters()) + list(model.norm.parameters())
    optimizer = torch.optim.AdamW([
        {'params': pretrained_parameters, 'lr': args.tail_lr},
        {'params': model.head.parameters(), 'lr': args.head_lr}], weight_decay=args.weight_decay)
    history, best_score, best_epoch, stale, starting_epoch = [], (-float('inf'), -float('inf')), 0, 0, 1
    gradient_seen = 0.
    last_path, best_path = directory / 'last.pt', directory / 'best.pt'
    if last_path.exists():
        saved = torch.load(last_path, map_location='cpu', weights_only=True)
        if saved['configuration'] != config or saved['initial_pretrained_suffix_sha256'] != initial_hash:
            raise ValueError('Resume checkpoint provenance mismatch')
        model.load_state_dict(saved['model_state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        history, best_score = saved['history'], tuple(saved['best_score'])
        best_epoch, stale, starting_epoch = saved['best_epoch'], saved['stale'], saved['epoch'] + 1
        gradient_seen = saved['backbone_gradient_norm_first_batch']
        torch.set_rng_state(saved['torch_rng_state'])
        emit('training_resumed', next_epoch=starting_epoch, best_epoch=best_epoch)
    emit('training_started', examples=len(tr), epochs=args.epochs,
         backbone_trainable_parameters=sum(p.numel() for p in pretrained_parameters),
         head_trainable_parameters=sum(p.numel() for p in model.head.parameters()))
    for epoch in range(starting_epoch, args.epochs + 1):
        if stale >= args.patience:
            break
        started = time.perf_counter()
        multiplier = 0.1 + 0.9 * 0.5 * (1. + math.cos(math.pi * (epoch - 1) / args.epochs))
        optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'] = args.tail_lr * multiplier, args.head_lr * multiplier
        model.train()
        count, correct, loss_sum = 0, 0, 0.
        for tokens, targets, _ in token_loader(args, cache, tr, y, shuffle=True, epoch=epoch):
            optimizer.zero_grad(set_to_none=True)
            with autocast(args):
                logits = model(tokens)
                loss = nn.functional.cross_entropy(logits, targets)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            if gradient_seen == 0.:
                gradient_seen = math.sqrt(sum(float(p.grad.float().square().sum())
                                              for p in pretrained_parameters if p.grad is not None))
                if not math.isfinite(gradient_seen) or gradient_seen <= 0.:
                    raise RuntimeError('No nonzero gradient reaches the pretrained suffix')
                emit('backbone_gradient_verified', gradient_l2=gradient_seen)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            count += len(targets)
            correct += int(logits.detach().argmax(dim=1).eq(targets).sum())
            loss_sum += float(loss.detach()) * len(targets)
        probabilities = predict_tail(args, model, cache, va)
        metric = metrics(probabilities, y[va])
        row = {'epoch': epoch, 'train_loss': loss_sum / count, 'train_accuracy': correct / count,
               'validation_accuracy': metric['accuracy'], 'validation_correct': metric['correct'],
               'validation_cross_entropy': metric['cross_entropy'],
               'backbone_lr': optimizer.param_groups[0]['lr'], 'head_lr': optimizer.param_groups[1]['lr'],
               'seconds': round(time.perf_counter() - started, 3)}
        history.append(row)
        score = (metric['accuracy'], -metric['cross_entropy'])
        improved = score > best_score
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
        saved = {'format_version': 1, 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch,
                 'configuration': config, 'history': history, 'best_score': list(best_score),
                 'best_epoch': best_epoch, 'stale': stale, 'class_names': CLASSES,
                 'initial_pretrained_suffix_sha256': initial_hash,
                 'initial_classifier_sha256': state_hash(initial_head),
                 'backbone_gradient_norm_first_batch': gradient_seen, 'torch_rng_state': torch.get_rng_state()}
        if improved:
            torch_write(best_path, saved)
        torch_write(last_path, saved)
        pd.DataFrame(history).to_csv(directory / 'history.csv', index=False)
        emit('epoch_complete', best_epoch=best_epoch, stale=stale, **row)
    selected = torch.load(best_path, map_location='cpu', weights_only=True)
    model.load_state_dict(selected['model_state_dict'], strict=True)
    change = backbone_change(model, initial)
    if not change['true_backbone_fine_tuning']:
        raise RuntimeError('Selected checkpoint did not update the pretrained backbone')
    summary = {'method': 'finetune', 'selected_epoch': selected['epoch'], 'epochs_completed': len(history),
               'configuration': config, 'backbone_gradient_norm_first_batch': gradient_seen, **change,
               'selected_checkpoint_sha256': sha(best_path), 'all_label_refit': False,
               'holdout_used_for_model_selection': True, 'test_submission_performed': False}
    save_validation(args, directory, labels, va, y, predict_tail(args, model, cache, va), summary)
    emit('training_finished', selected_epoch=selected['epoch'], accuracy=summary['accuracy'])


def frozen_features(args, backbone, ids, split):
    cache = open_cache(args, ids, split)
    directory = args.out / 'frozen_features'
    directory.mkdir(exist_ok=True)
    path, meta = directory / f'{split}.npy', directory / f'{split}_metadata.json'
    metadata = {**signature(args, ids, split), 'representation': 'Original pretrained final-stage pooled and normalized 768 features',
                'feature_storage': 'float32'}
    if path.exists() and meta.exists():
        if json.loads(meta.read_text()) != metadata:
            raise ValueError('Frozen pooled-feature provenance mismatch')
        features = np.load(path)
        if features.shape != (len(ids), 768) or not np.isfinite(features).all():
            raise ValueError('Invalid frozen pooled features')
        return features
    model = TailClassifier(backbone).eval()
    features = np.empty((len(ids), 768), dtype=np.float32)
    started = previous = time.perf_counter()
    completed = 0
    with torch.inference_mode():
        for tokens, _, indices in token_loader(args, cache, np.arange(len(ids))):
            with autocast(args):
                values = model.features(tokens)
            features[indices.numpy()] = values.float().numpy()
            completed += len(indices)
            now = time.perf_counter()
            if now - previous >= 15 or completed == len(ids):
                emit('frozen_features_progress', split=split, completed=completed, total=len(ids),
                     seconds=round(now - started, 2))
                previous = now
    if not np.isfinite(features).all():
        raise FloatingPointError('Nonfinite frozen pooled features')
    np.save(path, features)
    pd.DataFrame({'Id': ids}).to_csv(directory / f'{split}_ids.csv', index=False)
    json_write(meta, metadata)
    return features


def linear_probabilities(features, state):
    values = (features.astype(np.float64) - state['mean']) / state['scale']
    logits = values @ state['coef'].T + state['intercept']
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def union_pixel_groups(rgb96_groups, rgb224_groups):
    """Union either exact-pixel relation, including transitive duplicate chains."""
    parent = np.arange(len(rgb96_groups))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for column in (rgb96_groups, rgb224_groups):
        seen = {}
        for index, value in enumerate(column):
            if value in seen:
                first, second = find(index), find(seen[value])
                parent[max(first, second)] = min(first, second)
            else:
                seen[value] = index
    return np.asarray([find(index) for index in range(len(parent))])


def fit_logreg(args, backbone, labels, tr, va, groups):
    features = frozen_features(args, backbone, labels.Id.tolist(), 'train')
    check_pixel_split(args, tr, va)
    actual_hashes = np.load(args.out / 'cache/train_pixel_sha256.npy', mmap_mode='r')
    groups = union_pixel_groups(groups, actual_hashes)
    if set(groups[tr]) & set(groups[va]):
        raise ValueError('A unioned RGB96/RGB224 duplicate group crosses the held-out split')
    y = labels.Label.map({c: i for i, c in enumerate(CLASSES)}).to_numpy(dtype=np.int64)
    directory = args.out / 'logreg'
    directory.mkdir(exist_ok=True)
    config = {**configuration(args, 'frozen_standardized_multinomial_logistic_regression'),
              'trainable_suffix': 'None; entire ConvNeXt backbone frozen', 'C_grid': [0.1, 1., 10.],
              'C_selection': '3-fold StratifiedGroupKFold within training split only; accuracy, then log loss',
              'group_rule': 'Union of exact RGB96 groups from the supplied audit and actual RGB224 input SHA256 groups',
              'standardization': 'StandardScaler fit separately inside each inner training fold',
              'solver': 'lbfgs', 'max_iter': 3000, 'tolerance': 1e-4,
              'holdout_access': 'Only one evaluation after choosing C on inner cross-validation'}
    config_path = directory / 'configuration.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Frozen-classifier settings changed; use a new --out directory')
    json_write(config_path, config)
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=args.seed).split(features[tr], y[tr], groups[tr]))
    fold_assignment = np.full(len(tr), -1, dtype=int)
    for fold, (training, validation) in enumerate(folds):
        if set(groups[tr[training]]) & set(groups[tr[validation]]):
            raise ValueError('An exact-image group crosses an inner fold')
        fold_assignment[validation] = fold
    pd.DataFrame({'Id': labels.iloc[tr].Id.to_numpy(), 'group': groups[tr], 'fold': fold_assignment}).to_csv(directory / 'inner_folds.csv', index=False)
    rows = []
    best_score, best_c = (-float('inf'), -float('inf')), None
    with threadpool_limits(limits=args.threads):
        for c in config['C_grid']:
            oof = np.empty((len(tr), 7), dtype=np.float64)
            for fold, (training, validation) in enumerate(folds):
                scaler = StandardScaler().fit(features[tr[training]])
                classifier = LogisticRegression(C=c, max_iter=3000, tol=1e-4, solver='lbfgs', random_state=args.seed)
                classifier.fit(scaler.transform(features[tr[training]]), y[tr[training]])
                if classifier.classes_.tolist() != list(range(7)):
                    raise ValueError('Inner-fold classifier class order differs')
                if int(classifier.n_iter_.max()) >= 3000:
                    raise RuntimeError('Logistic regression failed to converge within 3000 iterations')
                probability = classifier.predict_proba(scaler.transform(features[tr[validation]]))
                oof[validation] = probability
                metric = metrics(probability, y[tr[validation]])
                emit('logreg_fold_complete', C=c, fold=fold, **metric)
            metric = metrics(oof, y[tr])
            rows.append({'C': c, **metric})
            score = (metric['accuracy'], -metric['cross_entropy'])
            if score > best_score:
                best_score, best_c = score, c
            emit('logreg_candidate_complete', C=c, **metric)
        scaler = StandardScaler().fit(features[tr])
        classifier = LogisticRegression(C=best_c, max_iter=3000, tol=1e-4, solver='lbfgs', random_state=args.seed)
        classifier.fit(scaler.transform(features[tr]), y[tr])
    if classifier.classes_.tolist() != list(range(7)) or int(classifier.n_iter_.max()) >= 3000:
        raise RuntimeError('Final logistic classifier class order or convergence failed')
    state = {'mean': scaler.mean_, 'scale': scaler.scale_, 'coef': classifier.coef_,
             'intercept': classifier.intercept_, 'classes': np.asarray(CLASSES)}
    np.savez_compressed(directory / 'model.npz', **state)
    probabilities = linear_probabilities(features[va], state)
    expected = classifier.predict_proba(scaler.transform(features[va]))
    if not np.allclose(probabilities, expected, rtol=1e-5, atol=1e-6):
        raise ValueError('Portable linear predictor differs from fitted scikit-learn predictor')
    pd.DataFrame(rows).to_csv(directory / 'C_selection.csv', index=False)
    summary = {'method': config['method'], 'configuration': config, 'chosen_C': best_c,
               'inner_cv': rows, 'model_sha256': sha(directory / 'model.npz'),
               'holdout_used_for_C_selection': False, 'all_label_refit': False,
               'test_submission_performed': False, 'true_backbone_fine_tuning': False}
    save_validation(args, directory, labels, va, y, probabilities, summary)


def predict_test(args, backbone, test_ids):
    cache_images(args, backbone, test_ids, 'test')
    if args.method == 'finetune':
        path = args.model_file or args.out / 'selection/best.pt'
        saved = torch.load(path, map_location='cpu', weights_only=True)
        config = saved['configuration']
        for key in ['model', 'resolution', 'checkpoint_sha256', 'split_manifest_sha256', 'compute_precision']:
            if config[key] != configuration(args, 'finetune')[key]:
                raise ValueError(f'Checkpoint prediction provenance mismatch: {key}')
        if saved['class_names'] != CLASSES or config['all_label_refit']:
            raise ValueError('Checkpoint classes or refit status differs')
        model = TailClassifier(backbone)
        initial = pretrained_state(model)
        model.load_state_dict(saved['model_state_dict'], strict=True)
        change = backbone_change(model, initial)
        if not change['true_backbone_fine_tuning']:
            raise ValueError('Fine-tuning checkpoint did not update pretrained parameters')
        probabilities = predict_tail(args, model, open_cache(args, test_ids, 'test'), np.arange(len(test_ids)))
        details = {'selected_epoch': saved['epoch'], **change}
    else:
        path = args.model_file or args.out / 'logreg/model.npz'
        config = json.loads((path.parent / 'configuration.json').read_text())
        for key in ['model', 'resolution', 'checkpoint_sha256', 'split_manifest_sha256', 'compute_precision']:
            if config[key] != configuration(args, 'logreg')[key]:
                raise ValueError(f'Logistic checkpoint prediction provenance mismatch: {key}')
        with np.load(path, allow_pickle=False) as state:
            if state['classes'].tolist() != CLASSES:
                raise ValueError('Logistic class order differs')
            probabilities = linear_probabilities(frozen_features(args, backbone, test_ids, 'test'), state)
        details = {'true_backbone_fine_tuning': False}
    if not np.isfinite(probabilities).all() or probabilities.shape != (9879, 7):
        raise ValueError('Invalid test probability matrix')
    if not np.allclose(probabilities.sum(axis=1), 1., atol=1e-5):
        raise ValueError('Test probabilities are not normalized')
    frame = pd.DataFrame({'Id': test_ids, 'Label': [CLASSES[i] for i in probabilities.argmax(axis=1)]})
    if (frame.columns.tolist() != ['Id', 'Label'] or not frame.Id.is_unique or frame.isna().any().any()
            or frame.Id.tolist() != [str(i) for i in range(10001, 19880)]):
        raise ValueError('Invalid submission format, IDs, or order')
    stem = 'convnext_finetuned' if args.method == 'finetune' else 'convnext_frozen_logreg'
    destination = args.out / f'submission_{stem}.csv'
    frame.to_csv(destination, index=False)
    pd.testing.assert_frame_equal(frame, pd.read_csv(destination, dtype=str))
    np.save(args.out / f'test_probabilities_{stem}.npy', probabilities)
    receipt = {'submission_file': str(destination), 'submission_sha256': sha(destination),
               'rows': len(frame), 'class_names': CLASSES, 'counts': frame.Label.value_counts().to_dict(),
               'model_file': str(path), 'model_sha256': sha(path), 'configuration': config,
               'all_label_refit': False, 'test_submission_performed': False, **details}
    json_write(args.out / f'submission_receipt_{stem}.json', receipt)
    emit('submission_saved', **receipt)


def benchmark(args, backbone, ids):
    sample = next(iter(DataLoader(ImageDataset(args.data, 'train', ids[:args.batch_size]),
                                 batch_size=args.batch_size)))[0]
    tail = TailClassifier(backbone).eval()
    original_precision, rows = args.precision, []
    for precision in ['fp32', 'bf16']:
        args.precision = precision
        with torch.inference_mode(), autocast(args):
            prefix = prefix_forward(backbone, sample)
            actual = tail.features(prefix)
            expected = backbone.classifier[0](backbone.avgpool(backbone.features(sample))).flatten(1)
            error = float((actual.float() - expected.float()).abs().max())
            if error > 1e-4:
                raise ValueError(f'Prefix/suffix feature split differs from full model: {error}')
            cached = prefix.to(torch.float16).float()
            cached_error = float((tail.features(cached).float() - expected.float()).abs().max())
            start = time.perf_counter()
            for _ in range(3):
                prefix_forward(backbone, sample)
            prefix_seconds = (time.perf_counter() - start) / 3
            start = time.perf_counter()
            for _ in range(3):
                tail.features(cached)
            tail_seconds = (time.perf_counter() - start) / 3
        rows.append({'precision': precision, 'batch_size': len(sample), 'prefix_seconds': prefix_seconds,
                     'tail_inference_seconds': tail_seconds, 'prefix_images_per_second': len(sample) / prefix_seconds,
                     'full_feature_max_abs_error': error, 'float16_cache_roundtrip_feature_max_abs_error': cached_error,
                     'train_prefix_estimate_seconds': 9879 * prefix_seconds / len(sample),
                     'cache_bytes_per_full_set': 9879 * 384 * 14 * 14 * 2})
    args.precision = original_precision
    json_write(args.out / 'benchmark.json', rows)
    emit('benchmark', results=rows)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['audit', 'benchmark', 'cache', 'features', 'logreg', 'train', 'predict', 'all'], default='audit')
    parser.add_argument('--data', type=Path, default=DEFAULT_ROOT / 'cv_next_recovered/dataset')
    parser.add_argument('--split', type=Path, default=EXPERIMENT / 'run_224/split.csv')
    parser.add_argument('--hashes', type=Path, default=EXPERIMENT / 'image_hashes.csv')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_ROOT / 'cv_next_convnext/convnext_tiny-983f1562.pth')
    parser.add_argument('--out', type=Path, default=DEFAULT_ROOT / 'cv_next_convnext/run_224')
    parser.add_argument('--cache-set', choices=['train', 'test', 'both'], default='train')
    parser.add_argument('--method', choices=['finetune', 'logreg'], default='finetune')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--cache-batch-size', type=int, default=16)
    parser.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--tail-lr', type=float, default=1e-4)
    parser.add_argument('--head-lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--reference-correct', type=int, default=1895)
    parser.add_argument('--min-correct-gain', type=int, default=8)
    parser.add_argument('--model-file', type=Path)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 10 or args.patience < 1 or min(args.batch_size, args.cache_batch_size, args.threads) < 1:
        parser.error('Epoch budget must be 1..10; patience, batch sizes, and threads must be positive')
    if args.tail_lr <= 0 or args.head_lr <= 0 or args.weight_decay < 0 or args.workers < 0:
        parser.error('Invalid learning rate, weight decay, or worker count')
    args.out.mkdir(parents=True, exist_ok=True)
    return args


def main():
    args = arguments()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    seed_all(args.seed)
    labels, test_ids, tr, va, groups = prepare_data(args)
    if args.stage == 'audit':
        return
    backbone = load_backbone(args)
    if args.stage == 'benchmark':
        benchmark(args, backbone, labels.Id.tolist())
        return
    if args.stage in ['cache', 'all']:
        if args.cache_set in ['train', 'both'] or args.stage == 'all':
            cache_images(args, backbone, labels.Id.tolist(), 'train')
        if args.cache_set in ['test', 'both']:
            cache_images(args, backbone, test_ids, 'test')
    if args.stage == 'features':
        if args.cache_set in ['train', 'both']:
            frozen_features(args, backbone, labels.Id.tolist(), 'train')
        if args.cache_set in ['test', 'both']:
            frozen_features(args, backbone, test_ids, 'test')
    if args.stage == 'logreg' or (args.stage == 'all' and args.method == 'logreg'):
        fit_logreg(args, backbone, labels, tr, va, groups)
    if args.stage == 'train' or (args.stage == 'all' and args.method == 'finetune'):
        fit_tail(args, backbone, labels, tr, va)
    if args.stage in ['predict', 'all']:
        predict_test(args, backbone, test_ids)


if __name__ == '__main__':
    main()
