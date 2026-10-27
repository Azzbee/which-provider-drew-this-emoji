#!/usr/bin/env python3
"""Fine-tune the final DINOv2 transformer blocks for emoji-provider classification.

Frozen-prefix caches reduce CPU cost; the pretrained suffix and classifier are
updated by backpropagation. No test labels or image-matching predictions are used.
AI assistance: OpenAI Codex generated this pipeline at the user's request to
fine-tune a Meta vision model and produce a competition CSV. Keep this script,
the run provenance, and the actual conversation prompt log with the codebase.
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
import timm
from timm.models.vision_transformer import checkpoint_filter_fn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CLASSES = ['apple', 'facebook', 'google', 'messenger', 'mozilla', 'samsung', 'whatsapp']
MODEL_NAME = 'vit_small_patch14_dinov2.lvd142m'
DEFAULT_ROOT = Path('/workspace/scratch/9c7afe2f05a2/tmp')


def emit(event, **fields):
    print(json.dumps({'event': event, **fields}, default=str), flush=True)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
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
        raise ValueError('train_labels.csv must contain exactly Id,Label')
    labels = labels.sort_values('Id').reset_index(drop=True)
    if len(labels) != 9879 or not labels.Id.is_unique or sorted(labels.Label.unique()) != CLASSES:
        raise ValueError('Unexpected training count, duplicated IDs, or provider classes')
    train_ids = sorted(p.stem for p in (args.data / 'train').glob('*.png'))
    test_ids = sorted(p.stem for p in (args.data / 'test').glob('*.png'))
    if labels.Id.tolist() != train_ids:
        raise ValueError('Training image IDs do not match the labels')
    if test_ids != [str(i) for i in range(10001, 19880)]:
        raise ValueError('Expected precisely test IDs 10001 through 19879')
    if set(test_ids) & set(train_ids):
        raise ValueError('Train/test IDs overlap')
    hashes = pd.read_csv(args.hashes, dtype=str, keep_default_na=False)
    hashes = hashes.loc[hashes['split'].eq('train')].set_index('Id')
    if not hashes.index.is_unique or set(hashes.index) != set(train_ids):
        raise ValueError('Hash manifest must cover each labelled image once')
    groups = hashes.loc[train_ids, 'pixel_hash'].to_numpy()
    local_split = args.out / 'split.csv'
    if local_split.exists():
        manifest = pd.read_csv(local_split, dtype=str)
        if args.split is not None:
            requested = pd.read_csv(args.split, dtype=str).sort_values('Id').reset_index(drop=True)
            pd.testing.assert_frame_equal(manifest.sort_values('Id').reset_index(drop=True), requested)
    elif args.split is not None:
        manifest = pd.read_csv(args.split, dtype=str)
    else:
        y = labels.Label.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
        tr, va = next(StratifiedGroupKFold(5, shuffle=True, random_state=args.seed).split(
            np.zeros(len(y)), y, groups=groups))
        manifest = labels.copy()
        manifest['split'] = 'train'
        manifest.loc[va, 'split'] = 'validation'
    manifest = manifest.sort_values('Id').reset_index(drop=True)
    pd.testing.assert_frame_equal(manifest[['Id', 'Label']], labels)
    if set(manifest['split']) != {'train', 'validation'}:
        raise ValueError('Split must contain train and validation only')
    tr = np.flatnonzero(manifest['split'].eq('train'))
    va = np.flatnonzero(manifest['split'].eq('validation'))
    if set(groups[tr]) & set(groups[va]):
        raise ValueError('An exact-image group crosses train/validation')
    for indices in (tr, va):
        if set(labels.iloc[indices].Label) != set(CLASSES):
            raise ValueError('Each split must contain all seven classes')
    if not local_split.exists():
        manifest.to_csv(local_split, index=False)
    info = {
        'train_count': len(tr), 'validation_count': len(va), 'test_count': len(test_ids),
        'class_names': CLASSES, 'split_sha256': sha(local_split),
        'labels_sha256': sha(args.data / 'train_labels.csv'),
        'image_hash_manifest_sha256': sha(args.hashes),
        'group_rule': 'Exact RGB96 white-composited preprocessed pixel SHA256 from supplied audit',
        'source_split': str(args.split) if args.split else 'New first StratifiedGroupKFold split',
        'old_baseline_comparison': 'This holdout is not the missing original baseline holdout.',
        'test_label_usage': 'None; test images are used only for frozen feature extraction and inference.',
    }
    json_write(args.out / 'data_audit.json', info)
    emit('data_validated', **info)
    return labels, test_ids, tr, va


def load_backbone(args):
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0,
                             img_size=args.resolution, dynamic_img_size=True)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if 'model' in state:
        state = state['model']
    if 'state_dict' in state:
        state = state['state_dict']
    state = checkpoint_filter_fn(state, model)
    model.load_state_dict(state, strict=True)
    model.eval()
    model.requires_grad_(False)
    if not 1 <= args.last_blocks < len(model.blocks):
        raise ValueError('last-blocks must leave a nonempty frozen prefix and trainable suffix')
    return model


def prefix_forward(model, images, last_blocks):
    tokens = model.patch_embed(images)
    tokens = model._pos_embed(tokens)
    tokens = model.patch_drop(tokens)
    tokens = model.norm_pre(tokens)
    for block in list(model.blocks)[:-last_blocks]:
        tokens = block(tokens)
    return tokens


def pooled_features(tokens, norm, prefix_tokens=1):
    tokens = norm(tokens)
    return torch.cat((tokens[:, 0], tokens[:, prefix_tokens:].mean(dim=1)), dim=1)


class ImageDataset(Dataset):
    def __init__(self, data, split, ids, resolution, indices=None):
        self.root = data / split
        self.ids = ids
        self.resolution = resolution
        self.indices = np.arange(len(ids)) if indices is None else np.asarray(indices)
        self.mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        index = int(self.indices[i])
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


def cache_signature(args, ids, split, token_count, dimension):
    return {
        'format_version': 1, 'model': MODEL_NAME, 'checkpoint_sha256': sha(args.checkpoint),
        'split': split, 'ids_sha256': hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
        'split_manifest_sha256': sha(args.out / 'split.csv'),
        'image_hash_manifest_sha256': sha(args.hashes),
        'resolution': args.resolution, 'frozen_prefix_blocks': 12 - args.last_blocks,
        'compute_precision': args.precision, 'storage_dtype': 'float16',
        'shape': [len(ids), token_count, dimension],
        'preprocessing': 'RGBA on white; Lanczos contain; centred square; ImageNet mean/std',
        'versions': {name: importlib.metadata.version(name) for name in ['torch', 'timm', 'Pillow']},
    }


def cache_images(args, model, ids, split):
    cache = args.out / 'cache'
    cache.mkdir(exist_ok=True)
    token_count = (args.resolution // 14) ** 2 + model.num_prefix_tokens
    signature = cache_signature(args, ids, split, token_count, model.embed_dim)
    meta_path = cache / f'{split}_metadata.json'
    token_path = cache / f'{split}_tokens.npy'
    done_path = cache / f'{split}_complete.npy'
    hash_path = cache / f'{split}_pixel_sha256.npy'
    exists = [p.exists() for p in [meta_path, token_path, done_path, hash_path]]
    if any(exists) and not all(exists):
        raise RuntimeError(f'Incomplete cache header/files in {cache}; move this incomplete cache aside before retrying')
    if all(exists):
        if json.loads(meta_path.read_text()) != signature:
            raise ValueError('Cache provenance differs from this run; use a new --out directory')
        tokens = np.load(token_path, mmap_mode='r+')
        done = np.load(done_path, mmap_mode='r+')
        pixel_hashes = np.load(hash_path, mmap_mode='r+')
        if tuple(tokens.shape) != tuple(signature['shape']) or done.shape != (len(ids),):
            raise ValueError('Cache shape mismatch')
    else:
        tokens = np.lib.format.open_memmap(token_path, mode='w+', dtype=np.float16,
                                          shape=tuple(signature['shape']))
        done = np.lib.format.open_memmap(done_path, mode='w+', dtype=bool, shape=(len(ids),))
        done[:] = False
        pixel_hashes = np.lib.format.open_memmap(hash_path, mode='w+', dtype='S64', shape=(len(ids),))
        done.flush()
        json_write(meta_path, signature)
    remaining = np.flatnonzero(~done)
    if len(remaining) == 0:
        emit('cache_already_complete', split=split, count=len(ids))
        return
    loader = DataLoader(ImageDataset(args.data, split, ids, args.resolution, remaining),
                        batch_size=args.cache_batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=False)
    started = time.perf_counter()
    last_emit = started
    new_count = 0
    with torch.inference_mode():
        for images, indices, digests in loader:
            with autocast(args):
                output = prefix_forward(model, images, args.last_blocks)
            values = output.float().numpy().astype(np.float16)
            if not np.isfinite(values).all():
                raise FloatingPointError('Nonfinite prefix cache features')
            idx = indices.numpy()
            tokens[idx] = values
            pixel_hashes[idx] = np.asarray(digests, dtype='S64')
            tokens.flush()
            pixel_hashes.flush()
            done[idx] = True
            done.flush()
            new_count += len(idx)
            now = time.perf_counter()
            if now - last_emit >= 15 or new_count == len(remaining):
                emit('cache_progress', split=split, completed=int(done.sum()), total=len(ids),
                     new_images_per_second=round(new_count / (now - started), 2),
                     remaining_seconds=round((len(remaining) - new_count) * (now - started) / new_count, 1))
                last_emit = now
    emit('cache_finished', split=split, seconds=round(time.perf_counter() - started, 2))


def open_cache(args, split):
    root = args.out / 'cache'
    metadata = json.loads((root / f'{split}_metadata.json').read_text())
    expected = {'resolution': args.resolution, 'frozen_prefix_blocks': 12 - args.last_blocks,
                'compute_precision': args.precision, 'checkpoint_sha256': sha(args.checkpoint),
                'split_manifest_sha256': sha(args.out / 'split.csv'),
                'image_hash_manifest_sha256': sha(args.hashes)}
    for key, value in expected.items():
        if metadata[key] != value:
            raise ValueError(f'Cache provenance mismatch: {key}')
    done = np.load(root / f'{split}_complete.npy', mmap_mode='r')
    if not bool(done.all()):
        raise ValueError(f'{split} prefix cache is not complete')
    tokens = np.load(root / f'{split}_tokens.npy', mmap_mode='r')
    if tokens.dtype != np.float16 or tuple(tokens.shape) != tuple(metadata['shape']):
        raise ValueError('Unexpected cached token dtype or shape')
    return tokens


class TailClassifier(nn.Module):
    def __init__(self, backbone, last_blocks):
        super().__init__()
        self.blocks = nn.ModuleList(copy.deepcopy(list(backbone.blocks)[-last_blocks:]))
        self.norm = copy.deepcopy(backbone.norm)
        self.num_prefix_tokens = backbone.num_prefix_tokens
        self.head = nn.Linear(backbone.embed_dim * 2, len(CLASSES))
        nn.init.trunc_normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
        self.requires_grad_(True)

    def forward(self, tokens):
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(pooled_features(tokens, self.norm, self.num_prefix_tokens))


class TokenDataset(Dataset):
    def __init__(self, cache, indices, labels=None):
        self.cache = cache
        self.indices = np.asarray(indices)
        self.labels = labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        index = int(self.indices[i])
        # A real copy avoids read-only memmap tensors and inference-mode tensors.
        tokens = torch.from_numpy(np.array(self.cache[index], dtype=np.float32))
        target = int(self.labels[index]) if self.labels is not None else -1
        return tokens, target, index


def token_loader(args, cache, indices, labels=None, shuffle=False, epoch=0):
    return DataLoader(TokenDataset(cache, indices, labels), batch_size=args.batch_size,
                      shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(args.seed + epoch))


def pretrained_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            if not name.startswith('head.')}


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.contiguous().numpy().tobytes())
    return h.hexdigest()


def backbone_change(model, initial):
    current = pretrained_state(model)
    squared, count, maximum, changed = 0., 0, 0., 0
    for name, before in initial.items():
        difference = current[name].double() - before.double()
        squared += float(difference.square().sum())
        count += difference.numel()
        maximum = max(maximum, float(difference.abs().max()))
        changed += int(bool(difference.ne(0).any()))
    return {'initial_pretrained_suffix_sha256': state_hash(initial),
            'final_pretrained_suffix_sha256': state_hash(current),
            'pretrained_tensors_changed': changed, 'pretrained_parameter_count': count,
            'pretrained_parameter_delta_l2': math.sqrt(squared),
            'pretrained_parameter_delta_rms': math.sqrt(squared / count),
            'pretrained_parameter_delta_max_abs': maximum,
            'true_backbone_fine_tuning': changed > 0 and squared > 0}


def evaluate(args, model, cache, indices, y=None):
    model.eval()
    all_probabilities, ordered_indices = [], []
    with torch.inference_mode():
        for tokens, targets, batch_indices in token_loader(args, cache, indices, y):
            with autocast(args):
                logits = model(tokens)
            probability = logits.float().softmax(dim=1).numpy()
            if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1., atol=1e-5):
                raise FloatingPointError('Invalid prediction probabilities')
            all_probabilities.append(probability)
            ordered_indices.extend(batch_indices.tolist())
    probabilities = np.concatenate(all_probabilities)
    if ordered_indices != list(map(int, indices)):
        raise ValueError('Prediction order changed')
    metrics = {}
    if y is not None:
        target = y[np.asarray(indices)]
        metrics = {'accuracy': float(accuracy_score(target, probabilities.argmax(axis=1))),
                   'cross_entropy': float(-np.log(np.maximum(probabilities[np.arange(len(target)), target], 1e-12)).mean())}
    return probabilities, metrics


def training_configuration(args, mode, fixed_epochs):
    return {'mode': mode, 'model': MODEL_NAME, 'resolution': args.resolution,
            'trainable_last_blocks': args.last_blocks, 'head': 'Linear over final-normalized CLS plus mean patch tokens',
            'epochs_budget': args.epochs, 'fixed_refit_epochs': fixed_epochs,
            'patience': args.patience, 'batch_size': args.batch_size,
            'backbone_lr': args.tail_lr, 'classifier_lr': args.head_lr,
            'weight_decay': args.weight_decay, 'optimizer': 'AdamW',
            'lr_schedule': 'Epoch cosine multiplier from1.0 toward0.1, horizon epochs_budget',
            'label_smoothing': args.label_smoothing, 'seed': args.seed,
            'compute_precision': args.precision,
            'selection_rule': ('Final fixed epoch chosen before all-label refit; no holdout re-evaluation'
                               if mode == 'all_data_refit' else
                               'Highest held-out accuracy, then lowest held-out cross-entropy'),
            'checkpoint_sha256': sha(args.checkpoint),
            'split_manifest_sha256': sha(args.out / 'split.csv'),
            'augmentations': 'None: cached deterministic white-composited letterbox images',
            'versions': {p: importlib.metadata.version(p) for p in ['torch', 'timm', 'numpy', 'pandas', 'scikit-learn', 'Pillow']},
            'python_version': sys.version,
            'source_url': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth',
            'ai_usage': 'OpenAI Codex generated and debugged this training pipeline; retain conversation prompts with submission.'}


def fit(args, backbone, labels, tr, va, refit=False):
    cache = open_cache(args, 'train')
    input_hashes = np.load(args.out / 'cache/train_pixel_sha256.npy', mmap_mode='r')
    if set(input_hashes[tr].tolist()) & set(input_hashes[va].tolist()):
        raise ValueError('Identical model-resolution images cross train/validation; revise grouping before training')
    y = labels.Label.map({name: i for i, name in enumerate(CLASSES)}).to_numpy(dtype=np.int64)
    seed_all(args.seed)
    model = TailClassifier(backbone, args.last_blocks)
    initial = pretrained_state(model)
    initial_head = {name: value.detach().cpu().clone() for name, value in model.head.state_dict().items()}
    mode = 'all_data_refit' if refit else 'selection'
    directory = args.out / mode
    directory.mkdir(exist_ok=True)
    chosen = None
    if refit:
        chosen = torch.load(args.out / 'selection/best.pt', map_location='cpu', weights_only=True)
        fixed_epochs = int(chosen['epoch'])
        indices = np.arange(len(labels))
    else:
        fixed_epochs = None
        indices = tr
    config = training_configuration(args, mode, fixed_epochs)
    if refit:
        expected_config = dict(chosen['configuration'], mode='all_data_refit', fixed_refit_epochs=fixed_epochs,
                               selection_rule='Final fixed epoch chosen before all-label refit; no holdout re-evaluation')
        if config != expected_config:
            raise ValueError('All-data refit must preserve the selected training settings and environment')
    config_path = directory / 'configuration.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Training configuration changed in an existing run; use a new output directory')
    json_write(config_path, config)
    torch_write(directory / 'initial_pretrained_suffix.pt', initial)
    torch_write(directory / 'initial_classifier.pt', initial_head)
    initial_hash = state_hash(initial)
    optimizer = torch.optim.AdamW([
        {'params': list(model.blocks.parameters()) + list(model.norm.parameters()), 'lr': args.tail_lr},
        {'params': model.head.parameters(), 'lr': args.head_lr}], weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    total_epochs = fixed_epochs if refit else args.epochs
    history, best_score, best_epoch, stale, starting_epoch = [], (-float('inf'), -float('inf')), 0, 0, 1
    gradient_seen = 0.
    last_path, best_path = directory / 'last.pt', directory / 'best.pt'
    if last_path.exists():
        saved = torch.load(last_path, map_location='cpu', weights_only=True)
        if saved['configuration'] != config or saved['initial_pretrained_suffix_sha256'] != initial_hash:
            raise ValueError('Resume checkpoint provenance differs from this training configuration')
        model.load_state_dict(saved['model_state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        history = saved['history']
        best_score = tuple(saved['best_score'])
        best_epoch, stale = saved['best_epoch'], saved['stale']
        starting_epoch = saved['epoch'] + 1
        gradient_seen = saved.get('backbone_gradient_norm_first_batch', 0.)
        torch.set_rng_state(saved['torch_rng_state'])
        emit('training_resumed', mode=mode, next_epoch=starting_epoch, best_epoch=best_epoch)
    emit('training_started', mode=mode, examples=len(indices), epochs=total_epochs,
         backbone_trainable_parameters=sum(p.numel() for p in model.blocks.parameters()) + sum(p.numel() for p in model.norm.parameters()),
         head_trainable_parameters=sum(p.numel() for p in model.head.parameters()))
    for epoch in range(starting_epoch, total_epochs + 1):
        if not refit and stale >= args.patience:
            break
        begun = time.perf_counter()
        multiplier = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / args.epochs))
        optimizer.param_groups[0]['lr'] = args.tail_lr * multiplier
        optimizer.param_groups[1]['lr'] = args.head_lr * multiplier
        model.train()
        loss_sum, correct, count = 0., 0, 0
        for tokens, targets, _ in token_loader(args, cache, indices, y, shuffle=True, epoch=epoch):
            optimizer.zero_grad(set_to_none=True)
            with autocast(args):
                logits = model(tokens)
                loss = criterion(logits, targets)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            if gradient_seen == 0.:
                gradient_seen = math.sqrt(sum(float(p.grad.detach().float().square().sum())
                    for p in model.blocks.parameters() if p.grad is not None))
                if not math.isfinite(gradient_seen) or gradient_seen <= 0.:
                    raise RuntimeError('No nonzero gradient reaches the pretrained backbone suffix')
                emit('backbone_gradient_verified', mode=mode, gradient_l2=gradient_seen)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            size = len(targets)
            count += size
            loss_sum += float(loss.detach()) * size
            correct += int(logits.detach().argmax(dim=1).eq(targets).sum())
        row = {'epoch': epoch, 'train_loss': loss_sum / count, 'train_accuracy': correct / count,
               'backbone_lr': optimizer.param_groups[0]['lr'], 'head_lr': optimizer.param_groups[1]['lr']}
        if refit:
            score = (float(epoch), 0.)
        else:
            _, metric = evaluate(args, model, cache, va, y)
            row.update({'validation_accuracy': metric['accuracy'], 'validation_cross_entropy': metric['cross_entropy']})
            score = (metric['accuracy'], -metric['cross_entropy'])
        row['seconds'] = round(time.perf_counter() - begun, 3)
        history.append(row)
        improved = score > best_score
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
        payload = {'format_version': 1, 'model_state_dict': model.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch,
                   'configuration': config, 'history': history, 'best_score': list(best_score),
                   'best_epoch': best_epoch, 'stale': stale, 'class_names': CLASSES,
                   'initial_pretrained_suffix_sha256': initial_hash,
                   'initial_classifier_sha256': state_hash(initial_head),
                   'backbone_gradient_norm_first_batch': gradient_seen,
                   'torch_rng_state': torch.get_rng_state()}
        if improved:
            torch_write(best_path, payload)
        torch_write(last_path, payload)
        pd.DataFrame(history).to_csv(directory / 'history.csv', index=False)
        emit('epoch_complete', mode=mode, best_epoch=best_epoch, stale=stale, **row)
    selected = torch.load(best_path, map_location='cpu', weights_only=True)
    model.load_state_dict(selected['model_state_dict'], strict=True)
    change = backbone_change(model, initial)
    if not change['true_backbone_fine_tuning']:
        raise RuntimeError('Selected model did not update pretrained backbone parameters')
    summary = {'mode': mode, 'selected_epoch': selected['epoch'], 'epochs_completed': len(history),
               'backbone_gradient_norm_first_batch': gradient_seen, **change,
               'initial_classifier_sha256': state_hash(initial_head),
               'final_classifier_sha256': state_hash(model.head.state_dict()),
               'holdout_used_for_model_selection': not refit,
               'test_submission_performed': False}
    if refit:
        summary['validation_reporting'] = 'No post-refit holdout score: all labelled images were used for refit.'
        summary['selection_checkpoint_sha256'] = sha(args.out / 'selection/best.pt')
    else:
        probabilities, metric = evaluate(args, model, cache, va, y)
        predictions = probabilities.argmax(axis=1)
        frame = pd.DataFrame({'Id': labels.iloc[va].Id.to_numpy(), 'true_label': labels.iloc[va].Label.to_numpy(),
                              'predicted_label': [CLASSES[i] for i in predictions]})
        for i, name in enumerate(CLASSES):
            frame[f'p_{name}'] = probabilities[:, i]
        frame.to_csv(directory / 'validation_predictions.csv', index=False)
        summary.update(metric)
        summary['confusion_matrix'] = confusion_matrix(y[va], predictions, labels=range(len(CLASSES))).tolist()
        summary['classification_report'] = classification_report(y[va], predictions,
            labels=range(len(CLASSES)), target_names=CLASSES, output_dict=True, zero_division=0)
    json_write(directory / 'metrics.json', summary)
    emit('training_finished', **summary)
    return best_path


def predict_test(args, backbone, test_ids):
    cache = open_cache(args, 'test')
    if len(cache) != len(test_ids):
        raise ValueError('Test cache row count differs from image count')
    path = args.model_file or args.out / ('all_data_refit/best.pt' if args.refit_all else 'selection/best.pt')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    config = saved['configuration']
    if saved['class_names'] != CLASSES or config['checkpoint_sha256'] != sha(args.checkpoint):
        raise ValueError('Prediction checkpoint classes or pretrained origin mismatch')
    if config['resolution'] != args.resolution or config['trainable_last_blocks'] != args.last_blocks:
        raise ValueError('Prediction checkpoint resolution or trainable architecture mismatch')
    if config['compute_precision'] != args.precision or config['split_manifest_sha256'] != sha(args.out / 'split.csv'):
        raise ValueError('Prediction checkpoint precision or split provenance mismatch')
    model = TailClassifier(backbone, args.last_blocks)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    initial = pretrained_state(TailClassifier(backbone, args.last_blocks))
    change = backbone_change(model, initial)
    if not change['true_backbone_fine_tuning']:
        raise RuntimeError('The prediction checkpoint is not a fine-tuned pretrained backbone')
    probabilities, _ = evaluate(args, model, cache, np.arange(len(test_ids)))
    prediction = pd.DataFrame({'Id': test_ids, 'Label': [CLASSES[i] for i in probabilities.argmax(axis=1)]})
    if prediction.columns.tolist() != ['Id', 'Label'] or len(prediction) != 9879 or not prediction.Id.is_unique:
        raise ValueError('Invalid submission columns, length or duplicate IDs')
    if prediction.Id.tolist() != [str(i) for i in range(10001, 19880)]:
        raise ValueError('Invalid test submission IDs/order')
    if not set(prediction.Label).issubset(CLASSES) or prediction.isna().any().any():
        raise ValueError('Invalid test labels')
    path_out = args.out / 'submission_dinov2_finetuned.csv'
    prediction.to_csv(path_out, index=False)
    pd.testing.assert_frame_equal(prediction, pd.read_csv(path_out, dtype=str))
    np.save(args.out / 'test_probabilities.npy', probabilities)
    receipt = {'submission_file': str(path_out), 'submission_sha256': sha(path_out),
               'rows': len(prediction), 'class_names': CLASSES, 'counts': prediction.Label.value_counts().to_dict(),
               'fine_tuned_checkpoint': str(path), 'fine_tuned_checkpoint_sha256': sha(path),
               'pretrained_checkpoint_sha256': sha(args.checkpoint), 'selected_epoch': saved['epoch'],
               'configuration': config, 'test_submission_performed': False, **change}
    json_write(args.out / 'submission_receipt.json', receipt)
    emit('submission_saved', **receipt)


def benchmark(args, model, ids):
    sample = next(iter(DataLoader(ImageDataset(args.data, 'train', ids[:args.batch_size], args.resolution),
                                 batch_size=args.batch_size)))[0]
    rows = []
    for precision in ('fp32', 'bf16'):
        args.precision = precision
        with torch.inference_mode(), autocast(args):
            prefix = prefix_forward(model, sample, args.last_blocks)
            suffix = prefix
            for block in list(model.blocks)[-args.last_blocks:]:
                suffix = block(suffix)
            expected = model.forward_features(sample)
            actual = model.norm(suffix)
            maximum_error = float((actual.float() - expected.float()).abs().max())
            if maximum_error > 1e-4:
                raise ValueError(f'Frozen-prefix split does not reproduce full backbone: {maximum_error}')
            start = time.perf_counter()
            for _ in range(3):
                prefix_forward(model, sample, args.last_blocks)
            elapsed = (time.perf_counter() - start) / 3
        rows.append({'precision': precision, 'batch': len(sample), 'prefix_seconds': elapsed,
                     'images_per_second': len(sample) / elapsed, 'full_backbone_max_error': maximum_error})
    json_write(args.out / 'benchmark.json', rows)
    emit('benchmark', results=rows)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['audit', 'benchmark', 'cache', 'train', 'refit', 'predict', 'all'], default='all')
    parser.add_argument('--data', type=Path, default=DEFAULT_ROOT / 'cv_review/dataset')
    parser.add_argument('--hashes', type=Path, default=DEFAULT_ROOT / 'cv_review/image_hashes.csv')
    parser.add_argument('--split', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_ROOT / 'cv_meta_training/dinov2_vits14_pretrain.pth')
    parser.add_argument('--out', type=Path, default=DEFAULT_ROOT / 'cv_meta_training/run_224')
    parser.add_argument('--cache-set', choices=['train', 'test', 'both'], default='train')
    parser.add_argument('--resolution', type=int, default=224)
    parser.add_argument('--last-blocks', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--cache-batch-size', type=int, default=16)
    parser.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--tail-lr', type=float, default=3e-5)
    parser.add_argument('--head-lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--label-smoothing', type=float, default=0.)
    parser.add_argument('--refit-all', action='store_true')
    parser.add_argument('--model-file', type=Path)
    args = parser.parse_args()
    if args.resolution % 14 or args.resolution < 28:
        parser.error('DINOv2 resolution must be a positive multiple of14 (e.g.112,126,168,224)')
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.cache_batch_size < 1:
        parser.error('Epochs, patience and batch sizes must all be positive')
    if args.tail_lr <= 0 or args.head_lr <= 0 or not 0 <= args.label_smoothing < 1:
        parser.error('Learning rates must be positive and label smoothing must be in[0,1)')
    args.out.mkdir(parents=True, exist_ok=True)
    return args


def main():
    args = arguments()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    seed_all(args.seed)
    labels, test_ids, tr, va = prepare_data(args)
    if args.stage == 'audit':
        return
    model = load_backbone(args)
    if args.stage == 'benchmark':
        benchmark(args, model, labels.Id.tolist())
        return
    if args.stage in ('cache', 'all'):
        if args.cache_set in ('train', 'both') or args.stage == 'all':
            cache_images(args, model, labels.Id.tolist(), 'train')
        if args.cache_set in ('test', 'both'):
            cache_images(args, model, test_ids, 'test')
    if args.stage == 'cache':
        return
    if args.stage in ('train', 'all'):
        fit(args, model, labels, tr, va, refit=False)
    if args.stage == 'refit' or (args.stage == 'all' and args.refit_all):
        fit(args, model, labels, tr, va, refit=True)
    if args.stage in ('predict', 'all'):
        cache_images(args, model, test_ids, 'test')
        predict_test(args, model, test_ids)


if __name__ == '__main__':
    main()
