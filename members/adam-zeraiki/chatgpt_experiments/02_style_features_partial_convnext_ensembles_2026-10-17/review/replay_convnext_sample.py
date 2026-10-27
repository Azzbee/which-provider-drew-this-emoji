#!/usr/bin/env python3
"""Replay a small deterministic test sample from packaged ConvNeXt weights.

This independently maps the trained suffix into Torchvision's full model;
it does not import the experiment's TailClassifier or preprocessing helpers.
Only inference is performed. AI-assisted audit; retain with experiment code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision.models import convnext_tiny

CLASSES = ['apple', 'facebook', 'google', 'messenger', 'mozilla', 'samsung', 'whatsapp']
PRETRAINED_SHA = '983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d'
MODEL_STATE_SHA = '11e863642b547222500392fa25ea32ff8b3871c80b43c046ea1a99bd28bfbf3b'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def image_tensor(path):
    with Image.open(path) as image:
        rgba = image.convert('RGBA')
        rgb = Image.alpha_composite(Image.new('RGBA', rgba.size, (255, 255, 255, 255)), rgba).convert('RGB')
        resized = ImageOps.contain(rgb, (224, 224), Image.Resampling.LANCZOS)
        canvas = Image.new('RGB', (224, 224), (255, 255, 255))
        canvas.paste(resized, ((224 - resized.width) // 2, (224 - resized.height) // 2))
        pixels = np.asarray(canvas, dtype=np.uint8).copy()
    values = torch.from_numpy(pixels).permute(2, 0, 1).to(torch.float32) / 255
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return (values - mean) / std, hashlib.sha256(pixels.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    pretrained = args.package / 'convnext/convnext_tiny-983f1562.pth'
    checkpoint = args.package / 'convnext/run_224/selection/best.pt'
    packaging = json.loads((args.package / 'convnext/checkpoint_packaging.json').read_text())
    assert sha(pretrained) == PRETRAINED_SHA
    assert sha(checkpoint) == packaging['portable_inference_checkpoint_sha256']
    assert packaging['all_model_tensors_exactly_equal'] is True
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert saved['class_names'] == CLASSES and saved['epoch'] == 4
    assert state_hash(saved['model_state_dict']) == MODEL_STATE_SHA
    assert saved['configuration']['resolution'] == 224
    assert saved['configuration']['compute_precision'] == 'bf16'
    assert saved['configuration']['all_label_refit'] is False

    model = convnext_tiny(weights=None, stochastic_depth_prob=0.1)
    model.load_state_dict(torch.load(pretrained, map_location='cpu', weights_only=True), strict=True)
    model.classifier[2] = torch.nn.Linear(768, 7)
    complete = model.state_dict()
    renamed = {}
    mapping = [('tail.0.', 'features.6.'), ('tail.1.', 'features.7.'),
               ('norm.', 'classifier.0.'), ('head.', 'classifier.2.')]
    for name, value in saved['model_state_dict'].items():
        destinations = [target + name[len(source):] for source, target in mapping if name.startswith(source)]
        assert len(destinations) == 1, name
        destination = destinations[0]
        assert destination in complete and complete[destination].shape == value.shape
        assert destination not in renamed
        renamed[destination] = value
    complete.update(renamed)
    model.load_state_dict(complete, strict=True)
    model.eval().requires_grad_(False)

    # Spread the 12 requested samples across the test set. Four padding examples
    # preserve the original batch shape of 16; their predictions are discarded.
    indices = np.linspace(0, 9878, 12, dtype=int)
    ids = [str(10001 + int(index)) for index in indices]
    padded_ids = ids + ['10002', '10003', '10004', '10005']
    rendered = [image_tensor(args.data / 'test' / f'{name}.png') for name in padded_ids]
    images = torch.stack([item[0] for item in rendered])
    with torch.inference_mode():
        with torch.autocast('cpu', dtype=torch.bfloat16):
            tokens = model.features[:6](images)
        tokens = tokens.float().to(torch.float16).float()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            logits = model.classifier(model.avgpool(model.features[6:](tokens)))
        probabilities = logits.float().softmax(dim=1).numpy()[:12]
    assert probabilities.shape == (12, 7) and probabilities.dtype == np.float32
    assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(1), 1, atol=1e-6)
    np.save(args.out / 'sample_probabilities.npy', probabilities)
    (args.out / 'sample_ids.json').write_text(json.dumps(ids) + '\n')
    receipt = {'scope': 'Independent sample inference from packaged original and fine-tuned ConvNeXt weights',
               'checkpoint_sha256': sha(checkpoint), 'pretrained_checkpoint_sha256': sha(pretrained),
               'full_trained_model_state_sha256': MODEL_STATE_SHA, 'selected_epoch': 4,
               'sample_ids': ids, 'test_row_indices': indices.tolist(), 'class_names': CLASSES,
               'predicted_labels': [CLASSES[i] for i in probabilities.argmax(1)],
               'rendered_pixel_sha256': [item[1] for item in rendered[:12]],
               'sample_probabilities_sha256': sha(args.out / 'sample_probabilities.npy'),
               'threads': 2, 'batch_size': 16, 'samples_verified': 12, 'padding_examples': 4,
               'training_performed': False, 'all_label_refit': False, 'kaggle_uploaded': False,
               'full_test_array_comparison': 'Pending full inference completion',
               'script_sha256': sha(__file__)}
    (args.out / 'sample_replay_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
