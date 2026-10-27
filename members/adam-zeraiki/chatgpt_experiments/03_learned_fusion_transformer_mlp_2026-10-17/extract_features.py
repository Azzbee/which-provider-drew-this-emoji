#!/usr/bin/env python3
"""Extract aligned frozen and task-fine-tuned encoder features for fusion.

AI assistance: OpenAI Codex authored this experiment at the user's request.
The original package supplies audited preprocessing, pretrained weights and
selected supervised tails. No labels are used during this extraction.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

CLASSES = ['apple', 'facebook', 'google', 'messenger', 'mozilla', 'samsung', 'whatsapp']

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + '\n')

def emit(event, **data):
    print(json.dumps({'event':event, **data}), flush=True)

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

def autocast():
    return torch.autocast('cpu', dtype=torch.bfloat16)

def init_arrays(out, split, branches, count, signature):
    meta = out / f'{split}_metadata.json'
    if meta.exists():
        if json.loads(meta.read_text()) != signature:
            raise ValueError(f'Cache signature changed: {out}')
        arrays = {b: np.load(out / f'{split}_{b}.npy', mmap_mode='r+') for b in branches}
        done = np.load(out / f'{split}_complete.npy', mmap_mode='r+')
    else:
        out.mkdir(parents=True, exist_ok=True)
        arrays = {b: np.lib.format.open_memmap(out / f'{split}_{b}.npy', mode='w+',
                   dtype=np.float32, shape=(count, d)) for b,d in branches.items()}
        done = np.lib.format.open_memmap(out / f'{split}_complete.npy', mode='w+',
                                        dtype=bool, shape=(count,))
        done[:] = False
        done.flush()
        write_json(meta, signature)
    for b,d in branches.items():
        assert arrays[b].shape == (count,d) and arrays[b].dtype == np.float32
    assert done.shape == (count,)
    return arrays,done

def signature(args, split, ids, extra):
    return {'format_version':1, 'split':split, 'ids':ids,
            'split_sha256':sha(args.package/'dino_original/run_224/split.csv'),
            'labels_sha256':sha(args.data/'train_labels.csv'),
            'resolution':224, 'compute':'CPU bfloat16', 'prefix_roundtrip':'float16 then float32',
            'preprocessing':'RGBA white composite, Lanczos contain, centered white padding, ImageNet mean/std',
            'torch_version':str(torch.__version__), **extra}

def dino_features(tail, tokens):
    for block in tail.blocks:
        tokens = block(tokens)
    tokens = tail.norm(tokens)
    return torch.cat((tokens[:,0], tokens[:,tail.num_prefix_tokens:].mean(dim=1)),dim=1)

def run_dino(args, ids, split):
    d = args.package/'dino_original'
    tm = module(d/'train_meta.py','fusion_original_dino')
    pretrained = d/'dinov2_vits14_pretrain.pth'
    tuned_file = d/'run_224/selection/best.pt'
    assert sha(pretrained) == 'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
    sig = signature(args,split,ids,{'model':'DINOv2 ViT-S/14', 'pretrained_sha256':sha(pretrained),
                    'fine_tuned_checkpoint_sha256':sha(tuned_file), 'feature':'CLS plus mean patch, 768 values'})
    out=args.out/'dino'
    arrays,done=init_arrays(out,split,{'pretrained':768,'tuned':768,'tuned_probabilities':7},len(ids),sig)
    if done.all():
        emit('already_complete',component='dino',split=split)
        return
    backbone=tm.load_backbone(SimpleNamespace(checkpoint=pretrained,resolution=224,last_blocks=2))
    original=tm.TailClassifier(backbone,2).eval().requires_grad_(False)
    tuned=tm.TailClassifier(backbone,2).eval().requires_grad_(False)
    saved=torch.load(tuned_file,map_location='cpu',weights_only=True)
    assert saved['class_names']==CLASSES
    tuned.load_state_dict(saved['model_state_dict'],strict=True)
    del saved
    pending=np.flatnonzero(~done)
    loader=DataLoader(tm.ImageDataset(args.data,split,ids,224,indices=pending),
                      batch_size=args.batch_size,shuffle=False,num_workers=0)
    started=last=time.perf_counter()
    count=0
    with torch.inference_mode():
        for images,index,_ in loader:
            with autocast():
                prefix=tm.prefix_forward(backbone,images,2)
            prefix=prefix.float().to(torch.float16).float()
            with autocast():
                frozen_features=dino_features(original,prefix)
                trained_features=dino_features(tuned,prefix)
                logits=tuned.head(trained_features)
            values={'pretrained':frozen_features.float().numpy(),
                    'tuned':trained_features.float().numpy(),
                    'tuned_probabilities':logits.float().softmax(1).numpy()}
            ix=index.numpy()
            for name,value in values.items():
                assert np.isfinite(value).all()
                arrays[name][ix]=value
            for array in arrays.values(): array.flush()
            done[ix]=True
            done.flush()
            count+=len(ix)
            now=time.perf_counter()
            if now-last>=20 or done.all():
                emit('features_progress',component='dino',split=split,done=int(done.sum()),
                     total=len(ids),seconds=round(now-started,1),
                     remaining_seconds=round((len(pending)-count)*(now-started)/count,1))
                last=now
    write_json(out/f'{split}_receipt.json',{'complete':bool(done.all()),'rows':len(ids),
       'seconds':time.perf_counter()-started,'hashes':{name:sha(out/f'{split}_{name}.npy') for name in arrays},
       'label_training_performed':False,'original_encoder_task_supervision':False,
       'tuned_encoder_note':'Existing last-two-block checkpoint learned on7903 and selected on reused1976 development holdout.'})

def run_conv(args,ids,split):
    d=args.package/'convnext'
    cm=module(d/'train_convnext.py','fusion_original_conv')
    pretrained=d/'convnext_tiny-983f1562.pth'
    tuned_file=d/'run_224/selection/best.pt'
    cache_meta=json.loads((args.conv_cache/f'{split}_metadata.json').read_text())
    assert cache_meta['ids_sha256']==hashlib.sha256('\n'.join(ids).encode()).hexdigest()
    assert cache_meta['checkpoint_sha256']==sha(pretrained)
    assert cache_meta['split_manifest_sha256']==sha(args.package/'dino_original/run_224/split.csv')
    assert np.load(args.conv_cache/f'{split}_complete.npy').all()
    cache=np.load(args.conv_cache/f'{split}_tokens.npy',mmap_mode='r')
    assert cache.shape==(len(ids),384,14,14) and cache.dtype==np.float16
    sig=signature(args,split,ids,{'model':'ConvNeXt-Tiny','pretrained_sha256':sha(pretrained),
                   'fine_tuned_checkpoint_sha256':sha(tuned_file),'feature':'Normalized pooled final-stage768 features',
                   'prefix_cache_metadata_sha256':sha(args.conv_cache/f'{split}_metadata.json')})
    out=args.out/'convnext'
    arrays,done=init_arrays(out,split,{'pretrained':768,'tuned':768,'tuned_probabilities':7},len(ids),sig)
    if done.all():
        emit('already_complete',component='convnext',split=split)
        return
    backbone=cm.load_backbone(SimpleNamespace(checkpoint=pretrained))
    original=cm.TailClassifier(backbone).eval().requires_grad_(False)
    tuned=cm.TailClassifier(backbone).eval().requires_grad_(False)
    saved=torch.load(tuned_file,map_location='cpu',weights_only=True)
    assert saved['class_names']==CLASSES
    tuned.load_state_dict(saved['model_state_dict'],strict=True)
    del saved,backbone
    existing=args.conv_cache.parent/'frozen_features'/f'{split}.npy'
    frozen=np.load(existing,mmap_mode='r') if existing.exists() else None
    if frozen is not None:
        assert frozen.shape==(len(ids),768) and np.isfinite(frozen).all()
    pending=np.flatnonzero(~done)
    started=last=time.perf_counter()
    count=0
    with torch.inference_mode():
        for first in range(0,len(pending),args.batch_size):
            ix=pending[first:first+args.batch_size]
            tokens=torch.from_numpy(np.array(cache[ix],dtype=np.float32))
            with autocast():
                frozen_features=original.features(tokens) if frozen is None else None
                trained_features=tuned.features(tokens)
                logits=tuned.head(trained_features)
            values={'pretrained':frozen_features.float().numpy() if frozen is None else frozen[ix],
                    'tuned':trained_features.float().numpy(),
                    'tuned_probabilities':logits.float().softmax(1).numpy()}
            for name,value in values.items():
                assert np.isfinite(value).all()
                arrays[name][ix]=value
            for array in arrays.values(): array.flush()
            done[ix]=True
            done.flush()
            count+=len(ix)
            now=time.perf_counter()
            if now-last>=20 or done.all():
                emit('features_progress',component='convnext',split=split,done=int(done.sum()),total=len(ids),
                     seconds=round(now-started,1),remaining_seconds=round((len(pending)-count)*(now-started)/count,1))
                last=now
    write_json(out/f'{split}_receipt.json',{'complete':bool(done.all()),'rows':len(ids),
       'seconds':time.perf_counter()-started,'hashes':{name:sha(out/f'{split}_{name}.npy') for name in arrays},
       'label_training_performed':False,'original_encoder_task_supervision':False,
       'tuned_encoder_note':'Existing final-stage checkpoint learned on7903 and selected on reused1976 development holdout.'})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--conv-cache',type=Path,required=True)
    p.add_argument('--components',nargs='+',default=['convnext','dino'])
    p.add_argument('--splits',nargs='+',default=['train','test'])
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=16)
    args=p.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    labels=pd.read_csv(args.data/'train_labels.csv',dtype=str).sort_values('Id').reset_index(drop=True)
    split=pd.read_csv(args.package/'dino_original/run_224/split.csv',dtype=str).sort_values('Id').reset_index(drop=True)
    pd.testing.assert_frame_equal(labels,split[['Id','Label']])
    assert len(labels)==9879 and labels.Id.is_unique and sorted(labels.Label.unique())==CLASSES
    test_ids=sorted(f.stem for f in (args.data/'test').glob('*.png'))
    assert test_ids==list(map(str,range(10001,19880)))
    args.out.mkdir(parents=True,exist_ok=True)
    write_json(args.out/'train_ids.json',labels.Id.tolist())
    write_json(args.out/'test_ids.json',test_ids)
    for s in args.splits:
        ids=labels.Id.tolist() if s=='train' else test_ids
        for component in args.components:
            if component=='dino': run_dino(args,ids,s)
            elif component=='convnext': run_conv(args,ids,s)
            else: raise ValueError(component)
    emit('extraction_finished',components=args.components,splits=args.splits)

if __name__=='__main__': main()
