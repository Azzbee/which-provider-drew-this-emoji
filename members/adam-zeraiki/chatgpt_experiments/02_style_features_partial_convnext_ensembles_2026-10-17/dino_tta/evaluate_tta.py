#!/usr/bin/env python3
"""Reproduce a saved DINOv2 model and evaluate fixed original/flip averaging.

AI-assisted implementation for the emoji-provider project. No fitting, hidden
test labels, file-metadata features, or image-matching prediction shortcuts.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--split',choices=['validation','test'],default='validation')
    p.add_argument('--views',nargs='+',choices=['original','flip'],default=['original','flip'])
    p.add_argument('--threads',type=int,default=6)
    p.add_argument('--batch-size',type=int,default=16)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    sys.path.insert(0,str(args.experiment))
    import train_meta as tm
    manifest=pd.read_csv(args.experiment/'run_224/split.csv',dtype=str)
    labels=pd.read_csv(args.data/'train_labels.csv',dtype=str).sort_values('Id').reset_index(drop=True)
    pd.testing.assert_frame_equal(manifest[['Id','Label']],labels)
    model_file=args.experiment/'run_224/selection/best.pt'
    saved=torch.load(model_file,map_location='cpu',weights_only=True)
    cfg=saved['configuration']
    assert saved['class_names']==tm.CLASSES
    assert cfg['compute_precision']=='bf16', 'This inference implementation requires the saved BF16 recipe.'
    assert cfg['split_manifest_sha256']==sha(args.experiment/'run_224/split.csv')
    assert cfg['checkpoint_sha256']==sha(args.experiment/'dinov2_vits14_pretrain.pth')
    ma=SimpleNamespace(resolution=cfg['resolution'],checkpoint=args.experiment/'dinov2_vits14_pretrain.pth',last_blocks=cfg['trainable_last_blocks'])
    backbone=tm.load_backbone(ma)
    tail=tm.TailClassifier(backbone,ma.last_blocks)
    tail.load_state_dict(saved['model_state_dict'],strict=True)
    tail.eval().requires_grad_(False)
    if args.split=='validation':
        selected=manifest.loc[manifest['split'].eq('validation')].copy()
        ids=selected.Id.tolist()
        y=selected.Label.map({c:i for i,c in enumerate(tm.CLASSES)}).to_numpy()
        dataset_split='train'
    else:
        ids=sorted(x.stem for x in (args.data/'test').glob('*.png'))
        assert ids==[str(x) for x in range(10001,19880)]
        dataset_split='test'
        y=None
    (args.out/f'{args.split}_ids.json').write_text(json.dumps(ids))
    loader=DataLoader(tm.ImageDataset(args.data,dataset_split,ids,ma.resolution),batch_size=args.batch_size,shuffle=False,num_workers=0)
    metrics={}
    for view in args.views:
        probabilities=[]
        seen=[]
        started=last=time.perf_counter()
        with torch.inference_mode():
            for image,indices,_ in loader:
                if view=='flip':
                    image=torch.flip(image,[-1])
                with torch.autocast('cpu',dtype=torch.bfloat16):
                    token=tm.prefix_forward(backbone,image,ma.last_blocks)
                token=token.float().to(torch.float16).float()
                with torch.autocast('cpu',dtype=torch.bfloat16):
                    logits=tail(token)
                probabilities.append(logits.float().softmax(1).numpy())
                seen.extend(indices.tolist())
                now=time.perf_counter()
                if now-last>=15 or len(seen)==len(ids):
                    print(json.dumps({'event':'inference_progress','split':args.split,'view':view,'completed':len(seen),'total':len(ids),'seconds':round(now-started,1)}),flush=True)
                    last=now
        assert seen==list(range(len(ids)))
        prob=np.concatenate(probabilities)
        assert prob.shape==(len(ids),7) and np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-6)
        np.save(args.out/f'{args.split}_{view}_probabilities.npy',prob)
        if y is not None:
            pred=prob.argmax(1)
            metrics[view]={'correct':int((pred==y).sum()),'total':len(y),'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro')),'log_loss':float(log_loss(y,prob,labels=list(range(7))))}
            print(json.dumps({'event':'view_result','view':view,**metrics[view]}),flush=True)
    original_file=args.out/f'{args.split}_original_probabilities.npy'
    if 'original' not in args.views:
        if args.split=='test':
            np.save(original_file,np.load(args.experiment/'run_224/test_probabilities.npy'))
        else:
            ref=pd.read_csv(args.experiment/'run_224/selection/validation_predictions.csv',dtype={'Id':str})
            assert ref.Id.tolist()==ids
            np.save(original_file,ref[[f'p_{c}' for c in tm.CLASSES]].to_numpy(np.float32))
    if 'flip' in args.views:
        original=np.load(original_file)
        flipped=np.load(args.out/f'{args.split}_flip_probabilities.npy')
        average=(original+flipped)/2
        np.save(args.out/f'{args.split}_tta_probabilities.npy',average)
        if y is not None:
            pred=average.argmax(1)
            metrics['tta']={'correct':int((pred==y).sum()),'total':len(y),'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro')),'log_loss':float(log_loss(y,average,labels=list(range(7)))),'classification_report':classification_report(y,pred,target_names=tm.CLASSES,output_dict=True,zero_division=0)}
            ref=pd.read_csv(args.experiment/'run_224/selection/validation_predictions.csv',dtype={'Id':str})
            assert ref.Id.tolist()==ids
            baseline=ref[[f'p_{c}' for c in tm.CLASSES]].to_numpy(np.float32)
            metrics['reproduction']={'maximum_probability_difference':float(np.max(np.abs(original-baseline))),'prediction_disagreements':int((original.argmax(1)!=baseline.argmax(1)).sum()),'reference_correct':int((baseline.argmax(1)==y).sum())}
            metrics['tta']['baseline_wrong_candidate_correct']=int(((baseline.argmax(1)!=y)&(pred==y)).sum())
            metrics['tta']['baseline_correct_candidate_wrong']=int(((baseline.argmax(1)==y)&(pred!=y)).sum())
            print(json.dumps({'event':'tta_result','correct':metrics['tta']['correct'],'accuracy':metrics['tta']['accuracy'],'reproduction':metrics['reproduction']}),flush=True)
        else:
            pd.DataFrame({'Id':ids,'Label':[tm.CLASSES[i] for i in average.argmax(1)]}).to_csv(args.out/'submission_dinov2_tta.csv',index=False)
    receipt={'class_names':tm.CLASSES,'split':args.split,'views':args.views,'average_rule':'0.5 original + 0.5 horizontal-flip probabilities','model_checkpoint_sha256':sha(model_file),'pretrained_checkpoint_sha256':sha(ma.checkpoint),'split_manifest_sha256':sha(args.experiment/'run_224/split.csv'),'metrics':metrics,'validation_is_exploratory':True,'kaggle_uploaded':False}
    (args.out/f'{args.split}_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':
    main()
