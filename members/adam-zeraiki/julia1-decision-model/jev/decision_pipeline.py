#!/usr/bin/env python3
"""Evaluate an actual open JEV-style typed decision model over image expert scores.

Julia-1 is not TypeSafe Jev. The published model is frozen and used zero-shot;
no emoji labels fit its parameters, prompt, calibration, or routing rule.
See experiment_plan.json and AI_USAGE.json for the fixed request and disclosure.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

# Bound BLAS pools before NumPy/PyTorch import in this shared CPU environment.
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'

HERE = Path(__file__).resolve().parent
if (HERE/'vendor').exists():
    sys.path.insert(0,str(HERE/'vendor'))
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['USE_TF']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'

import numpy as np
import pandas as pd
import torch

CLASSES=['apple','facebook','google','messenger','mozilla','samsung','whatsapp']
WEIGHTS_SHA='df853bf7fe424420011f3d0c47a05d7341aa9eefa7fb9f203ea4aada4ad95b72'
MODEL='SupersonicLabs/Julia-1'
REVISION='a85b127321d580d65176c89ced8273f305745d85'
QUESTION="Which provider most likely rendered this emoji image? Use the three image classifiers' probability estimates as evidence. Each classifier can be wrong; choose the provider best supported by the combined evidence."
EXPERTS=['DINOv2 visual classifier','ConvNeXt visual classifier','colour-edge-texture style classifier']


def emit(event,**fields):
    print(json.dumps({'event':event,**fields},default=str),flush=True)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def save_json(path,data):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,default=str)+'\n');tmp.replace(path)


def validate_probability(p,n):
    if p.shape!=(n,7) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-5):
        raise ValueError('Invalid expert probabilities')


def load_experts(package,split):
    package=Path(package)
    if json.loads((package/'style/results/classes.json').read_text())!=CLASSES:
        raise ValueError('Style class order differs')
    if split=='validation':
        dino_path=package/'dino_original/run_224/selection/validation_predictions.csv'
        conv_path=package/'convnext/run_224/selection/validation_predictions.csv'
        style_path=package/'style/results/hist_gradient_boosting_validation_probabilities.npy'
        dino=pd.read_csv(dino_path,dtype={'Id':str}).sort_values('Id').reset_index(drop=True)
        conv=pd.read_csv(conv_path,dtype={'Id':str}).sort_values('Id').reset_index(drop=True)
        pd.testing.assert_frame_equal(dino[['Id','true_label']],conv[['Id','true_label']])
        ids=dino.Id.tolist();truth=dino.true_label.tolist()
        if len(ids)!=1976 or len(set(ids))!=1976:raise ValueError('Unexpected validation IDs')
        d=dino[['p_'+c for c in CLASSES]].to_numpy(np.float32)
        c=conv[['p_'+c for c in CLASSES]].to_numpy(np.float32)
        style_ids=json.loads((package/'style/results/validation_ids.json').read_text())
    else:
        dino_path=package/'dino_original/run_224/test_probabilities.npy'
        conv_path=package/'convnext/run_224/test_probabilities_convnext_finetuned.npy'
        style_path=package/'style/results/hist_gradient_boosting_test_probabilities.npy'
        ids=pd.read_csv(package/'dino_original/run_224/submission_dinov2_finetuned.csv',dtype=str).Id.tolist()
        conv_ids=pd.read_csv(package/'convnext/run_224/submission_convnext_finetuned.csv',dtype=str).Id.tolist()
        if ids!=conv_ids or ids!=[str(i) for i in range(10001,19880)]:
            raise ValueError('Unexpected test IDs/order')
        truth=None;d=np.load(dino_path);c=np.load(conv_path)
        style_ids=json.loads((package/'style/results/test_ids.json').read_text())
    if len(set(style_ids))!=len(ids) or set(style_ids)!=set(ids):raise ValueError('Style IDs differ')
    style_index={v:i for i,v in enumerate(style_ids)}
    s=np.load(style_path)[[style_index[v] for v in ids]]
    for p in [d,c,s]:validate_probability(p,len(ids))
    provenance={str(p.relative_to(package)):sha(p) for p in [dino_path,conv_path,style_path]}
    return ids,truth,np.stack([d,c,s],axis=1),provenance


def make_requests(probabilities):
    rows=[]
    for example in probabilities:
        state={'task':'emoji rendering provider classification','predictions':{
            name:{c:round(float(value),4) for c,value in zip(CLASSES,p)}
            for name,p in zip(EXPERTS,example)}}
        rows.append({'type':'choice','state':state,'question':QUESTION,'options':CLASSES.copy()})
    return rows


def load_model(args):
    # Upstream source was reviewed before import. Use its plain PyTorch engine,
    # never the optional native build/router or remote-model code.
    if sha(args.model/'model.safetensors')!=WEIGHTS_SHA:
        raise ValueError('Julia checkpoint differs from the pinned public artifact')
    sys.path.insert(0,str(args.model))
    from julia.inference import TransformerEngine
    from julia.data import sequence,validate_row
    os.environ['JULIA_CPU_THREADS']=str(args.threads)
    engine=TransformerEngine(args.model,device='cpu',max_length=1024,head_length=256,memory_map=True)
    engine.model.requires_grad_(False)
    receipt={'model':MODEL,'revision':REVISION,'weights_sha256':WEIGHTS_SHA,
             'parameters':sum(p.numel() for p in engine.model.parameters()),
             'engine':'Upstream TransformerEngine, CPU float32, strict safetensors loading',
             'fine_tuned_on_emojis':False,'all_parameters_frozen':all(not p.requires_grad for p in engine.model.parameters()),
             'versions':{n:importlib.metadata.version(n) for n in ['torch','transformers','tokenizers','huggingface_hub','numpy','safetensors']}}
    save_json(args.out/'model_load_receipt.json',receipt)
    emit('model_loaded',**receipt)
    return engine,sequence,validate_row


def predict(args):
    ids,truth,expert,provenance=load_experts(args.package,args.split)
    # Routing is determined only from model top1 outputs, before labels are scored.
    disagreement=expert[:,0].argmax(1)!=expert[:,1].argmax(1)
    chosen=np.flatnonzero(disagreement) if args.disagreements_only else np.arange(len(ids))
    if args.limit:chosen=chosen[:args.limit]
    out_ids=[ids[i] for i in chosen]
    requests=make_requests(expert[chosen])
    signature={'model':MODEL,'revision':REVISION,'question':QUESTION,'classes':CLASSES,
               'split':args.split,'ids':out_ids,'source_probabilities_sha256':provenance,
               'disagreements_only':args.disagreements_only,'limit':args.limit,
               'weights_sha256':WEIGHTS_SHA,'precision':'float32','batch_size':args.batch_size,
               'max_length':1024,'head_length':256,'strict_encoding':True}
    args.out.mkdir(exist_ok=True,parents=True)
    meta=args.out/'prediction_config.json'
    if meta.exists() and json.loads(meta.read_text())!=signature:
        raise ValueError('Output directory contains another prediction configuration')
    save_json(meta,signature)
    with (args.out/'requests.jsonl').open('w') as f:
        for key,row in zip(out_ids,requests):
            # The identifier is outside the model request; it is never encoded.
            f.write(json.dumps({'Id':key,'request':row},ensure_ascii=False)+'\n')
    save_json(args.out/'ids.json',out_ids)
    engine,sequence,validate_row=load_model(args)
    lengths=[]
    for i,row in enumerate(requests):
        validate_row(row,i+1)
        row['_encoded']=sequence(engine.tokenizer,row,1024,256,strict=True)
        if row['_encoded']['truncated']:raise ValueError('Unexpected truncation')
        lengths.append(len(row['_encoded']['ids']))
    emit('requests_encoded',count=len(requests),min_tokens=min(lengths),max_tokens=max(lengths),mean_tokens=float(np.mean(lengths)))
    warm=engine.logits(requests[:min(args.batch_size,len(requests))])
    probabilities=np.zeros((len(requests),7),dtype=np.float32)
    logits=np.zeros_like(probabilities)
    done=np.zeros(len(requests),dtype=bool)
    started=time.perf_counter();last=started
    for offset in range(0,len(requests),args.batch_size):
        rows=requests[offset:offset+args.batch_size]
        z=np.asarray(engine.logits(rows),dtype=np.float32)
        q=np.exp(z-z.max(axis=1,keepdims=True));q/=q.sum(axis=1,keepdims=True)
        validate_probability(q,len(rows))
        logits[offset:offset+len(rows)]=z
        probabilities[offset:offset+len(rows)]=q
        done[offset:offset+len(rows)]=True
        np.save(args.out/'probabilities.npy',probabilities)
        np.save(args.out/'logits.npy',logits)
        np.save(args.out/'complete.npy',done)
        if time.perf_counter()-last>=20:
            emit('prediction_progress',completed=int(done.sum()),total=len(requests),elapsed_s=time.perf_counter()-started)
            last=time.perf_counter()
    elapsed=time.perf_counter()-started
    validation=pd.DataFrame({'Id':out_ids,'predicted_label':np.asarray(CLASSES)[probabilities.argmax(1)]})
    for i,c in enumerate(CLASSES):validation['p_'+c]=probabilities[:,i]
    validation.to_csv(args.out/'predictions.csv',index=False)
    receipt={'count':len(requests),'elapsed_s':elapsed,'rows_per_second':len(requests)/elapsed,
             'threads':args.threads,'batch_size':args.batch_size,'token_length_mean':float(np.mean(lengths)),
             'token_length_max':max(lengths),'all_complete':bool(done.all()),'finite':bool(np.isfinite(probabilities).all()),
             'probabilities_sha256':sha(args.out/'probabilities.npy'),
             'no_model_request_contains_labels_or_ids':True,'source_probabilities_sha256':provenance,
             'mode':'JEV-style zero-shot Julia-1 decision model over three image experts; not TypeSafe Jev'}
    save_json(args.out/'prediction_receipt.json',receipt)
    emit('prediction_complete',**receipt)
    if not args.limit and args.split=='validation' and not args.disagreements_only:
        hybrid=(expert[:,0]+expert[:,1])*.5
        hybrid[disagreement]=probabilities[disagreement]
        np.save(args.out/'hybrid_probabilities.npy',hybrid)
        save_json(args.out/'hybrid_routing.json',{'rule':'Julia only where DINOv2 and ConvNeXt top1 differ',
                                                'routed_count':int(disagreement.sum()),'routed_ids':np.asarray(ids)[disagreement].tolist(),
                                                'uses_true_labels':False})
        # Scoring is intentionally separate; root can compare all fixed models.
    elif not args.limit and args.split=='test' and args.disagreements_only:
        hybrid=(expert[:,0]+expert[:,1])*.5
        hybrid[chosen]=probabilities
        np.save(args.out/'hybrid_probabilities.npy',hybrid)
        save_json(args.out/'hybrid_ids.json',ids)
        csv=args.out/'submission_julia_decision_hybrid.csv'
        pd.DataFrame({'Id':ids,'Label':np.asarray(CLASSES)[hybrid.argmax(1)]}).to_csv(csv,index=False)
        save_json(args.out/'hybrid_submission_receipt.json',{'count':len(ids),'routed_count':len(chosen),'csv_sha256':sha(csv),
                                                          'rule':'Julia only where DINOv2 and ConvNeXt top1 differ'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--model',type=Path,default=HERE/'julia_model')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--split',choices=['validation','test'],default='validation')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--disagreements-only',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    torch.manual_seed(42);np.random.seed(42)
    predict(args)


if __name__=='__main__':main()
