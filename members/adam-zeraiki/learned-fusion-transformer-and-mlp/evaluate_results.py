#!/usr/bin/env python3
"""Central, post-selection comparison. Never trains or modifies a candidate.

AI-assisted experiment code. All results use the previously reused development
holdout; rankings are exploratory and are not Kaggle leaderboard scores.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, classification_report

CLASSES=['apple','facebook','google','messenger','mozilla','samsung','whatsapp']

def read(path): return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2)+'\n')
def validate(p,n):
    assert p.shape==(n,7) and np.isfinite(p).all()
    assert p.min()>=0 and p.max()<=1 and np.allclose(p.sum(1),1,atol=1e-5)
def expected_calibration_error(p,y,bins=15):
    confidence=p.max(1); correct=p.argmax(1)==y
    assignments=np.minimum((confidence*bins).astype(int),bins-1)
    return float(sum((assignments==b).mean()*abs(correct[assignments==b].mean()-confidence[assignments==b].mean())
                     for b in range(bins) if (assignments==b).any()))
def metrics(p,y,reference):
    pred=p.argmax(1); correct=pred==y; ref_correct=reference.argmax(1)==y
    return {'total':len(y),'correct':int(correct.sum()),'accuracy':float(accuracy_score(y,pred)),
      'macro_f1':float(f1_score(y,pred,labels=range(7),average='macro',zero_division=0)),
      'log_loss':float(log_loss(y,p,labels=range(7))),
      'multiclass_brier_sum':float(np.square(p-np.eye(7)[y]).sum(1).mean()),
      'ece_15_equal_width_bins':expected_calibration_error(p,y),
      'fixes_vs_reference':int((correct&~ref_correct).sum()),
      'regressions_vs_reference':int((~correct&ref_correct).sum()),
      'net_correct_vs_reference':int(correct.sum()-ref_correct.sum()),
      'prediction_disagreements_vs_reference':int((pred!=reference.argmax(1)).sum())}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--julia',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    ids=read(args.experiment/'metadata/ids.json')
    va=np.asarray(read(args.experiment/'metadata/validation_indices.json'),dtype=int)
    y=np.asarray(read(args.experiment/'metadata/labels.json'),dtype=int)[va]
    val_ids=[ids[i] for i in va]
    assert len(va)==1976 and len(set(val_ids))==1976
    dframe=pd.read_csv(args.previous/'dino_original/run_224/selection/validation_predictions.csv',dtype={'Id':str})
    cframe=pd.read_csv(args.previous/'convnext/run_224/selection/validation_predictions.csv',dtype={'Id':str})
    assert dframe.Id.tolist()==cframe.Id.tolist()==val_ids
    assert dframe.true_label.tolist()==[CLASSES[i] for i in y]
    d=dframe[[f'p_{c}' for c in CLASSES]].to_numpy(np.float32)
    c=cframe[[f'p_{c}' for c in CLASSES]].to_numpy(np.float32)
    reference=np.float32(.5)*d+np.float32(.5)*c
    assert int((reference.argmax(1)==y).sum())==1919
    candidates={'baseline_dinov2_convnext_equal':reference}
    provenance={}; seeds={}
    for feature_set in ['frozen_style','tuned_style']:
        run=args.experiment/'runs'/feature_set
        assert (run/'selection_lock.json').is_file()
        selection=read(run/'selection_summary.json')
        assert selection['selection_locked_before_refit'] and selection['refit_completed']
        assert selection['outer_metrics_calculated'] is False
        assert read(run/'labeled_ids.json')==ids
        assert read(run/'class_order.json')==CLASSES
        for family in ['mlp','transformer']:
            name=f'{feature_set}_{family}'
            path=run/family/'outer_validation_probabilities.npy'
            p=np.load(path,allow_pickle=False); validate(p,len(y)); candidates[name]=p
            per_seed=[]
            for record in selection['families'][family]['refit_per_seed']:
                seed=record['seed']
                sp=np.load(run/family/f'seed_{seed}'/'refit/labeled_probabilities.npy')[va]
                validate(sp,len(y))
                per_seed.append({'seed':seed,**metrics(sp,y,reference)})
            seeds[name]=per_seed
            provenance[name]={'probability_sha256':sha(path),
                'selected_family_by_internal_validation':selection['selected_family_by_internal_validation'],
                'is_internal_selected_family':selection['selected_family_by_internal_validation']==family,
                'selection_lock_sha256':sha(run/'selection_lock.json'),
                'encoder_exposure':('No project-label fine-tuning of encoders' if feature_set=='frozen_style' else
                                    'Encoders saw all7903 training labels and were checkpoint-selected on reused1976'),
                'trainable_parameters_per_seed':selection['families'][family]['parameter_counts_per_seed']['total_trainable']}
    jdir=args.julia/'validation'
    assert (jdir/'COMPLETE.json').is_file(), 'Julia verification must finish before central evaluation'
    assert bool(np.load(jdir/'complete.npy').all())
    assert read(jdir/'ids.json')==val_ids
    jp=np.load(jdir/'probabilities.npy'); hp=np.load(jdir/'hybrid_probabilities.npy')
    validate(jp,len(y)); validate(hp,len(y))
    route=d.argmax(1)!=c.argmax(1)
    assert int(route.sum())==112
    assert np.array_equal(hp[route],jp[route]) and np.array_equal(hp[~route],reference[~route])
    candidates['julia1_decision']=jp
    candidates['julia1_disagreement_hybrid']=hp
    provenance['julia1_decision']={'probability_sha256':sha(jdir/'probabilities.npy'),
        'model':'SupersonicLabs/Julia-1','task_training':'None; frozen general decision model with fixed prompt',
        'input':'Typed7waychoice on compactDINO/Conv/style score state; noimages,IDs,groundtruthlabels',
        'not_a_jev_api_run':True}
    provenance['julia1_disagreement_hybrid']={'routing':'Use Julia only when DINO and Conv top1 disagree; otherwise current equal average',
                                           'routed_validation_rows':112,'probability_sha256':sha(jdir/'hybrid_probabilities.npy')}
    rows=[]; detailed={}
    for name,p in candidates.items():
        row={'candidate':name,**metrics(p,y,reference)}
        if name in seeds:
            row['seed_accuracy_mean']=float(np.mean([s['accuracy'] for s in seeds[name]]))
            row['seed_accuracy_sample_sd']=float(np.std([s['accuracy'] for s in seeds[name]],ddof=1))
        row['passes_complexity_threshold']=row['correct']>=1927
        rows.append(row)
        pred=p.argmax(1)
        detailed[name]={**row,'confusion_matrix':confusion_matrix(y,pred,labels=range(7)).tolist(),
            'classification_report':classification_report(y,pred,labels=range(7),target_names=CLASSES,output_dict=True,zero_division=0),
            'per_seed':seeds.get(name,[]),'provenance':provenance.get(name,{})}
        frame=pd.DataFrame({'Id':val_ids,'true_label':[CLASSES[i] for i in y],
                            'predicted_label':[CLASSES[i] for i in pred]})
        for i,label in enumerate(CLASSES): frame[f'p_{label}']=p[:,i]
        frame.to_csv(args.out/f'{name}_validation_predictions.csv',index=False)
    frame=pd.DataFrame(rows).sort_values(['correct','log_loss'],ascending=[False,True])
    frame.to_csv(args.out/'comparison.csv',index=False)
    eligible=[r for r in rows if r['candidate']!='baseline_dinov2_convnext_equal' and r['correct']>=1927]
    recommended=(max(eligible,key=lambda r:(r['correct'],-r['log_loss']))['candidate'] if eligible else 'baseline_dinov2_convnext_equal')
    fusion_rows=[r for r in rows if r['candidate'].startswith(('frozen_style_','tuned_style_'))]
    best_fusion=max(fusion_rows,key=lambda r:(r['correct'],-r['log_loss']))['candidate']
    result={'evaluation':'Reused development holdout; exploratory modelcomparison, not untouchedtestorKaggle',
        'classes':CLASSES,'validation_rows':1976,'reference_correct':1919,
        'recommended_candidate':recommended,'best_observed_fusion_candidate':best_fusion,
        'complexity_threshold':1927,'details':detailed,
        'notes':['All modelrecipes and selectedepochs fixed before this scoring run.',
                 'Outer holdout already influenced prior models; no unbiased generalization claim.',
                 'Jev API was not run; Julia1 is an independently published open decision model.',
                 'No new Kaggle upload and no hidden testlabels used.']}
    write(args.out/'comparison.json',result)
    print(frame[['candidate','correct','accuracy','fixes_vs_reference','regressions_vs_reference','passes_complexity_threshold']].to_string(index=False))
    print(json.dumps({'recommended_candidate':recommended,'best_observed_fusion_candidate':best_fusion}))

if __name__=='__main__': main()
