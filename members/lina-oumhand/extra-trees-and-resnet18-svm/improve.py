"""Train and export an emoji-provider SVM. See method-README.md for reproducibility.

Feature extraction never uses image IDs as predictors or reads file metadata.
Preprocessing is fit on the training fold, then refit on all labels for submission.
"""
from pathlib import Path
import argparse
import json
import platform
import hashlib
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from style_features import extract_style


def cached_style(root, split, ids):
    cache=root/'artifacts'/f'style_{split}.npz'
    if cache.exists():
        data=np.load(cache)
        assert data['ids'].tolist()==list(ids), 'Cached style image IDs do not match'
        return data['features']
    rows=[]
    for i,id in enumerate(ids):
        rows.append(extract_style(root/split/f'{id}.png'))
        if (i+1)%1000==0: print(f'{split}: style features {i+1}/{len(ids)}',flush=True)
    result=np.stack(rows)
    assert np.isfinite(result).all()
    np.savez_compressed(cache,features=result,ids=np.asarray(ids,dtype=str))
    return result


def features(root,split,ids,kind):
    blocks=[]
    if kind in ('style','combined'):blocks.append(cached_style(root,split,ids))
    if kind in ('learned','combined'):
        from learned_features import extract_learned
        blocks.append(extract_learned(root,split,ids))
    # Keep a fixed order: handcrafted descriptors, then ResNet descriptors.
    # Permuting columns changes neither standardized RBF distances nor predictions.
    result=np.concatenate(blocks,axis=1)
    assert result.shape[0]==len(ids) and np.isfinite(result).all()
    return result


def estimator(c):
    return make_pipeline(StandardScaler(),SVC(C=c,kernel='rbf',gamma='scale',cache_size=512))


def export_submission(root,ids,predictions,classes):
    frame=pd.DataFrame({'Id':ids,'Label':predictions})
    sample_path=root/'sample_submission.csv'
    if sample_path.exists():
        sample=pd.read_csv(sample_path,dtype={'Id':str})
        assert list(sample.columns)==['Id','Label'], 'Check official submission columns'
        assert sample.Id.is_unique and set(sample.Id)==set(ids)
        frame=sample[['Id']].merge(frame,on='Id',validate='one_to_one')
    assert len(frame)==len(ids) and frame.Id.is_unique and frame.notna().all().all()
    assert set(frame.Label)<=set(classes)
    frame.to_csv(root/'submission_improved.csv',index=False)
    # Preserve the original submission before promoting the new one.
    previous=root/'submission.csv'; backup=root/'submission_before_improvement.csv'
    if previous.exists() and not backup.exists(): backup.write_bytes(previous.read_bytes())
    frame.to_csv(previous,index=False)
    return frame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--features',choices=['style','learned','combined'],default='combined')
    parser.add_argument('--C',type=float,default=10.)
    parser.add_argument('--validation-seed',type=int,default=42)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args();root=args.data_dir.resolve();out=root/'artifacts';out.mkdir(exist_ok=True)
    labels=pd.read_csv(root/'train_labels.csv',dtype={'Id':str})
    assert list(labels.columns)==['Id','Label'] and labels.notna().all().all() and labels.Id.is_unique
    train_ids=sorted(p.stem for p in (root/'train').glob('*.png'))
    test_ids=sorted(p.stem for p in (root/'test').glob('*.png'))
    assert set(labels.Id)==set(train_ids), 'Missing or extra training images'
    assert test_ids and len(test_ids)==len(set(test_ids)) and not set(train_ids)&set(test_ids)
    X=features(root,'train',labels.Id.tolist(),args.features);y=labels.Label.to_numpy()
    a,b=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=args.validation_seed)
    print(f'Train: {len(a)}; validation: {len(b)}; descriptors: {X.shape[1]}',flush=True)
    model=estimator(args.C)
    model.fit(X[a],y[a]);prediction=model.predict(X[b]);score=float(accuracy_score(y[b],prediction))
    classes=model.classes_.tolist()
    metrics={'validation_accuracy':score,'validation_seed':args.validation_seed,'validation_fraction':.2,
             'features':args.features,'C':args.C,'gamma':'scale','descriptor_count':X.shape[1],
             'classes':classes,'python':platform.python_version(),
             'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','pillow','scikit-learn','scipy']},
             'labels_sha256':hashlib.sha256((root/'train_labels.csv').read_bytes()).hexdigest(),
             'report':classification_report(y[b],prediction,output_dict=True,zero_division=0)}
    if args.features!='style':
        metrics['pretrained_weights']='torchvision ResNet18 IMAGENET1K_V1 (frozen)'
        for package in ['torch','torchvision']:metrics['packages'][package]=importlib.metadata.version(package)
    suffix='' if args.validation_seed==42 else f'_seed{args.validation_seed}'
    (out/f'improved_metrics{suffix}.json').write_text(json.dumps(metrics,indent=2))
    pd.DataFrame({'Id':labels.Id.iloc[b].to_numpy(),'true':y[b],'predicted':prediction}).to_csv(out/f'improved_validation_predictions{suffix}.csv',index=False)
    pd.DataFrame(confusion_matrix(y[b],prediction,labels=classes),index=classes,columns=classes).to_csv(out/f'improved_confusion_matrix{suffix}.csv')
    print(f'Validation accuracy: {score:.4%}',flush=True)
    if args.validate_only:return
    print('Refitting selected model on all training labels...',flush=True)
    model.fit(X,y)
    joblib.dump({'pipeline':model,'features':args.features,'classes':classes,'C':args.C},out/'improved_model.joblib',compress=3)
    test_X=features(root,'test',test_ids,args.features)
    result=export_submission(root,test_ids,model.predict(test_X),classes)
    pd.Series(result.Label.value_counts()).to_csv(out/'improved_test_label_counts.csv')
    print(f'Saved submission_improved.csv and submission.csv: {len(result)} rows',flush=True)

if __name__=='__main__':main()
