"""Explainable CPU baseline; run with: python baseline.py."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
OUT = ROOT / 'artifacts'
OUT.mkdir(exist_ok=True)

def features(path):
    with Image.open(path) as im:
        rgba = im.convert('RGBA')
        white = Image.new('RGBA', rgba.size, 'white')
        rgb = np.asarray(Image.alpha_composite(white, rgba).convert('RGB').resize((24,24)), dtype=np.float32)/255
        alpha = np.asarray(rgba.getchannel('A').resize((24,24)),dtype=np.float32)/255
        hist = np.concatenate([np.histogram(rgb[:,:,c], bins=16, range=(0,1),density=False)[0]/576 for c in range(3)])
        return np.concatenate([rgb.ravel(),alpha.ravel(),hist]).astype('float32')

def image_map(folder):
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg'})
    result = {p.stem:p for p in paths}
    assert len(result)==len(paths), 'Duplicate image IDs'
    return result

def submission(ids, predictions, filename):
    result = pd.DataFrame({'Id':ids,'Label':predictions})
    sample = ROOT/'sample_submission.csv'
    if sample.exists():
        template = pd.read_csv(sample,dtype={'Id':str})
        assert list(template.columns)==['Id','Label']
        assert set(template.Id)==set(result.Id)
        result=template[['Id']].merge(result,on='Id',validate='one_to_one')
    assert result.Id.is_unique and result.Label.notna().all()
    assert set(result.Label)<=set(pd.read_csv(ROOT/'train_labels.csv').Label)
    result.to_csv(ROOT/filename,index=False)
    return result

def main():
    labels=pd.read_csv(ROOT/'train_labels.csv',dtype={'Id':str})
    assert labels.Id.is_unique and labels.notna().all().all()
    train=image_map(ROOT/'train'); test=image_map(ROOT/'test')
    assert set(labels.Id)==set(train)
    print('Extracting features...',flush=True)
    X=np.stack([features(train[i]) for i in labels.Id]); y=labels.Label.to_numpy()
    a,b=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=42)
    model=ExtraTreesClassifier(n_estimators=400,max_features=.7,min_samples_leaf=1,n_jobs=-1,random_state=42)
    model.fit(X[a],y[a]); pred=model.predict(X[b]); score=accuracy_score(y[b],pred)
    report=classification_report(y[b],pred,output_dict=True,zero_division=0)
    (OUT/'baseline_metrics.json').write_text(json.dumps({'validation_accuracy':score,'split_seed':42,'report':report},indent=2))
    pd.DataFrame(confusion_matrix(y[b],pred,labels=model.classes_),index=model.classes_,columns=model.classes_).to_csv(OUT/'baseline_confusion_matrix.csv')
    pd.DataFrame({'Id':labels.Id.iloc[b].to_numpy(),'true':y[b],'predicted':pred}).to_csv(OUT/'baseline_validation_predictions.csv',index=False)
    print(f'Validation accuracy: {score:.4%}',flush=True)
    model.fit(X,y)
    ids=sorted(test)
    prediction=model.predict(np.stack([features(test[i]) for i in ids]))
    submission(ids,prediction,'submission_baseline.csv')
    print('Saved submission_baseline.csv',flush=True)
if __name__=='__main__':
    main()
