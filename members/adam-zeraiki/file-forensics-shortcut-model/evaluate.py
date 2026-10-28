import numpy as np, pandas as pd, json
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from pathlib import Path as _P; CB=str(_P(__file__).resolve().parents[3]/'codebase')
F=pd.read_csv('features.csv',dtype={'Id':str})
lab=pd.read_csv(''+CB+'/data/extracted/train_labels.csv',dtype=str)
meta=pd.read_csv(f'{CB}/outputs/eda/metadata_train.csv',dtype={'Id':str})[['Id','pix96_sha256']]
tr=F[F.split=='train'].merge(lab,on='Id').merge(meta,on='Id').sort_values('Id').reset_index(drop=True)
C=sorted(tr.Label.unique()); y=tr.Label.map({c:i for i,c in enumerate(C)}).to_numpy()
cols=[c for c in F.columns if c not in ('Id','split')]
X=tr[cols].astype(float).fillna(-1).to_numpy()
groups=tr.pix96_sha256.to_numpy()
skf=StratifiedGroupKFold(5,shuffle=True,random_state=42)
def oof(make,X):
    P=np.zeros((len(y),7))
    for k,(a,b) in enumerate(skf.split(X,y,groups)):
        m=make(); m.fit(X[a],y[a]); P[b]=m.predict_proba(X[b])
    return P
res={}
for name,make in [('hgb',lambda: HistGradientBoostingClassifier(max_iter=400,learning_rate=0.06,max_leaf_nodes=31,random_state=0)),
                  ('rf',lambda: RandomForestClassifier(500,n_jobs=-1,random_state=0))]:
    P=oof(make,X); acc=(P.argmax(1)==y).mean(); res[name]=acc; np.save(f'oof_{name}.npy',P); print(name,'file-only OOF acc',round(acc*100,2))
# importance by drop-group
json.dump({k:float(v) for k,v in res.items()},open('oof_file_only.json','w'))
