import numpy as np, pandas as pd, json
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from pathlib import Path as _P; CB=str(_P(__file__).resolve().parents[3]/'codebase')
F=pd.read_csv('features.csv',dtype={'Id':str})
lab=pd.read_csv(''+CB+'/data/extracted/train_labels.csv',dtype=str).sort_values('Id').reset_index(drop=True)
meta=pd.read_csv(f'{CB}/outputs/eda/metadata_train.csv',dtype={'Id':str})[['Id','pix96_sha256']]
tr=lab.merge(F[F.split=='train'],on='Id').merge(meta,on='Id')
assert (tr.Id.values==lab.Id.values).all()
C=sorted(lab.Label.unique()); y=tr.Label.map({c:i for i,c in enumerate(C)}).to_numpy()
cols=[c for c in F.columns if c not in ('Id','split')]
Xf=tr[cols].astype(float).fillna(-1).to_numpy()
B=np.load(f'{CB}/outputs/convnext224_last/oof_proba.npy'); print('convnext oof',(B.argmax(1)==y).mean())
groups=tr.pix96_sha256.to_numpy()
skf=StratifiedGroupKFold(5,shuffle=True,random_state=42)
X=np.hstack([Xf,np.log(B+1e-6)])
P=np.zeros((len(y),7)); fold=np.zeros(len(y),int)
for k,(a,b) in enumerate(skf.split(X,y,groups)):
    m=HistGradientBoostingClassifier(max_iter=400,learning_rate=0.05,max_leaf_nodes=31,random_state=0); m.fit(X[a],y[a]); P[b]=m.predict_proba(X[b]); fold[b]=k
acc=(P.argmax(1)==y).mean(); print('pixels+file stack OOF acc',round(acc*100,3), (P.argmax(1)==y).sum())
pf=[(P.argmax(1)==y)[fold==k].mean() for k in range(5)]; print('folds',np.round(np.array(pf)*100,2))
rng=np.random.default_rng(0); ok=P.argmax(1)==y; bs=[ok[rng.integers(0,len(y),len(y))].mean() for _ in range(2000)]; print('CI',np.round(np.percentile(bs,[2.5,97.5])*100,2))
np.save('oof_stack.npy',P)
