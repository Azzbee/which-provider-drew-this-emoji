from pathlib import Path
import numpy as np,pandas as pd,json,time
from style_features import extract_style
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
root=Path(__file__).resolve().parent
labels=pd.read_csv(root/'train_labels.csv',dtype={'Id':str})
out=root/'artifacts'; out.mkdir(exist_ok=True)
cache=out/'style_train.npz'
if cache.exists():
 d=np.load(cache); assert d['ids'].tolist()==labels.Id.tolist(); X=d['features']
else:
 rows=[]
 for i,id in enumerate(labels.Id):
  rows.append(extract_style(root/'train'/f'{id}.png'))
  if (i+1)%1000==0:print('Style features:',i+1,flush=True)
 X=np.stack(rows); np.savez_compressed(cache,features=X,ids=labels.Id.to_numpy(dtype=str))
y=labels.Label.to_numpy(); a,b=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=42)
scaler=StandardScaler(); A=scaler.fit_transform(X[a]); B=scaler.transform(X[b]); results=[]
for c in [1,10,100]:
 model=SVC(C=c,kernel='rbf',gamma='scale')
 model.fit(A,y[a]); pred=model.predict(B); score=accuracy_score(y[b],pred)
 results.append({'C':c,'accuracy':score}); print(results[-1],flush=True)
(out/'style_experiments.json').write_text(json.dumps(results,indent=2))
