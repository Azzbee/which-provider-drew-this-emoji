from pathlib import Path
import numpy as np,pandas as pd,json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
root=Path(__file__).resolve().parent; out=root/'artifacts'
l=pd.read_csv(root/'train_labels.csv',dtype={'Id':str}); d=np.load(out/'learned_train.npz')
assert d['ids'].tolist()==l.Id.tolist()
X=d['features']; y=l.Label.to_numpy(); a,b=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=42)
results=[]
for c in [1,10,100]:
 model=make_pipeline(StandardScaler(),SVC(C=c,kernel='rbf',gamma='scale',cache_size=512))
 model.fit(X[a],y[a]); pred=model.predict(X[b]); score=accuracy_score(y[b],pred)
 results.append({'features':'learned','C':c,'accuracy':score})
 print(results[-1],flush=True)
 np.savez_compressed(out/f'learned_validation_C{c}.npz',ids=l.Id.iloc[b].to_numpy(dtype=str),predictions=pred,decision=model.decision_function(X[b]),classes=model.classes_)
(out/'learned_experiments.json').write_text(json.dumps(results,indent=2))
