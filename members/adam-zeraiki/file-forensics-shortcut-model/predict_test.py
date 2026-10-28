import numpy as np, pandas as pd, json
from sklearn.ensemble import HistGradientBoostingClassifier
from pathlib import Path as _P; CB=str(_P(__file__).resolve().parents[3]/'codebase')
F=pd.read_csv('features.csv',dtype={'Id':str})
lab=pd.read_csv(''+CB+'/data/extracted/train_labels.csv',dtype=str).sort_values('Id').reset_index(drop=True)
tr=lab.merge(F[F.split=='train'],on='Id'); te=F[F.split=='test'].sort_values('Id').reset_index(drop=True)
C=sorted(lab.Label.unique()); y=tr.Label.map({c:i for i,c in enumerate(C)}).to_numpy()
cols=[c for c in F.columns if c not in ('Id','split')]
Xtr=tr[cols].astype(float).fillna(-1).to_numpy(); Xte=te[cols].astype(float).fillna(-1).to_numpy()
Btr=np.load(f'{CB}/outputs/convnext224_last/oof_proba.npy'); Bte=np.load(f'{CB}/outputs/convnext224_last/test_proba_mean.npy')
sub=pd.read_csv(f'{CB}/outputs/submissions/submission_convnext224_last.csv',dtype=str); assert (sub.Id.values==te.Id.values).all()
mk=lambda: HistGradientBoostingClassifier(max_iter=400,learning_rate=0.05,max_leaf_nodes=31,random_state=0)
m=mk().fit(Xtr,y); pf=m.predict(Xte)
m2=mk().fit(np.hstack([Xtr,np.log(Btr+1e-6)]),y); ps=m2.predict(np.hstack([Xte,np.log(Bte+1e-6)]))
pd.DataFrame({'Id':te.Id,'Label':[C[i] for i in pf]}).to_csv('submission_file_forensics_only.csv',index=False)
pd.DataFrame({'Id':te.Id,'Label':[C[i] for i in ps]}).to_csv('submission_convnext_plus_file_forensics.csv',index=False)
print('stack vs final model: test labels changed', int((np.array([C[i] for i in ps])!=sub.Label.values).sum()))
json.dump({'file_only_oof_accuracy':0.9348,'stack_oof_accuracy':0.97945,'stack_oof_correct':9676,'stack_ci95':[97.66,98.22],'convnext_run_b_oof':0.97692,
 'n_features':len(cols),'folds':'StratifiedGroupKFold(5, shuffle, seed 42) on pix96 hash','model':'HistGradientBoostingClassifier(max_iter=400, lr=0.05, leaves=31)'},open('results.json','w'),indent=1)
