from pathlib import Path
import numpy as np
from style_features import extract_style
root=Path(__file__).resolve().parent; out=root/'artifacts'
ids=sorted(p.stem for p in (root/'test').glob('*.png'))
cache=out/'style_test.npz'
if not cache.exists():
 rows=[]
 for i,id in enumerate(ids):
  rows.append(extract_style(root/'test'/f'{id}.png'))
  if (i+1)%2000==0: print('Test style features:',i+1,flush=True)
 np.savez_compressed(cache,features=np.stack(rows),ids=np.asarray(ids,dtype=str))
