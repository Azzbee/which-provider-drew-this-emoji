"""Test-set file for the three-expert combination that can be built for the test images.

DINOv2's test probabilities were not saved (only its predicted labels, in the DINOv2 branch's
submission file), so DINOv2 enters as a smoothed one-hot vector (0.9 on its label, the rest
spread evenly). ConvNeXt is the final model (run B, mean of the 5 fold models, flip TTA).
The file-byte model is refitted on all 9,879 training images. Combination: geometric mean.
On the 1,976 development images this exact recipe scores 98.08% (run B alone: 97.87%).
"""
import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
ROOT = Path('/home/claude/emoji'); ADAM = Path('/home/claude/which-provider-drew-this-emoji/members/adam-zeraiki')
CLASSES = ['apple', 'facebook', 'google', 'messenger', 'mozilla', 'samsung', 'whatsapp']
feat = pd.read_csv(ROOT / 'forensics/features.csv', dtype={'Id': str})
cols = [c for c in feat.columns if c not in ('Id', 'split')]
lab = pd.read_csv(ROOT / 'data/extracted/train_labels.csv', dtype=str).sort_values('Id')
tr = feat[feat.split == 'train'].set_index('Id').loc[lab.Id]
te = feat[feat.split == 'test'].sort_values('Id').set_index('Id')
y = lab.Label.map(CLASSES.index).to_numpy()
hgb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=31, random_state=0)
hgb.fit(tr[cols].astype(float).fillna(-1).to_numpy(), y)
F = hgb.predict_proba(te[cols].astype(float).fillna(-1).to_numpy())
R = np.load(ROOT / 'codebase/outputs/convnext224_last/test_proba_mean.npy')
sub_b = pd.read_csv(ROOT / 'codebase/outputs/convnext224_last/submission.csv', dtype={'Id': str})
assert sub_b.Id.tolist() == te.index.tolist() and (np.array(CLASSES)[R.argmax(1)] == sub_b.Label).all()
dino = pd.read_csv(ADAM / 'learned-fusion-transformer-and-mlp/base_models/dino_original/run_224/submission_dinov2_finetuned.csv', dtype={'Id': str}).set_index('Id').loc[te.index]
D = np.full((len(te), 7), 0.1 / 6); D[np.arange(len(te)), dino.Label.map(CLASSES.index).to_numpy()] = 0.9
P = np.exp(np.log(D) + np.log(R + 1e-6) + np.log(F + 1e-6))
pred = np.array(CLASSES)[P.argmax(1)]
out = pd.DataFrame({'Id': te.index, 'Label': pred})
out.to_csv('submission_dinov2_convnext_filebytes_geomean.csv', index=False)
print('rows', len(out), '| labels changed vs final model (run B):', int((pred != sub_b.Label).sum()))
print(out.Label.value_counts().to_dict())
