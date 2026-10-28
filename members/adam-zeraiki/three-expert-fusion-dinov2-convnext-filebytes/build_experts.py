"""Collect the three experts' probabilities on the DINOv2 branch's 1,976-image development split.

D  DINOv2 ViT-S/14, partially fine-tuned on the 7,903 training images (saved by the DINOv2 branch)
C  ConvNeXt-Tiny, partially fine-tuned on the same 7,903 images (saved by the same branch)
R  ConvNeXt-Tiny run B (final model), out-of-fold probabilities from its own 5-fold CV
F  file-byte model: HistGradientBoosting on the 64 PNG byte features, trained here on the
   same 7,903 training images only, so it has never seen a development image.
"""
import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path('/home/claude/emoji')
ADAM = Path('/home/claude/which-provider-drew-this-emoji/members/adam-zeraiki')
FUS = ADAM / 'learned-fusion-transformer-and-mlp'
OUT = ROOT / 'fusion3'
CLASSES = json.load(open(FUS / 'metadata/class_order.json'))

ids = json.load(open(FUS / 'metadata/ids.json'))
labels = json.load(open(FUS / 'metadata/labels.json'))
groups = np.array(json.load(open(FUS / 'metadata/groups.json')))
tr_idx = np.array(json.load(open(FUS / 'metadata/train_indices.json')))
dv_idx = np.array(json.load(open(FUS / 'metadata/validation_indices.json')))
y_all = np.array([CLASSES.index(l) if isinstance(l, str) else int(l) for l in labels])
dev_ids = [ids[i] for i in dv_idx]


def load_probs(path):
    df = pd.read_csv(path, dtype={'Id': str}).set_index('Id').loc[dev_ids]
    assert (df.true_label.map(CLASSES.index).to_numpy() == y_all[dv_idx]).all()
    return df[[f'p_{c}' for c in CLASSES]].to_numpy(float)

D = load_probs(FUS / 'base_models/dino_original/run_224/selection/validation_predictions.csv')
C = load_probs(FUS / 'base_models/convnext/run_224/selection/validation_predictions.csv')

# run B out-of-fold probabilities are in sorted-Id order over all 9,879 training images
lab = pd.read_csv(ROOT / 'data/extracted/train_labels.csv', dtype=str).sort_values('Id').reset_index(drop=True)
assert lab.Id.tolist() == ids
R = np.load(ROOT / 'codebase/outputs/convnext224_last/oof_proba.npy')[dv_idx]

# file-byte expert trained on the branch's training split only
feat = pd.read_csv(ROOT / 'forensics/features.csv', dtype={'Id': str})
cols = [c for c in feat.columns if c not in ('Id', 'split')]
ftr = feat[feat.split == 'train'].set_index('Id').loc[ids]
X = ftr[cols].astype(float).fillna(-1).to_numpy()
hgb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=31, random_state=0)
hgb.fit(X[tr_idx], y_all[tr_idx])
F = hgb.predict_proba(X[dv_idx])

y = y_all[dv_idx]
np.savez(OUT / 'experts_dev.npz', D=D, C=C, R=R, F=F, y=y, groups=groups[dv_idx], ids=np.array(dev_ids))
for k, P in dict(D=D, C=C, R=R, F=F).items():
    print(k, f'{(P.argmax(1) == y).mean() * 100:.2f}%', int((P.argmax(1) == y).sum()))
