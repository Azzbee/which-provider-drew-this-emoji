"""32x32 RGB copy of the 9,879 training images (alpha composited on white) + the DINOv2 branch's split."""
import json, numpy as np
from pathlib import Path
from PIL import Image
HERE = Path(__file__).resolve().parent
FUS = HERE.parent / 'learned-fusion-transformer-and-mlp/metadata'
DATA = HERE.parents[2] / 'codebase/data/extracted/train'
ids = json.load(open(FUS/'ids.json')); y = np.array(json.load(open(FUS/'labels.json')), np.int64)
tr = np.array(json.load(open(FUS/'train_indices.json'))); dv = np.array(json.load(open(FUS/'validation_indices.json')))
X = np.zeros((len(ids), 32, 32, 3), np.uint8)
for k, i in enumerate(ids):
    im = Image.open(DATA / f'{i}.png').convert('RGBA')
    bg = Image.new('RGBA', im.size, (255, 255, 255, 255)); im = Image.alpha_composite(bg, im).convert('RGB')
    X[k] = np.asarray(im.resize((32, 32), Image.Resampling.BICUBIC))
np.savez_compressed('emoji32.npz', X=X, y=y, train_idx=tr, dev_idx=dv)
print(X.shape, np.bincount(y))
