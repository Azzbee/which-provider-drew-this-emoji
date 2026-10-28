"""Epoch-wise and model-wise double descent on the emoji data (after Nakkiran et al., 2020).

Pre-activation ResNet18 whose four stages have k, 2k, 4k, 8k channels (k=64 is the standard
ResNet18), trained from scratch on 32x32 images with Adam (lr 1e-4), batch 128, random crop
and horizontal flip, for a fixed number of epochs with no early stopping and no schedule.
A fraction `noise` of the TRAINING labels is replaced by a different random class (fixed seed);
development labels are always clean. Train: the DINOv2 branch's 7,903 images; evaluation:
its 1,976 development images (exact duplicates never straddle the split).

Every epoch logs: train loss and accuracy on the (noisy) labels, accuracy on the clean and on the
corrupted subsets of the training set, and development loss and accuracy.
"""
import argparse, json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument('--width', type=int, default=64)
ap.add_argument('--noise', type=float, default=0.0)
ap.add_argument('--epochs', type=int, default=2000)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--data', default='emoji32.npz')
ap.add_argument('--out', default='logs')
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = 'cuda'
torch.backends.cudnn.benchmark = True

d = np.load(a.data)
X = torch.tensor(d['X']).permute(0, 3, 1, 2).float().div(255)
mean, std = X.mean((0, 2, 3), keepdim=True), X.std((0, 2, 3), keepdim=True)
X = ((X - mean) / std).to(dev)
y = torch.tensor(d['y']).to(dev)
tr, dv = torch.tensor(d['train_idx']).to(dev), torch.tensor(d['dev_idx']).to(dev)
ytr_clean = y[tr].clone(); ytr = ytr_clean.clone()
rng = np.random.default_rng(1234)                      # same corrupted set for every width
n_bad = int(round(a.noise * len(tr)))
bad = torch.tensor(rng.choice(len(tr), n_bad, replace=False), dtype=torch.long, device=dev)
if n_bad:
    shift = torch.tensor(rng.integers(1, 7, n_bad), device=dev)
    ytr[bad] = (ytr[bad] + shift) % 7
is_bad = torch.zeros(len(tr), dtype=torch.bool, device=dev); is_bad[bad] = True
Xtr, Xdv, ydv = X[tr], X[dv], y[dv]


class Block(nn.Module):
    def __init__(s, i, o, stride):
        super().__init__()
        s.bn1, s.c1 = nn.BatchNorm2d(i), nn.Conv2d(i, o, 3, stride, 1, bias=False)
        s.bn2, s.c2 = nn.BatchNorm2d(o), nn.Conv2d(o, o, 3, 1, 1, bias=False)
        s.sc = None if stride == 1 and i == o else nn.Conv2d(i, o, 1, stride, bias=False)
    def forward(s, x):
        o = F.relu(s.bn1(x)); sc = x if s.sc is None else s.sc(o)
        o = s.c1(o); o = s.c2(F.relu(s.bn2(o)))
        return o + sc


class PreActResNet18(nn.Module):
    def __init__(s, k, n=7):
        super().__init__()
        s.stem = nn.Conv2d(3, k, 3, 1, 1, bias=False)
        L, i = [], k
        for j, o in enumerate([k, 2 * k, 4 * k, 8 * k]):
            for b in range(2):
                L.append(Block(i, o, 2 if (b == 0 and j > 0) else 1)); i = o
        s.layers = nn.Sequential(*L); s.bn = nn.BatchNorm2d(i); s.fc = nn.Linear(i, n)
    def forward(s, x):
        x = s.layers(s.stem(x)); x = F.relu(s.bn(x))
        return s.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


model = PreActResNet18(a.width).to(dev).to(memory_format=torch.channels_last)
params = sum(p.numel() for p in model.parameters())
opt = torch.optim.Adam(model.parameters(), lr=1e-4)


def augment(x):
    flip = torch.rand(x.size(0), device=dev) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    p = F.pad(x, (4, 4, 4, 4), mode='replicate')
    i, j = np.random.randint(0, 9, 2)                    # one crop offset per batch (fast)
    return p[:, :, i:i + 32, j:j + 32]


@torch.no_grad()
def evaluate(Xe, ye):
    model.eval(); L, P = 0.0, []
    for s in range(0, len(Xe), 1024):
        with torch.autocast('cuda', torch.bfloat16):
            out = model(Xe[s:s + 1024].contiguous(memory_format=torch.channels_last)).float()
        L += F.cross_entropy(out, ye[s:s + 1024], reduction='sum').item(); P.append(out.argmax(1))
    return L / len(Xe), torch.cat(P)


name = f'w{a.width}_n{int(a.noise * 100)}_s{a.seed}'
import os; os.makedirs(a.out, exist_ok=True)
log = open(f'{a.out}/{name}.jsonl', 'w')
log.write(json.dumps({'config': vars(a), 'params': params, 'n_corrupted': n_bad}) + '\n')
t0 = time.time()
for ep in range(1, a.epochs + 1):
    model.train(); perm = torch.randperm(len(tr), device=dev)
    for s in range(0, len(tr), 128):
        b = perm[s:s + 128]
        with torch.autocast('cuda', torch.bfloat16):
            loss = F.cross_entropy(model(augment(Xtr[b]).contiguous(memory_format=torch.channels_last)), ytr[b])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    trl, trp = evaluate(Xtr, ytr)
    dvl, dvp = evaluate(Xdv, ydv)
    rec = {'epoch': ep, 'train_loss': trl, 'train_acc': (trp == ytr).float().mean().item(),
           'train_acc_clean_subset': (trp[~is_bad] == ytr[~is_bad]).float().mean().item(),
           'memorised_corrupted': (trp[is_bad] == ytr[is_bad]).float().mean().item() if n_bad else None,
           'dev_loss': dvl, 'dev_acc': (dvp == ydv).float().mean().item(), 'sec': round(time.time() - t0, 1)}
    log.write(json.dumps(rec) + '\n'); log.flush()
    if ep % 50 == 0 or ep == 1:
        print(name, rec, flush=True)
torch.save({'state_dict': model.state_dict(), 'config': vars(a)}, f'{a.out}/{name}_final.pt')
