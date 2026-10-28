"""Every way we tried to combine DINOv2, ConvNeXt and the file-byte model.

Evaluated on the 1,976 development images. Fixed rules need no training. Learned combiners
are trained and scored by grouped 5-fold cross-validation inside the development split
(exact-duplicate groups never straddle folds), repeated over 5 seeds; the mean is reported.
Jev rows replay the 324 cached Jev answers from 18 October (no new API calls).
"""
import json, itertools, numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).resolve().parent
JEV = Path('/home/claude/which-provider-drew-this-emoji/members/adam-zeraiki/jev-decision-model/prepared/results/development')
CLASSES = ['apple', 'facebook', 'google', 'messenger', 'mozilla', 'samsung', 'whatsapp']
E = np.load(HERE / 'experts_dev.npz')
y, groups, ids = E['y'], E['groups'], E['ids']
N = len(y)
eps = 1e-6
SEEDS = [0, 1, 2, 3, 4]


def onehot(P):
    return np.eye(7)[P.argmax(1)]


def agreement(Ps):
    a = [P.argmax(1) for P in Ps]
    f = [(a[i] == a[j]).astype(float) for i, j in itertools.combinations(range(len(a)), 2)]
    f.append(np.array([len(set(t)) for t in zip(*a)], float))
    f += [P.max(1) for P in Ps]
    return np.column_stack(f)


# ---------- Jev cache: stage 1 = Jev as an image expert, stage 2 = Jev final decision ----------
def jev_probs(stage):
    out = {}
    for f in JEV.glob('*.json'):
        d = json.load(open(f))
        if d.get(stage) and d[stage].get('response'):
            p = d[stage]['response']['answers']['provider']['probabilities']
            v = np.array([p[c] for c in CLASSES], float)
            out[d['id']] = v / v.sum()
    return out

J1, J2 = jev_probs('expert'), jev_probs('final')


def fixed_rules(Ps, names):
    D, C, F = Ps
    a = [P.argmax(1) for P in Ps]
    avg2 = (D + C) / 2
    avg3 = (D + C + F) / 3
    geo = np.exp(np.log(D + eps) + np.log(C + eps) + np.log(F + eps))
    votes = onehot(D) + onehot(C) + onehot(F)
    vote = np.where(votes.max(1) >= 2, votes.argmax(1), avg3.argmax(1))
    dis = a[0] != a[1]
    # F breaks the tie between the two image models' choices
    tie = np.where(F[np.arange(N), a[0]] >= F[np.arange(N), a[1]], a[0], a[1])
    router_f = np.where(dis, tie, a[0])
    router_avg3 = np.where(dis, avg3.argmax(1), a[0])
    oracle_pair = np.where(dis & ((a[0] == y) | (a[1] == y)), y, avg2.argmax(1))
    return {
        f'{names[0]} + {names[1]} equal average (reference)': avg2.argmax(1),
        'All three, equal average': avg3.argmax(1),
        'All three, geometric mean': geo.argmax(1),
        'All three, majority vote on one-hot labels (tie: average)': vote,
        f'Agreement router: keep if {names[0]} = {names[1]}, else file-byte model picks between their two answers': router_f,
        f'Agreement router: keep if {names[0]} = {names[1]}, else three-way average': router_avg3,
        f'Oracle: always right when {names[0]} or {names[1]} is right (upper bound, not a model)': oracle_pair,
    }


def cv_predict(make_X, make_model, seed):
    X = make_X()
    pred = np.zeros(N, int)
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
        m = make_model(); m.fit(X[a], y[a]); pred[b] = m.predict(X[b])
    return pred


class WeightedAverage:
    def __init__(self, Ps): self.Ps = Ps
    def fit(self, idx, yy):
        best = (-1, None)
        for w0 in np.arange(0, 1.01, 0.05):
            for w1 in np.arange(0, 1.01 - w0, 0.05):
                w = (w0, w1, 1 - w0 - w1)
                acc = (sum(wi * P[idx] for wi, P in zip(w, self.Ps)).argmax(1) == yy).mean()
                if acc > best[0] + 1e-12: best = (acc, w)
        self.w = best[1]; return self
    def predict(self, idx):
        return sum(wi * P[idx] for wi, P in zip(self.w, self.Ps)).argmax(1)


def learned(Ps, seed):
    logp = np.column_stack([np.log(P + eps) for P in Ps])
    oh = np.column_stack([onehot(P) for P in Ps])
    ag = agreement(Ps)
    lr = lambda C=1.0: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
    res = {}
    # weighted average (weights chosen on the training folds)
    pred = np.zeros(N, int)
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(logp, y, groups):
        wa = WeightedAverage(Ps).fit(a, y[a]); pred[b] = wa.predict(b)
    res['Weighted average, weights learned on the training folds'] = pred
    res['Logistic regression on the 21 log-probabilities (stacking)'] = cv_predict(lambda: logp, lambda: lr(0.1), seed)
    res['Logistic regression on one-hot labels only (21 binary inputs)'] = cv_predict(lambda: oh, lambda: lr(1.0), seed)
    res['Logistic regression on one-hot labels + agreement features'] = cv_predict(lambda: np.hstack([oh, ag]), lambda: lr(1.0), seed)
    res['Logistic regression on log-probabilities + agreement features'] = cv_predict(lambda: np.hstack([logp, ag]), lambda: lr(0.1), seed)
    res['Gradient boosting on log-probabilities + agreement features'] = cv_predict(
        lambda: np.hstack([logp, ag]),
        lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed), seed)
    return res


def jev_rows(Ps, names):
    """Jev answers exist only for the 112 images where DINOv2 and the branch ConvNeXt disagree."""
    D, C, F = Ps
    a = [P.argmax(1) for P in Ps]
    avg2 = ((D + C) / 2).argmax(1)
    dis = np.where(a[0] != a[1])[0]
    have = [i for i in dis if ids[i] in J1 and ids[i] in J2]
    assert len(have) == len(dis) == 112, (len(have), len(dis))
    j1 = np.array([J1[ids[i]].argmax() for i in dis])
    j2 = np.array([J2[ids[i]].argmax() for i in dis])
    out = {}
    p = avg2.copy(); p[dis] = j2
    out['Jev final decision on the 112 disagreements (cached, 18 Oct)'] = p
    # four voters on the disagreements: DINOv2, ConvNeXt, file-byte, Jev-as-image-expert
    p = avg2.copy()
    for k, i in enumerate(dis):
        v = np.zeros(7)
        for c in (a[0][i], a[1][i], a[2][i], j1[k]): v[c] += 1
        top = np.flatnonzero(v == v.max())
        p[i] = top[0] if len(top) == 1 else max(top, key=lambda c: F[i, c] + D[i, c] + C[i, c])
    out['Majority vote of DINOv2, ConvNeXt, file-byte and Jev-image on the 112 disagreements'] = p
    # Jev final decision only when the file-byte model sides with neither image model
    p = avg2.copy()
    for k, i in enumerate(dis):
        if a[2][i] == a[0][i] or a[2][i] == a[1][i]: p[i] = a[2][i]
        else: p[i] = j2[k]
    out['File-byte model breaks the tie; Jev decides when it sides with neither'] = p
    return out


def summarise(name_set, Ps, names, with_jev):
    rows = []
    def add(name, preds_by_seed, kind):
        accs = [(p == y).mean() for p in preds_by_seed]
        ref = rows[0]['pred'] if rows else preds_by_seed[0]
        p0 = preds_by_seed[0]
        rows.append(dict(name=name, kind=kind, correct=int(round(np.mean([(p == y).sum() for p in preds_by_seed]))),
                         acc=float(np.mean(accs) * 100), sd=float(np.std(accs) * 100),
                         fixed=int(((p0 == y) & (ref != y)).sum()), broke=int(((p0 != y) & (ref == y)).sum()), pred=p0))
    for k, P in zip(names, Ps):
        pass
    fr = fixed_rules(Ps, names)
    for k, v in fr.items(): add(k, [v], 'rule' if 'Oracle' not in k else 'bound')
    for k, P in zip(names, Ps): add(f'{k} alone', [P.argmax(1)], 'expert')
    per_seed = [learned(Ps, s) for s in SEEDS]
    for k in per_seed[0]: add(k, [r[k] for r in per_seed], 'learned (CV)')
    if with_jev:
        for k, v in jev_rows(Ps, names).items(): add(k, [v], 'Jev (cached)')
    for r in rows: r.pop('pred')
    return rows


if __name__ == '__main__':
    out = {}
    out['A'] = dict(desc='DINOv2 + branch ConvNeXt + file-byte model (the three experts of the DINOv2 branch, file-byte replaces style features)',
                    rows=summarise('A', [E['D'], E['C'], E['F']], ['DINOv2', 'ConvNeXt (branch)', 'File-byte'], True))
    out['B'] = dict(desc='DINOv2 + final ConvNeXt run B (OOF) + file-byte model',
                    rows=summarise('B', [E['D'], E['R'], E['F']], ['DINOv2', 'ConvNeXt run B', 'File-byte'], False))
    json.dump(out, open(HERE / 'results.json', 'w'), indent=1)
    for s, blk in out.items():
        print(f'\n=== Set {s}: {blk["desc"]}')
        for r in blk['rows']:
            sd = f' ±{r["sd"]:.2f}' if r['sd'] > 0 else ''
            print(f'{r["acc"]:6.2f}%{sd:7s} {r["correct"]:5d}  +{r["fixed"]:<3d} -{r["broke"]:<3d} [{r["kind"]}] {r["name"]}')
