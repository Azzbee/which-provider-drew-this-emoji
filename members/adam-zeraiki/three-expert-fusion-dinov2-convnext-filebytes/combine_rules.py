"""Combination rules beyond averaging, on the same 1,976 development images and the same
grouped 5-fold x 5-seed protocol as combine.py.

1. Confusion-matrix Bayes (Dawid-Skene style): each expert's answer is evidence weighted by how
   that expert actually confuses classes, P(y | answers) ~ P(y) * prod_e P(answer_e | y).
2. Most confident expert wins (after each expert's probabilities are temperature-calibrated).
3. Per-class trust: weighted log-probabilities with one weight per expert and class.
4. Error-correcting output codes, the "poisoned cup" idea: each provider gets a binary code word,
   one yes/no classifier ("taster") per bit, prediction = nearest code word. 3 bits is the
   minimum for 7 classes (2^3 = 8, no error correction); 7 and 15 random bits and the exhaustive
   63-bit code add redundancy that can correct wrong bits.
"""
import itertools, json, numpy as np
from pathlib import Path
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).resolve().parent
E = np.load(HERE / 'experts_dev.npz')
y, groups = E['y'], E['groups']
N, K, eps = len(y), 7, 1e-6
SEEDS = [0, 1, 2, 3, 4]


def folds(seed):
    return StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(np.zeros(N), y, groups)


def confusion_bayes(Ps, seed):
    lab = [P.argmax(1) for P in Ps]; pred = np.zeros(N, int)
    for a, b in folds(seed):
        prior = np.bincount(y[a], minlength=K) + 1.0
        logpost = np.log(prior / prior.sum())[None, :].repeat(len(b), 0)
        for l in lab:
            C = np.ones((K, K))                       # Laplace smoothing
            np.add.at(C, (y[a], l[a]), 1)
            C /= C.sum(1, keepdims=True)              # C[true, answered]
            logpost += np.log(C[:, l[b]].T)
        pred[b] = logpost.argmax(1)
    return pred


def temper(P, idx):
    """Fit one temperature on idx by log-loss; return the function applying it."""
    L = np.log(P + eps)
    f = lambda t: -np.mean(np.log(np.exp(L[idx] / t[0] - np.log(np.exp(L[idx] / t[0]).sum(1, keepdims=True)))[np.arange(len(idx)), y[idx]] + 1e-12))
    t = minimize(f, [1.0], bounds=[(0.05, 20)]).x[0]
    def apply(rows):
        Z = L[rows] / t; Z -= Z.max(1, keepdims=True); Q = np.exp(Z); return Q / Q.sum(1, keepdims=True)
    return apply


def most_confident(Ps, seed):
    pred = np.zeros(N, int)
    for a, b in folds(seed):
        Q = [temper(P, a)(b) for P in Ps]
        conf = np.stack([q.max(1) for q in Q]); who = conf.argmax(0)
        pred[b] = np.stack([q.argmax(1) for q in Q])[who, np.arange(len(b))]
    return pred


def per_class_trust(Ps, seed):
    L = np.stack([np.log(P + eps) for P in Ps])      # (3, N, K)
    pred = np.zeros(N, int)
    for a, b in folds(seed):
        def loss(w):
            W = w.reshape(len(Ps), K); S = (W[:, None, :] * L[:, a, :]).sum(0)
            S = S - S.max(1, keepdims=True); lp = S - np.log(np.exp(S).sum(1, keepdims=True))
            return -lp[np.arange(len(a)), y[a]].mean() + 1e-3 * np.sum((w - 1) ** 2)
        w = minimize(loss, np.ones(len(Ps) * K), method='L-BFGS-B').x.reshape(len(Ps), K)
        pred[b] = (w[:, None, :] * L[:, b, :]).sum(0).argmax(1)
    return pred


def ecoc(Ps, seed, code):
    X = np.column_stack([np.log(P + eps) for P in Ps]); pred = np.zeros(N, int)
    for a, b in folds(seed):
        bits = []
        for j in range(code.shape[1]):
            t = code[y[a], j]
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(X[a], t)
            bits.append(m.predict_proba(X[b])[:, 1])
        B = np.column_stack(bits)                      # soft bits in [0, 1]
        dist = np.abs(B[:, None, :] - code[None, :, :]).sum(2)   # soft Hamming distance
        pred[b] = dist.argmin(1)
    return pred


def codes():
    rng = np.random.default_rng(7)
    minimal = np.array([[int(c) for c in format(i + 1, '03b')] for i in range(K)])  # 3 bits, 7 distinct words
    def random_code(L):
        best, bd = None, -1
        for _ in range(2000):
            C = rng.integers(0, 2, (K, L))
            if any(C[:, j].min() == C[:, j].max() for j in range(L)): continue
            d = min(np.abs(C[i] - C[k]).sum() for i, k in itertools.combinations(range(K), 2))
            if d > bd: best, bd = C, d
        return best, bd
    c7, d7 = random_code(7); c15, d15 = random_code(15)
    exh = np.array([[int(c) for c in format(m, f'0{K}b')] for m in range(1, 2 ** (K - 1))]).T  # 7 x 63
    return {'ECOC, 3 bits (minimum, like the poisoned cups; min distance 1)': minimal,
            f'ECOC, 7 random bits (min distance {d7})': c7,
            f'ECOC, 15 random bits (min distance {d15})': c15,
            'ECOC, exhaustive 63 bits (min distance 32)': exh}


if __name__ == '__main__':
    out = {}
    for setname, Ps, ref_pair in [('A: DINOv2 + branch ConvNeXt + file-byte', [E['D'], E['C'], E['F']], (E['D'], E['C'])),
                                  ('B: DINOv2 + ConvNeXt run B + file-byte', [E['D'], E['R'], E['F']], (E['D'], E['R']))]:
        ref = ((ref_pair[0] + ref_pair[1]) / 2).argmax(1)
        geo = np.exp(sum(np.log(P + eps) for P in Ps)).argmax(1)
        rows = {'Reference: two image models, equal average': [ref], 'Geometric mean of the three (no training)': [geo]}
        rows['Confusion-matrix Bayes on the three answers'] = [confusion_bayes(Ps, s) for s in SEEDS]
        rows['Most confident calibrated expert wins'] = [most_confident(Ps, s) for s in SEEDS]
        rows['Per-class trust weights (21 weights)'] = [per_class_trust(Ps, s) for s in SEEDS]
        for name, C in codes().items():
            rows[name] = [ecoc(Ps, s, C) for s in SEEDS]
        print('\n==', setname)
        res = []
        for name, preds in rows.items():
            acc = np.array([(p == y).mean() for p in preds]) * 100; p0 = preds[0]
            r = dict(name=name, acc=round(float(acc.mean()), 2), sd=round(float(acc.std()), 2),
                     fixed=int(((p0 == y) & (ref != y)).sum()), broke=int(((p0 != y) & (ref == y)).sum()))
            res.append(r); print(f"{r['acc']:6.2f}% ±{r['sd']:.2f}  +{r['fixed']:<3d} -{r['broke']:<3d} {name}")
        out[setname] = res
    json.dump(out, open(HERE / 'results_rules.json', 'w'), indent=1)
