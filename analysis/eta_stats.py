"""Force map and Peclet number of Fig. 4 (App. E, Algorithm 3); median Pe quoted in Sec. IV.

Decomposition v_{t+1} = alpha v_t + F(x) + eta inside the fit range 0.3 <= u <= 3, 30 <= d_C <= 400, with one alpha
per group (the main-text value, sens_unified_main.json). Per 0.1-dex cell (absolute grid): F = median over runs of the
per-run medians of f = v_{t+1} - alpha v_t (>= 5 transitions per run, >= 3 runs); eta = f - F;
s_i = 1.4826 x MAD of eta along axis i; Pe = sqrt(sum_i (F_i / s_i)^2).
Output eta_stats.json: per group, alpha and per cell (key 'i,j': cell [i h, (i+1) h) x [j h, (j+1) h) in
(log10 u, log10 d_C), h = 0.1) F, s, Pe ('ratio'), number of runs and of transitions."""
import glob, json
import numpy as np
from runs import L, RAW, out

H = 0.1
G = {k: v['gamma'] for k, v in json.load(open(out('sens_unified_main.json'))).items()}
WIN = (np.log10(0.3), np.log10(3), np.log10(30), np.log10(400))


def rows(X, docs):
    """transitions (x_t, v_t, v_{t+1}) of one run, rows of 6 numbers, documents in docs (X: layers x docs x 2)"""
    res = []
    for j in docs:
        x = X[:, j]
        for t in range(1, len(x) - 1): res.append((*x[t], *(x[t] - x[t - 1]), *(x[t + 1] - x[t])))
    return np.array(res)


res = {}
for name, pat in [('untrained', ['native', 'init', 'unit', 'sweep']), ('trained', ['trained'])]:
    g = G[name]; Rs = []
    for i, f in enumerate(sorted(sum([glob.glob(f'{RAW}/{p}__*.npz') for p in pat], []))):
        Z = np.load(f); X = np.stack([L(Z['u']), L(Z['dC'])], -1); R = rows(X, np.arange(X.shape[1])); Rs.append(np.column_stack([R, np.full(len(R), i)]))
    R = np.vstack(Rs); c = np.floor(R[:, :2] / H) * H + H / 2
    ok = (c[:, 0] >= WIN[0]) & (c[:, 0] <= WIN[1]) & (c[:, 1] >= WIN[2]) & (c[:, 1] <= WIN[3]); R = R[ok]
    key = np.floor(R[:, :2] / H).astype(int); r = R[:, 4:6] - g * R[:, 2:4]; B = {}
    for k in set(map(tuple, key)):
        m = np.all(key == k, 1); runs = np.unique(R[m, 6])
        med = [np.median(r[m & (R[:, 6] == q)], 0) for q in runs if (m & (R[:, 6] == q)).sum() >= 5]
        if len(med) < 3: continue
        F = np.median(med, 0); e = r[m] - F
        s = 1.4826 * np.median(np.abs(e - np.median(e, 0)), 0)
        B[f'{k[0]},{k[1]}'] = dict(F=F.tolist(), runs=len(med), s=s.tolist(), ratio=float(np.hypot(*(np.abs(F) / s))), n=int(m.sum()))
    rat = np.array([b['ratio'] for b in B.values()])
    print(f"{name}: alpha {g:+.3f} cells {len(B)}  median Pe {np.median(rat):.2f}")
    res[name] = dict(gamma=g, bins=B)
json.dump(res, open(out('eta_stats.json'), 'w'), indent=1)
