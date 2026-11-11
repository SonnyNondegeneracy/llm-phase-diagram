"""Choice of the neighbourhood size k of the phase map (App. F, Fig. F-k).

Leave-one-architecture-out cross-validation: hold out every network of one architecture (its trained model and all its
untrained runs), predict P(working) for its deep points (relative depth >= 1/2) from the remaining points with the
estimator of phase_band_final.py, and score the class-balanced log loss over all held-out points; queries in chunks of 500. Output phase_k_choice.json: mean loss and per-architecture loss for each k."""
import glob, json
import numpy as np
from scipy.spatial import cKDTree
from runs import fam, L, RAW, out

P, W, Y, A = [], [], [], []
runs = [(f,) + fam(f) for f in sorted(glob.glob(f'{RAW}/*.npz'))]
runs = [r for r in runs if r[1] is not None]
nw = sum(r[1] == 'trained' for r in runs); nd = len(runs) - nw
for f, fm, arch in runs:
    Z = np.load(f); U, D = L(Z['u']), L(Z['dC']); n = U.shape[0]; k = int(np.ceil(0.5 * (n - 1) - 1e-9))
    X = np.stack([U[k:].ravel(), D[k:].ravel()], 1); X = X[np.isfinite(X).all(1)]; tr = fm == 'trained'
    P.append(X); W.append(np.full(len(X), 1 / len(X) / (nw if tr else nd))); Y.append(np.full(len(X), float(tr))); A.append(np.full(len(X), arch))
P = np.concatenate(P); W = np.concatenate(W); Y = np.concatenate(Y); A = np.concatenate(A)
KS = [50, 100, 200, 500, 1000, 2000, 3000]; loss = {k: [] for k in KS}
for arch in np.unique(A):
    te = A == arch; tree = cKDTree(P[~te]); Wtr, Ytr = W[~te], Y[~te]
    it = np.where(te)[0]
    yt = Y[it]; sw = np.where(yt == 1, 0.5 / max(yt.sum(), 1), 0.5 / max((1 - yt).sum(), 1))   # class-balanced scoring weights
    for k in KS:
        pr = np.empty(len(it))
        for i in range(0, len(it), 500):
            _, idx = tree.query(P[it[i:i + 500]], k=k, workers=2); w = Wtr[idx]; pr[i:i + 500] = (w * Ytr[idx]).sum(1) / w.sum(1)
        pr = np.clip(pr, 1e-3, 1 - 1e-3); loss[k].append(np.sum(sw * -(yt * np.log(pr) + (1 - yt) * np.log(1 - pr))))
OUT = {'loss': {k: float(np.mean(loss[k])) for k in KS}, 'loss_per_arch': {k: [float(x) for x in loss[k]] for k in KS}}
for k in KS: print('k=%5d  held-out-architecture log loss %.3f' % (k, np.mean(loss[k])))
json.dump(OUT, open(out('phase_k_choice.json'), 'w'), indent=1)
