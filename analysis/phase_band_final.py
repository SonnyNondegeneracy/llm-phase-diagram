"""Working / drifting phase map of Fig. 3 (App. F, 'phase regions'); also the entry depth quoted in Sec. III.

Points: every (network, document, layer at relative depth >= 1/2) in (log10 u, log10 d_C), from the 8 trained models
(tag 1) and the 156 untrained runs (tag 0). Weights: each network's points share weight 1; each tag totals 1.
P(working) at a grid point = weighted share of tag-1 points among its k = 2000 nearest points (k chosen in
phase_k_choice.py). No value (NaN) where no point lies within R = 0.1 dex.
Statistics for the text: for each trained model's document-median trajectory, the relative depth of the first layer
inside the region P > 0.5 and of the first layer from which it stays inside.
Output: phase_band_final.npz (grid gx, gy and map P), phase_band_final.json. Queries are chunked (peak ~0.2 GB)."""
import glob, json
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
from runs import fam, L, RAW, out

R = 0.1; CH = 500; K = 2000
runs = [(f,) + fam(f) for f in sorted(glob.glob(f'{RAW}/*.npz'))]
runs = [r for r in runs if r[1] is not None]
nw = sum(r[1] == 'trained' for r in runs); nd = len(runs) - nw
P, W, Y = [], [], []; med = []
for f, fm, arch in runs:
    Z = np.load(f); U, D = L(Z['u']), L(Z['dC']); n = U.shape[0]; k = int(np.ceil(0.5 * (n - 1) - 1e-9)); tr = fm == 'trained'
    X = np.stack([U[k:].ravel(), D[k:].ravel()], 1); ok = np.isfinite(X).all(1)
    P.append(X[ok]); W.append(np.full(ok.sum(), 1 / ok.sum() / (nw if tr else nd))); Y.append(np.full(ok.sum(), float(tr)))
    if tr: med.append(np.stack([np.median(U, 1), np.median(D, 1)], 1))
P = np.concatenate(P); W = np.concatenate(W); Y = np.concatenate(Y)
gx, gy = np.meshgrid(np.linspace(-2.6, 1.7, 216), np.linspace(-0.1, 4.0, 206)); G = np.stack([gx.ravel(), gy.ravel()], 1)
tree = cKDTree(P); M = np.empty(len(G)); d0 = np.empty(len(G))
for i in range(0, len(G), CH):
    dist, idx = tree.query(G[i:i + CH], k=K, workers=2); w = W[idx]; M[i:i + CH] = (w * Y[idx]).sum(1) / w.sum(1); d0[i:i + CH] = dist[:, 0]
M = np.where(d0 < R, M, np.nan).reshape(gx.shape)


def inside(X):
    """P(working) > 0.5 at the points X (bilinear interpolation on the grid; no-data cells count as 0)"""
    i = (X[:, 1] - gy[0, 0]) / (gy[1, 0] - gy[0, 0]); j = (X[:, 0] - gx[0, 0]) / (gx[0, 1] - gx[0, 0])
    return map_coordinates(np.nan_to_num(M, nan=0.0), [i, j], order=1) > 0.5


first, stay = [], []
for X in med:
    I = inside(X); n = len(X) - 1
    first.append(next(i for i in range(len(X)) if I[i]) / n); s = next((i for i in range(len(X)) if I[i:].all()), None); stay.append(None if s is None else s / n)
rep = dict(n_points=int(len(P)), n_points_trained=int(Y.sum()), first_entry_median=float(np.median(first)), first=first, stay_from=stay)
np.savez(out('phase_band_final.npz'), gx=gx, gy=gy, P=M)
json.dump(rep, open(out('phase_band_final.json'), 'w'), indent=1); print(json.dumps(rep))
