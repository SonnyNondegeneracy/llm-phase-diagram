"""Fates of the 156 untrained runs (App. B, Fig. B-fates).

The deep points of a run (96 documents x layers at relative depth >= 1/2) fall into three classes: in the working
region (P(working) > 0.5 on the phase map, phase_band_final.npz) or, otherwise, on either side of the line through
P* = (1.08, 117) along the restoring (negative-eigenvalue) eigen-direction of the untrained group's J (fig4_J.json):
collapse side (large u, small d_C) or noise side. A run takes the class with the largest share.
Output fates_share.json: counts, and per run its family, sigma, late-half position, class shares and fate."""
import glob, json, os
import numpy as np
from scipy.ndimage import map_coordinates
from runs import fam, late, L, RAW, out

B = np.load(out('phase_band_final.npz')); gx, gy, P = B['gx'], B['gy'], np.nan_to_num(B['P'], nan=0.0)
def pw(X): return map_coordinates(P, [(X[:, 1] - gy[0, 0]) / (gy[1, 0] - gy[0, 0]), (X[:, 0] - gx[0, 0]) / (gx[0, 1] - gx[0, 0])], order=1)
J = json.load(open(out('fig4_J.json')))['untrained']; V = np.array(J['V']); vs = V[:, int(np.argmin(J['ev_re']))]
nrm = np.array([-vs[1], vs[0]]); nrm = nrm if nrm @ [1, -1] > 0 else -nrm          # normal of the dividing line, toward collapse
Ps = np.array([np.log10(1.08), np.log10(117)])
C = ['working', 'collapse', 'noise']; res = {}
for f in sorted(glob.glob(f'{RAW}/*.npz')):
    fm, a = fam(f)
    if fm is None or fm == 'trained': continue
    Z = np.load(f); n = Z['u'].shape[0]; k = int(np.ceil(0.5 * (n - 1) - 1e-9))
    X = np.stack([L(Z['u'][k:]).ravel(), L(Z['dC'][k:]).ravel()], 1); X = X[np.isfinite(X).all(1)]
    w = pw(X) > 0.5; side = (X - Ps) @ nrm > 0
    share = [w.mean(), (~w & side).mean(), (~w & ~side).mean()]
    m, e = late(f)
    res[os.path.basename(f)[:-4]] = dict(family=fm[0], sigma=float(fm.split('(')[0].replace('G1', 'G1.0')[1:]), median=m.tolist(),
                                         share=dict(zip(C, map(float, share))), fate=C[int(np.argmax(share))])
cnt = {c: sum(r['fate'] == c for r in res.values()) for c in C}
json.dump(dict(counts=cnt, runs=res), open(out('fates_share.json'), 'w'), indent=1); print(cnt, 'min winning share', round(min(max(r['share'].values()) for r in res.values()), 3))
