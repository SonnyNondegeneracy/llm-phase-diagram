"""Share of each Pythia checkpoint's deep points in the working phase (Fig. 5a; App. F).

Deep points: 96 documents x layers at relative depth >= 1/2. Share = fraction with P(working) > 0.5 on the phase map of
Fig. 3 (phase_band_final.npz, bilinear interpolation, no-data cells count as 0). Error: 16-84 % of 200 bootstrap draws
over documents (seed 0). Output pythia_share.json."""
import glob, re, json, os
import numpy as np
from scipy.ndimage import map_coordinates
from runs import L, RAW, out

B = np.load(out('phase_band_final.npz')); gx, gy, P = B['gx'], B['gy'], np.nan_to_num(B['P'], nan=0.0)
def pw(X):
    i = (X[..., 1] - gy[0, 0]) / (gy[1, 0] - gy[0, 0]); j = (X[..., 0] - gx[0, 0]) / (gx[0, 1] - gx[0, 0])
    return map_coordinates(P, [i.ravel(), j.ravel()], order=1).reshape(i.shape)
step = lambda f: int(re.search(r'step(\d+)', os.path.basename(f)).group(1))
rng = np.random.default_rng(0); res = []
for f in sorted(glob.glob(f'{RAW}/pythia__step*__0.npz'), key=step):
    st = step(f); Z = np.load(f); U, D = L(Z['u']), L(Z['dC']); n = U.shape[0]; k = int(np.ceil(0.5 * (n - 1) - 1e-9))
    X = np.stack([U[k:], D[k:]], -1)                       # layers x docs x 2
    per_doc = (pw(X) > 0.5).mean(0)                        # share per document
    bs = [per_doc[rng.integers(0, len(per_doc), len(per_doc))].mean() for _ in range(200)]
    res.append(dict(step=st, share=float(per_doc.mean()), q=np.percentile(bs, [16, 84]).tolist()))
    print(res[-1])
json.dump(res, open(out('pythia_share.json'), 'w'), indent=1)
