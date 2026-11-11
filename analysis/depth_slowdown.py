"""Untrained flows slow down with depth (App. B).

Per untrained run: document-median trajectory x_t = (log10 u, log10 d_C) and per-layer displacement
s_t = |x_t - x_{t-1}|. Over the layers at relative depth >= 1/2: slope of log s against log t, and the extra
displacement on doubling the depth under s = c/t, c ln 2 with c = geometric mean of s_t t. Reported: median slope,
median extra displacement and the share of runs with a negative slope, each with the 16-84 % interval of a
two-level bootstrap (runs, then documents; 200 draws, seed 0). Output depth_slowdown.json."""
import glob, json
import numpy as np
from runs import fam, L, RAW, out

runs = []
for f in sorted(glob.glob(f'{RAW}/*.npz')):
    fm, a = fam(f)
    if fm is None or fm == 'trained': continue
    Z = np.load(f); runs.append((L(Z['u']), L(Z['dC'])))
assert len(runs) == 156, len(runs)


def per_run(lu, ld, idx=None):
    if idx is not None: lu, ld = lu[:, idx], ld[:, idx]
    X = np.stack([np.median(lu, 1), np.median(ld, 1)], 1); n = len(X); k = int(np.ceil(0.5 * (n - 1) - 1e-9))
    s = np.linalg.norm(np.diff(X, axis=0), axis=1); t = np.arange(1, n); h = t >= k
    sl = np.polyfit(np.log(t[h]), np.log(s[h] + 1e-12), 1)[0]
    c = np.exp(np.mean(np.log(s[h] * t[h] + 1e-12)))
    return sl, c * np.log(2)


def stats(rs):
    rs = np.array(rs, dtype=float)
    return [np.median(rs[:, 0]), np.median(rs[:, 1]), np.mean(rs[:, 0] < 0)]


point = stats([per_run(*r) for r in runs])
rng = np.random.default_rng(0); B = []
for b in range(200):
    pick = rng.integers(0, len(runs), len(runs)); rs = []
    for i in pick:
        lu, ld = runs[i]; rs.append(per_run(lu, ld, rng.integers(0, lu.shape[1], lu.shape[1])))
    B.append(stats(rs))
B = np.array(B); q = np.percentile(B, [16, 84], 0)
names = ['slope_median', 'extra_dex_per_doubling', 'frac_slope_neg']
res = {nm: {'point': float(point[i]), 'q16': float(q[0, i]), 'q84': float(q[1, i])} for i, nm in enumerate(names)}
json.dump(res, open(out('depth_slowdown.json'), 'w'), indent=1)
for k, v in res.items(): print(k, '%.3f (+%.3f / -%.3f)' % (v['point'], v['q84'] - v['point'], v['point'] - v['q16']))
