"""Spread of the late-half positions (Sec. III, App. B): RMS distance from the group centroid in (log10 u, log10 d_C)
for the 8 trained models and, for each of the initialisations F0.02, G0.02 and G1, across its 24 runs
(8 architectures x 3 seeds). Output dispersion.json."""
import glob, json, collections
import numpy as np
from runs import late, fam, RAW, out

pos = collections.defaultdict(list)
for f in sorted(glob.glob(f'{RAW}/*.npz')):
    fm, a = fam(f)
    if fm is not None: pos[fm].append(late(f)[0])
disp = lambda P: float(np.sqrt(np.sum(np.var(np.array(P), 0))))
res = {'trained': disp(pos['trained'])} | {fm: disp(pos[fm]) for fm in ['F0.02(native)', 'G0.02', 'G1(N01)']}
json.dump(res, open(out('dispersion.json'), 'w'), indent=1); print({k: round(v, 3) for k, v in res.items()})
