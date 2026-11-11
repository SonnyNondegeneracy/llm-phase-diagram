"""Joint J of each group at the trained P* (Fig. 4 eigen-directions; App. B fates dividing line).

Same estimator as sens_unified.py (task main): every run linearised at P0, one alpha per group, binned force, weights
from 50 bootstrap draws (seed 0), weighted fit F = a + J (x - P0). Output fig4_J.json: J, its eigenvalues (sorted by
real part) and the matching eigenvectors (columns of V)."""
import glob, json
import numpy as np
import fastpool as FP
from pythia_pstar import P0
from runs import RAW, out

res = {}
for name, pat in [('untrained', ['native', 'init', 'unit', 'sweep']), ('trained', ['trained'])]:
    runs = [FP.Run(f) for p in pat for f in sorted(glob.glob(f'{RAW}/{p}__*.npz'))]; rng = np.random.default_rng(0)
    dx, r, g = FP.pieces(runs, np.arange(runs[0].nd), P0); F = FP.field(dx, r)
    Fb = [FP.field(*FP.pieces(*FP.boot(runs, rng), P0)[:2]) for _ in range(50)]
    w = {k: 1 / max(np.linalg.norm(np.std([fb[k] for fb in Fb if k in fb], 0)), 1e-3) ** 2 for k in F if sum(k in fb for fb in Fb) >= 10}
    J = FP.fitJ(F, w); ev, V = np.linalg.eig(J); o = np.argsort(ev.real); ev, V = ev[o], V[:, o]
    res[name] = dict(J=J.tolist(), ev_re=ev.real.tolist(), ev_im=ev.imag.tolist(), V=np.real(V).tolist())
    print(name, 'eig', np.round(ev, 4), ' dirs (deg)', np.round(np.degrees(np.arctan2(np.real(V)[1], np.real(V)[0])) % 180, 0))
json.dump(res, open(out('fig4_J.json'), 'w'), indent=1)
