"""Pythia-1.4B binned joint fits (Fig. 5c, App. F Fig. F-bins).

Each bin of checkpoints is one joint fit (fastpool estimator: one alpha, binned force, weighted J fit; weights from
50 bootstrap draws; 200 two-level bootstrap draws, checkpoints then documents; seed 0) at a fixed linearisation point,
with the main-text fit range shifted to that point; -ln rho of the linear map. Step 0 is its own bin in every scheme.
Schemes: main (Fig. 5c; every bin has >= 2 checkpoints and >= 20 cells near P_Py, step 0 excepted), half decade
(edges 10^(k/2) steps), one decade (edges 10^k steps), all at P_Py = P* of the final checkpoint; and the main bins at
P_late = median of the P* of the checkpoints from step 6000 on. Output pythia_bins.json."""
import json
import numpy as np
import fastpool as FP
from pythia_pstar import pstar
from runs import raw, out

I = np.eye(2)
def R_of(J, g): return -np.log(np.abs(np.linalg.eigvals(np.block([[I + J, g * I], [J, g * I]]))).max())


def joint(runs, pfix, nb=200, seed=0):
    rng = np.random.default_rng(seed); dx, r, g = FP.pieces(runs, np.arange(runs[0].nd), pfix); F = FP.field(dx, r)
    Fb = [FP.field(*FP.pieces(*FP.boot(runs, rng), pfix)[:2]) for _ in range(50)]
    w = {k: 1 / max(np.linalg.norm(np.std([fb[k] for fb in Fb if k in fb], 0)), 1e-3) ** 2 for k in F if sum(k in fb for fb in Fb) >= 10}
    J = FP.fitJ(F, w)
    if J is None: return dict(evaluable=False, gamma=float(g), nbins=len(w))
    RB, GB = [], []
    for _ in range(nb):
        dxb, rb, gb = FP.pieces(*FP.boot(runs, rng), pfix); Jb = FP.fitJ(FP.field(dxb, rb), w); GB.append(gb)
        if Jb is not None: RB.append(R_of(Jb, gb))
    return dict(evaluable=True, gamma=float(g), gamma_q=np.percentile(GB, [16, 84]).tolist(), R=float(R_of(J, g)), R_q=np.percentile(RB, [16, 84]).tolist(), nbins=len(w))


ld = lambda b: [FP.Run(raw(f'pythia__step{s}__0.npz')) for s in b]
P_PY = pstar(ld([143000])[0].Z, np.arange(96))
P_LATE = np.median([pstar(r.Z, np.arange(r.nd)) for r in ld([6000, 8000, 12000, 16000, 24000, 32000, 48000, 64000, 96000, 143000])], 0)
MAIN = [[0], [64, 128, 256], [512, 1000], [2000, 3000, 4000, 6000, 8000, 12000, 16000, 24000], [32000, 48000, 64000], [96000, 143000]]
HALF = [[0], [64], [128, 256], [512], [1000, 2000, 3000], [4000, 6000, 8000], [12000, 16000, 24000], [32000, 48000, 64000, 96000], [143000]]
DEC = [[0], [64], [128, 256, 512], [1000, 2000, 3000, 4000, 6000, 8000], [12000, 16000, 24000, 32000, 48000, 64000, 96000], [143000]]
res = {'P_py': (10 ** P_PY).tolist(), 'P_late': (10 ** P_LATE).tolist()}
for name, bins, P in [('main', MAIN, P_PY), ('half_decade', HALF, P_PY), ('one_decade', DEC, P_PY), ('main_at_late', MAIN, P_LATE)]:
    res[name] = []
    for b in bins:
        r = joint(ld(b), P); r['steps'] = b; res[name].append(r)
        print(name, b[0], b[-1], round(r['R'], 4) if r['evaluable'] else '-', r['nbins'], flush=True)
json.dump(res, open(out('pythia_bins.json'), 'w'), indent=1)
