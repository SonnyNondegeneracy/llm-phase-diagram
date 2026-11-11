"""Per-model ablation of App. D ('quenched' ablation): -ln rho of each trained model fitted alone.

J fitted to one model at its own P* (late-half position from all documents), alpha = the trained group's value
(sens_unified_main.json), fit range = the absolute window of the main text (0.3 <= u <= 3, 30 <= d_C <= 400), weights
from 50 document resamples, 200 document-bootstrap draws (seed 0). Output rho_models_fixedwin.json: per model P*,
-ln rho, its 16-84 % interval and the number of cells; the median over models and the number with -ln rho > 0."""
import glob, json, os
import numpy as np
import fastpool as FP, pythia_pstar as PP
from runs import RAW, out

g = json.load(open(out('sens_unified_main.json')))['trained']['gamma']; I = np.eye(2); res = {}
def R_of(J): return -np.log(np.abs(np.linalg.eigvals(np.block([[I + J, g * I], [J, g * I]]))).max())
OFF0 = PP.OFF.copy()
for f in sorted(glob.glob(f'{RAW}/trained__*.npz')):
    nm = os.path.basename(f).split('__')[1]; run = FP.Run(f); rng = np.random.default_rng(0); P = PP.pstar(run.Z, np.arange(run.nd))
    FP.OFF = OFF0 + (PP.P0 - P)[:, None]              # fit range relative to P: same absolute window
    def pg(d): x, v, n = run.rows(d); return x - P, n - g * v
    F = FP.field(*pg(np.arange(run.nd))); Fb = [FP.field(*pg(rng.integers(0, run.nd, run.nd))) for _ in range(50)]
    w = {k: 1 / max(np.linalg.norm(np.std([fb[k] for fb in Fb if k in fb], 0)), 1e-3) ** 2 for k in F if sum(k in fb for fb in Fb) >= 10}
    J = FP.fitJ(F, w); B = []
    for _ in range(200):
        Jb = FP.fitJ(FP.field(*pg(rng.integers(0, run.nd, run.nd))), w)
        if Jb is not None: B.append(R_of(Jb))
    res[nm] = dict(P=(10 ** P).tolist(), R=float(R_of(J)) if J is not None else None, q=np.percentile(B, [16, 84]).tolist() if B else None, cells=len(w))
    print(nm, res[nm], flush=True)
R = [o['R'] for o in res.values() if o['R'] is not None]
res['median'] = float(np.median(R)); res['n_pos'] = int(sum(r > 0 for r in R)); print('median', res['median'], 'positive', res['n_pos'], 'of', len(R))
json.dump(res, open(out('rho_models_fixedwin.json'), 'w'), indent=1)
