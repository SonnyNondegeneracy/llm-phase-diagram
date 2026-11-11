"""alpha of every Pythia checkpoint separately (Fig. 5b).

Each checkpoint is one joint fit (fastpool estimator; weights from 50 bootstrap draws; 200 bootstrap draws over
documents, seed 0) linearised at Pythia's reference point P_Py = P* of the final checkpoint (step 143000), with the
fit range of the main text shifted to P_Py; -ln rho of the linear map is reported where J can be fitted.
At steps 6000 and 8000 too few cells lie near P_Py for a J fit; alpha does not need J, so for these checkpoints its
16-84 % interval comes from 200 further document-bootstrap draws of alpha alone (one generator, seed 0, shared by these
checkpoints in step order); they are marked 'gamma_q_alpha_only'. Output pythia_ck_fixedP.json."""
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
    return dict(evaluable=True, gamma=float(g), gamma_q=np.percentile(GB, [16, 84]).tolist(), R=float(R_of(J, g)),
                R_q=np.percentile(RB, [16, 84]).tolist(), nbins=len(w), nboot=len(RB))


STEPS = [0, 64, 128, 256, 512, 1000, 2000, 3000, 4000, 6000, 8000, 12000, 16000, 24000, 32000, 48000, 64000, 96000, 143000]
fin = FP.Run(raw('pythia__step143000__0.npz')); PPY = pstar(fin.Z, np.arange(fin.nd))
res = {'P_pythia': PPY.tolist(), 'at_Ppy': []}
for s in STEPS:
    run = [FP.Run(raw(f'pythia__step{s}__0.npz'))]
    r = joint(run, PPY); r['step'] = s; res['at_Ppy'].append(r)
    print(s, {k: (np.round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != 'step'}, flush=True)
rng = np.random.default_rng(0)                    # alpha-only interval for the checkpoints without a J fit
for r in res['at_Ppy']:
    if r['evaluable']: continue
    runs = [FP.Run(raw(f"pythia__step{r['step']}__0.npz"))]
    gb = [FP.pieces(*FP.boot(runs, rng), PPY)[2] for _ in range(200)]
    r['gamma_q'] = np.percentile(gb, [16, 84]).tolist(); r['gamma_q_alpha_only'] = True
    print(r['step'], round(r['gamma'], 3), np.round(r['gamma_q'], 3))
json.dump(res, open(out('pythia_ck_fixedP.json'), 'w'), indent=1)
