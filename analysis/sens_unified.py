"""Group estimates of Sec. IV and their sensitivity to the analysis settings (Fig. 5 bands, App. F Fig. F-settings,
Table F-settings).

Every setting uses the same estimator (fastpool joint fit, cell weights from 50 bootstrap draws) and the same
uncertainty (200 two-level bootstrap draws, networks then documents; 16-84 % percentiles) for alpha ('gamma' in the
code) and -ln rho of both groups. Usage: python sens_unified.py <task>, output sens_unified_<task>.json. Tasks:
  main         settings of the main text, at P0; -ln rho also with the force taken at the midpoint and at the end
               point of the transition (App. D / F)
  last_third   late segment = relative depth >= 2/3; linearisation point = median of the trained models' P* from that
               segment; fit range kept at the absolute window of the main text
  cell<h>      cell width h dex (cell0.07, cell0.14, cell0.2)
  win<d>       fit range moved outward by d dex on every side (win-0.1, win0.1)
  loo_<arch>   one architecture left out of both groups (its untrained runs and its trained model)
  half0/half1  only the documents with even / odd index (48 documents)"""
import sys, glob, json, os
import numpy as np
import fastpool as FP, pythia_pstar as PP
from runs import RAW, out

I = np.eye(2)


def Rho(J, g, scheme='explicit'):
    """-ln rho of the linear map. Force evaluated at x_t (explicit), at the midpoint of x_t and x_{t+1}, or at x_{t+1}
    (implicit): dv' = B (J dx + g dv), dx' = dx + dv', B = (I - c J)^-1, c = 0, 1/2, 1"""
    c = {'explicit': 0.0, 'midpoint': 0.5, 'implicit': 1.0}[scheme]; B = np.linalg.inv(I - c * J)
    Mx = np.block([[I + B @ J, g * B], [B @ J, g * B]]); return -np.log(np.abs(np.linalg.eigvals(Mx)).max())


def joint(runs, P, nb=200, nw=50, seed=0, schemes=('explicit',)):
    rng = np.random.default_rng(seed); dx, r, g = FP.pieces(runs, np.arange(runs[0].nd), P); F = FP.field(dx, r)
    Fb = [FP.field(*FP.pieces(*FP.boot(runs, rng), P)[:2]) for _ in range(nw)]
    w = {k: 1 / max(np.linalg.norm(np.std([fb[k] for fb in Fb if k in fb], 0)), 1e-3) ** 2 for k in F if sum(k in fb for fb in Fb) >= 10}
    J = FP.fitJ(F, w); ev = np.linalg.eigvals(J)
    res = dict(gamma=float(g), ev=sorted(ev.real.tolist()), imag=float(abs(ev.imag).max()), nbins=len(w), R={s: float(Rho(J, g, s)) for s in schemes})
    GB, RB, TY = [], {s: [] for s in schemes}, []
    for _ in range(nb):
        dxb, rb, gb = FP.pieces(*FP.boot(runs, rng), P); Jb = FP.fitJ(FP.field(dxb, rb), w); GB.append(gb)
        if Jb is None: continue
        for s in schemes: RB[s].append(Rho(Jb, gb, s))
        e = np.linalg.eigvals(Jb); TY.append('node' if np.all(e.real < 0) else 'saddle' if (e.real.min() < 0 < e.real.max()) else 'other')
    res['gamma_q'] = np.percentile(GB, [16, 84]).tolist(); res['R_q'] = {s: np.percentile(RB[s], [16, 84]).tolist() for s in schemes}
    res['type_share'] = {t: TY.count(t) / len(TY) for t in set(TY)}; res['nboot_J'] = len(TY)
    return res


UNT = ['native', 'init', 'unit', 'sweep']             # untrained families (file prefixes)
def files(pat): return [f for p in pat for f in sorted(glob.glob(f'{RAW}/{p}__*.npz'))]
def groups(): return {'untrained': [FP.Run(f) for f in files(UNT)], 'trained': [FP.Run(f) for f in files(['trained'])]}


task = sys.argv[1]
if task == 'main':
    G = groups(); res = {k: joint(v, PP.P0, schemes=('explicit', 'midpoint', 'implicit')) for k, v in G.items()}
elif task == 'last_third':
    G = groups(); PP.FROM = 2 / 3; P = np.median([PP.pstar(r.Z, np.arange(r.nd)) for r in G['trained']], 0); PP.FROM = 0.5
    FP.OFF = PP.OFF + (PP.P0 - P)[:, None]             # same absolute fit range as the main text
    res = {k: joint(v, P) for k, v in G.items()}; res['P'] = (10 ** P).tolist()
elif task.startswith('cell'):
    FP.H = float(task[4:]); G = groups(); res = {k: joint(v, PP.P0) for k, v in G.items()}
elif task.startswith('win'):
    d = float(task[3:]); FP.OFF = FP.OFF + np.array([[-d, d], [-d, d]]); G = groups(); res = {k: joint(v, PP.P0) for k, v in G.items()}
elif task.startswith('loo_'):
    a = task[4:]; G = {k: [FP.Run(f) for f in files(UNT if k == 'untrained' else ['trained']) if os.path.basename(f).split('__')[1] != a] for k in ('untrained', 'trained')}
    res = {k: joint(v, PP.P0) for k, v in G.items()}; res['n'] = {k: len(v) for k, v in G.items()}
elif task in ('half0', 'half1'):
    G = groups(); j = int(task[-1])
    for v in G.values():
        for r in v: r.x, r.v, r.n = r.x[:, j::2], r.v[:, j::2], r.n[:, j::2]; r.nd = r.x.shape[1]
    res = {k: joint(v, PP.P0) for k, v in G.items()}
else:
    raise SystemExit(f'unknown task {task}')
json.dump(res, open(out(f'sens_unified_{task}.json'), 'w'), indent=1); print(task, json.dumps(res)[:400])
