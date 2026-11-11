"""Joint estimator of one group of networks (App. E, Algorithm 1) and its two-level bootstrap (Algorithm 2).

Transitions are precomputed once per run as arrays (layer x document); a bootstrap draw only re-indexes runs and
documents. Coordinates x = (log10 u, log10 d_C); a transition at layer t is (x_t, v_t = x_t - x_{t-1},
v_{t+1} = x_{t+1} - x_t).

H is the cell width (dex) and OFF the fit range relative to the linearisation point; App. F changes both."""
import numpy as np
from pythia_pstar import pstar, OFF, L

H = 0.1


class Run:
    """one network: its moments arrays Z and its transitions x, v, n = (x_t, v_t, v_{t+1}), each (layers-2, docs, 2)"""
    def __init__(self, f):
        self.Z = Z = np.load(f); X = np.stack([L(Z['u']), L(Z['dC'])], -1)            # (layers, docs, 2)
        self.x, self.v, self.n = X[1:-1], X[1:-1] - X[:-2], X[2:] - X[1:-1]; self.nd = X.shape[1]

    def rows(self, d):
        return self.x[:, d].reshape(-1, 2), self.v[:, d].reshape(-1, 2), self.n[:, d].reshape(-1, 2)


def gfe(x, v, n, rid):
    """inertia coefficient alpha: slope of v_{t+1} on v_t after removing the mean of each (network, cell of x_t)
    class (fixed effects), one slope for both coordinates"""
    k = np.floor(x / H).astype(np.int64); kid = rid.astype(np.int64) * 10 ** 6 + (k[:, 0] + 500) * 1000 + k[:, 1] + 500
    _, inv, c = np.unique(kid, return_inverse=True, return_counts=True); inv = inv.ravel()
    mv = np.zeros((len(c), 2)); mo = np.zeros((len(c), 2)); np.add.at(mv, inv, v); np.add.at(mo, inv, n)
    X = v - mv[inv] / c[inv, None]; Y = n - mo[inv] / c[inv, None]; return (X * Y).sum() / (X * X).sum()


def pieces(runs, docs, pfix=None):
    """Pool the transitions of the given runs over the documents docs (one index array shared by all runs).
    Returns positions relative to the linearisation point, the force samples f = v_{t+1} - alpha v_t, and alpha.
    pfix: common linearisation point for every run; None: each run at its own P*."""
    xs, vs, ns, ps, rid = [], [], [], [], []
    for i, r in enumerate(runs):
        x, v, n = r.rows(docs); xs.append(x); vs.append(v); ns.append(n); ps.append(np.broadcast_to(pstar(r.Z, docs) if pfix is None else pfix, x.shape)); rid.append(np.full(len(x), i))
    x, v, n, p, rid = map(np.concatenate, (xs, vs, ns, ps, rid)); g = gfe(x, v, n, rid)
    return x - p, n - g * v, g


def field(dx, r):
    """binned force: median of r in each cell (width H, origin at the linearisation point) whose centre lies in the
    fit range OFF and that holds >= 20 transitions. Returns {cell index (i, j): median force}."""
    k = np.floor(dx / H).astype(np.int64); c = k * H + H / 2; ok = np.all((c >= OFF[:, 0]) & (c <= OFF[:, 1]), 1); k, r = k[ok], r[ok]
    kid = (k[:, 0] + 500) * 1000 + k[:, 1] + 500; o = np.argsort(kid, kind='stable'); kid, r = kid[o], r[o]
    u, start, cnt = np.unique(kid, return_index=True, return_counts=True); F = {}
    for uu, s0, cc in zip(u, start, cnt):
        if cc >= 20: F[(int(uu // 1000) - 500, int(uu % 1000) - 500)] = np.median(r[s0:s0 + cc], 0)
    return F


def fitJ(F, w):
    """weighted least squares F_k = a + J c_k over the cells that have a weight (c_k: cell centre relative to the
    linearisation point); None if fewer than 6 cells"""
    ks = [k for k in F if k in w]
    if len(ks) < 6: return None
    X = np.array([np.array(k) * H + H / 2 for k in ks]); Y = np.array([F[k] for k in ks]); W = np.sqrt([w[k] for k in ks])
    C = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]) * W[:, None], Y * W[:, None], rcond=None)[0]; return C[1:].T


def boot(runs, rng):
    """one two-level bootstrap draw: documents first (shared by all runs), then networks, both with replacement"""
    nd = runs[0].nd; d = rng.integers(0, nd, nd); return [runs[i] for i in rng.integers(0, len(runs), len(runs))], d

