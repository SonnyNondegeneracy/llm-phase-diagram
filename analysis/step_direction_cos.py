"""Persistence of the step direction at Pythia step 0 (App. F, 'phase criterion').

Document-median trajectory x_t in (log10 u, log10 d_C), v_t = x_t - x_{t-1}; mean cosine between consecutive steps over
the layers at relative depth >= 1/2. Output step_direction_cos.json."""
import json
import numpy as np
from runs import L, raw, out


def cosmean(f):
    Z = np.load(f); X = np.stack([np.median(L(Z['u']), 1), np.median(L(Z['dC']), 1)], 1); n = len(X); k = int(np.ceil(0.5 * (n - 1) - 1e-9))
    V = np.diff(X[k:], axis=0); c = [V[i] @ V[i + 1] / np.linalg.norm(V[i]) / np.linalg.norm(V[i + 1]) for i in range(len(V) - 1)]
    return float(np.mean(c))


res = {'pythia_step0': cosmean(raw('pythia__step0__0.npz'))}
json.dump(res, open(out('step_direction_cos.json'), 'w'), indent=1); print(round(res['pythia_step0'], 3))
