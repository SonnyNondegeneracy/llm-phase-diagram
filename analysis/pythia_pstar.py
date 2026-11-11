"""Linearisation point P* and fit range (Sec. IV, App. E).

P* of a network: median of its document-median trajectory (log10 u, log10 d_C) over the layers at relative depth
>= FROM (default 1/2). The group value P0 is the median over the 8 trained models, (u, d_C) = (1.08, 117).
The fit range is the absolute window 0.3 <= u <= 3, 30 <= d_C <= 400; OFF stores it as offsets from P0 (row 0: u,
row 1: d_C; columns: lower, upper), so that a fit at another point P uses the window P + OFF (used for Pythia, App. E)."""
import numpy as np
from runs import L

P0 = np.array([0.03371, 2.06960])
OFF = np.array([[np.log10(0.3), np.log10(3)], [np.log10(30), np.log10(400)]]) - P0[:, None]
FROM = 0.5                     # relative depth where the late segment starts; App. F 'last third' sets it to 2/3


def pstar(Z, docs):
    """P* of one network (Z: its moments arrays) from the documents in docs"""
    xu = L(np.median(Z['u'][:, docs], 1)); xd = L(np.median(Z['dC'][:, docs], 1)); n = len(xu)
    k = int(np.ceil(FROM * (n - 1) - 1e-9))          # first layer with relative depth >= FROM
    return np.array([np.median(xu[k:]), np.median(xd[k:])])
