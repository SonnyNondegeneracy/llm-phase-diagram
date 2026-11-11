"""Helpers shared by the analysis scripts: the log coordinates, the family of a moments file, and the late-half
position of a run.

Coordinates: x = (log10 u, log10 d_C). A run's document-median trajectory is the per-layer median over the 96
documents. Its late-half position is the median of that trajectory over the layers at relative depth >= 1/2."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import RAW, raw, out

L = lambda x: np.log10(np.clip(x, 1e-12, None))


def late(f):
    """late-half position and last-layer position of the document-median trajectory of run file f"""
    Z = np.load(f); xu = L(np.median(Z['u'], 1)); xd = L(np.median(Z['dC'], 1)); n = len(xu); k = int(np.ceil(0.5 * (n - 1) - 1e-9))
    return np.array([np.median(xu[k:]), np.median(xd[k:])]), np.array([xu[-1], xd[-1]])


def fam(f):
    """(family, architecture) of a moments file.
    native__<arch>__<seed>        -> 'F0.02(native)'  (the model's own initialisation, F family, sigma 0.02)
    init__<arch>__<seed>          -> 'G0.02'          (all weights N(0, 0.02^2), G family)
    unit__<arch>__<seed>          -> 'G1(N01)'        (all weights N(0, 1))
    sweep__<arch>__<F|G><s>__<seed> -> 'F<s>' / 'G<s>'  (initialisation-scale sweep; 'p' stands for the decimal point)
    trained__<arch>__0            -> 'trained'
    anything else (Pythia checkpoints)       -> (None, None)"""
    b = os.path.basename(f)[:-4]; p = b.split('__')
    if p[0] == 'native': return 'F0.02(native)', p[1]
    if p[0] == 'init': return 'G0.02', p[1]
    if p[0] == 'unit': return 'G1(N01)', p[1]
    if p[0] == 'sweep': return p[2].replace('p', '.'), p[1]
    if p[0] == 'trained': return 'trained', p[1]
    return None, None
