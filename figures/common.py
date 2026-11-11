"""Shared helpers for the figure scripts: input/output paths, document-median trajectories, the (u, d_C) frame,
P*, and the working / drifting phase map of Fig. 3."""
import os, sys, glob, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from paths import RAW, PROBE, OUT, FIG_OUT, ANALYSIS
from style import *
plt.rcParams['pdf.fonttype'] = 42   # vector PDF: TrueType fonts embedded

L10 = lambda x: np.log10(np.clip(x, 1e-12, None))
PSTAR = (0.03371, 2.06960)                       # P* in (log10 u, log10 d_C): median of the 8 trained models = (1.08, 117)
WIN = (np.log10(0.3), np.log10(3), np.log10(30), np.log10(400))     # fit range of Sec. IV
DMIN, DMAX = 1536, 5120                          # smallest / largest embedding dimension D of the measured networks
# edge-phase colours (light fill, darker text colour)
PH = dict(u0=('#C4D8BE', '#4E6B47'), uinf=('#E9CC8F', '#8A6414'), d1=('#D8C3DE', '#6B4C78'), dD=('#A9D3DF', '#2C6E80'))


def ana(name):
    """path of an analysis output (analysis/out/ by default)"""
    return os.path.join(OUT, name)


def fig_path(name):
    """path of a figure output (figures/out/ by default)"""
    return os.path.join(FIG_OUT, name)


def trajectories(group):
    """document-median trajectories (log10 u, log10 d_C) of every run of a group (moments file prefix)"""
    out = []
    for f in sorted(glob.glob(f'{RAW}/{group}__*.npz')):
        Z = np.load(f); out.append(dict(name=os.path.basename(f)[:-4], x=L10(np.median(Z['u'], 1)), y=L10(np.median(Z['dC'], 1))))
    return out


def frame_full(ax, xl=(-3, 1.8), yl=(0, 3.85)):
    ax.set_xlim(*xl); ax.set_ylim(*yl)
    ax.set_xticks([-3, -2, -1, 0, 1]); ax.set_xticklabels([r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', '1', '10'])
    ax.set_yticks([0, 1, 2, 3]); ax.set_yticklabels(['1', '10', r'$10^{2}$', r'$10^{3}$'])
    ax.set_xlabel(r'$u$'); ax.set_ylabel(r'$d_C$')


def frame_win(ax):
    ax.set_xlim(WIN[0], WIN[1]); ax.set_ylim(WIN[2], WIN[3])
    ax.set_xticks(np.log10([0.3, 1, 3])); ax.set_xticklabels(['0.3', '1', '3']); ax.set_yticks(np.log10([30, 100, 300])); ax.set_yticklabels(['30', '100', '300'])
    ax.set_xlabel(r'$u$'); ax.set_ylabel(r'$d_C$')


def pstar(ax, ms=7, zorder=8):
    """P*: gold star with ink edge"""
    ax.plot(*PSTAR, marker='*', ms=ms * 1.25, color=PSTAR_FILL, mec=PSTAR_EDGE, mew=0.7, zorder=zorder)


# working / drifting phase map (analysis/phase_band_final.py): P(working) on a grid; NaN = no data
_PBD = np.load(ana('phase_band_final.npz')); PB_X, PB_Y, PB_P = _PBD['gx'], _PBD['gy'], _PBD['P']
PB_LEVELS = (0.16, 0.84); PB_TINT_W, PB_TINT_D = '#f3d3ca', '#e3eaf4'


def _pb_smooth(s=1.5):
    """P smoothed with a Gaussian of s grid cells (NaN cells excluded), used only for the band lines"""
    from scipy.ndimage import gaussian_filter
    m = np.isfinite(PB_P); wsum = gaussian_filter(m.astype(float), s)
    Ps = gaussian_filter(np.where(m, PB_P, 0.0), s) / np.maximum(wsum, 1e-9)
    return np.where(wsum > 0.5, Ps, np.nan)


def phase_band(ax, fill=True, lw=0.55, color=None, zorder=6):
    """Fill: light red (working) to light blue (drifting) by P; white = no data. Lines: the two edges of the
    coexistence band 0.16 < P < 0.84."""
    from matplotlib.colors import to_rgb
    if fill:
        m = np.isfinite(PB_P); f = np.where(m, PB_P, 0.0)[..., None]; img = np.zeros(PB_P.shape + (4,))
        img[..., :3] = f * np.array(to_rgb(PB_TINT_W)) + (1 - f) * np.array(to_rgb(PB_TINT_D)); img[..., 3] = m
        ax.imshow(img, extent=[PB_X[0, 0], PB_X[0, -1], PB_Y[0, 0], PB_Y[-1, 0]], origin='lower', aspect='auto', interpolation='bilinear', zorder=0.2)
    ax.contour(PB_X, PB_Y, _pb_smooth(), levels=list(PB_LEVELS), colors=[color or TRN], linewidths=lw, zorder=zorder)
