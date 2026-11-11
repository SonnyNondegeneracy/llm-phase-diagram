"""Shared style of the paper figures (APS / PRX: full width = REVTeX \\textwidth 510 pt, single column 246 pt;
STIX (Times-design) text and math; panel labels "(a)", "(b)" in roman type; no titles, no grids).

Colour roles (identical in every figure): blue = untrained, red = trained (with a second encoding for greyscale);
INK = the object under discussion; TRAINED = individual trained models; EDGE = working phase in the schematic;
P* = gold star with ink edge."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': ['STIXGeneral'], 'mathtext.fontset': 'stix', 'axes.unicode_minus': True,
    'font.size': 7.5, 'axes.labelsize': 8.5, 'axes.titlesize': 8.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'lines.linewidth': 1.5, 'hatch.linewidth': 0.3, 'savefig.dpi': 300, 'legend.frameon': False, 'axes.titlepad': 3})

W = 510 / 72.27   # full width (\textwidth, 510 pt)
W1 = 246 / 72.27  # single column (\columnwidth, 246 pt)
INK = '#1a1a1a'; MUTED = '#5d6168'; GREY = '#8f949c'; TRAINED = '#b9bdc4'
WALLL = '#6b7078'; EDGE = '#c8553d'
UNT, UNTL, UNTD = '#2f6aa3', '#9dbbe2', '#1f4a80'          # untrained: base, light, dark
TRN, TRNL, TRND = '#c4472b', '#ecb3a5', '#8a2f1d'          # trained: base, light, dark
UNT_TINT, TRN_TINT = '#e6ebf2', '#f6e6e1'                  # background tints of the drifting / working phase
PSTAR_FILL, PSTAR_EDGE = '#f2c14e', INK
FS_LETTER, FS_AXIS, FS_TEXT = 9, 8.5, 7.5                # panel letters | axis and colour-bar titles | everything else
FS_KEY = FS_TEXT
LEG = dict(fontsize=FS_TEXT, frameon=False, handlelength=1.8, handletextpad=0.5, labelspacing=0.35, borderaxespad=0.4)   # one legend style


def letter(fig, ax, s, dx=-0.0, dy=0.0):
    """panel label "(s)" just above the top-left corner of an axes (figure coordinates)"""
    bb = ax.get_position()
    fig.text(bb.x0 + dx, bb.y1 + 0.012 + dy, f'({s})', fontsize=FS_LETTER, ha='left', va='bottom')
