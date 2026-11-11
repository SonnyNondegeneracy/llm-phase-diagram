"""Fig. 5: one training run (Pythia-1.4B) between the two phases.
(a) share of each checkpoint's deep points (96 documents x layers at relative depth >= 1/2) in the working phase of
    Fig. 3 (P(working) > 0.5); bars = 16-84 % over documents (analysis/pythia_share.py). The phase boundary lies where
    the share crosses 1/2, between steps 256 and 512; step 0 (open marker) is shaded with the drifting phase (App. F).
(b) inertia coefficient alpha per checkpoint, single-checkpoint fit at Pythia's reference point (0.63, 27), document
    bootstrap (analysis/pythia_ck_fixedP.py).
(c) contraction rate -ln rho per bin of checkpoints (joint fit, two-level bootstrap; analysis/pythia_bins.py, main
    bins); horizontal line = the bin's step range.
Bands in (b), (c): untrained (156) and trained (8) groups at P* (analysis/sens_unified.py, task main).
Dashed lines: zero, at the same height in all panels."""
from common import *
from matplotlib.patches import Patch
RP = {g: dict(gamma=r['gamma'], gamma_q=r['gamma_q'], R=r['R']['explicit'], R_q=r['R_q']['explicit'])   # the two groups (main text)
      for g, r in json.load(open(ana('sens_unified_main.json'))).items()}
SH = json.load(open(ana('pythia_share.json')))
CK = json.load(open(ana('pythia_ck_fixedP.json')))['at_Ppy']
MB = json.load(open(ana('pythia_bins.json')))['main']
X0 = 1.3; XB, XL, XR = np.log10(np.sqrt(256 * 512)), 1.0, 5.35          # x of step 0; phase boundary; x limits
UC, TC = UNT_TINT, TRN_TINT
fig, axs = plt.subplots(1, 3, figsize=(W, 2.45), gridspec_kw=dict(wspace=0.36, left=0.065, right=0.99, top=0.92, bottom=0.19))
xs = lambda s: np.log10(s) if s > 0 else X0
def frame(ax):
    ax.axvspan(XL, XB, color=UC, lw=0, zorder=0); ax.axvspan(XB, XR, color=TC, lw=0, zorder=0)
    ax.text((XL + XB) / 2, 0.97, 'drifting', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=FS_TEXT, color=UNTD)
    ax.text((XB + XR) / 2, 0.97, 'working', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=FS_TEXT, color=TRND)
    ax.set_xlim(XL, XR); ax.set_xticks([X0, 2, 3, 4, 5]); ax.set_xticklabels(['0', r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$']); ax.set_xlabel('training step', fontsize=FS_AXIS)
    kw = dict(transform=ax.get_xaxis_transform(), color=INK, lw=0.6, clip_on=False, zorder=10)
    for dx in (-0.03, 0.03): ax.plot([1.65 + dx - 0.03, 1.65 + dx + 0.03], [-0.04, 0.04], **kw)   # axis break between step 0 and 64
def bands(ax, key, qkey):
    for grp, x0, x1, col in [('untrained', XL, XB, UNT), ('trained', XB, XR, TRN)]:
        r = RP[grp]; m = r[key]; lo, hi = r[qkey]; ax.fill_between([x0, x1], lo, hi, color=col, alpha=0.22, lw=0, zorder=1); ax.plot([x0, x1], [m, m], color=col, lw=1.0, zorder=2)
def pts(ax, X, Y, LO, HI, open_first=False):
    for i, (x, y, lo, hi) in enumerate(zip(X, Y, LO, HI)):
        o = open_first and i == 0
        ax.errorbar([x], [y], yerr=[[max(y - lo, 0)], [max(hi - y, 0)]], fmt='o', ms=3.8, color=INK, mfc='white' if o else INK, mec=INK if o else 'white', mew=0.7 if o else 0.5, elinewidth=0.8, capsize=1.8, capthick=0.7, zorder=5)
# (a) share in the working phase
ax = axs[0]; frame(ax); ax.axhline(0, color='0.3', lw=0.6, ls=(0, (3, 2)), zorder=0.5)
pts(ax, [xs(r['step']) for r in SH], [r['share'] for r in SH], [r['q'][0] for r in SH], [r['q'][1] for r in SH], open_first=True)
ax.set_ylim(-1.25 * 0.3 / 0.7, 1.25); ax.set_yticks([0, 0.5, 1]); ax.set_ylabel('fraction in working phase', fontsize=FS_AXIS); letter(fig, ax, 'a')
# (b) alpha per checkpoint
ax = axs[1]; frame(ax); ax.axhline(0, color='0.3', lw=0.6, ls=(0, (3, 2)), zorder=0.5); bands(ax, 'gamma', 'gamma_q')
pts(ax, [xs(r['step']) for r in CK], [r['gamma'] for r in CK], [r['gamma_q'][0] for r in CK], [r['gamma_q'][1] for r in CK], open_first=True)
ax.set_ylim(-0.7 * 0.3 / 0.7, 0.7); ax.set_ylabel(r'inertia coefficient $\alpha$', fontsize=FS_AXIS); letter(fig, ax, 'b')
# (c) -ln rho per bin
ax = axs[2]; frame(ax); ax.axhline(0, color='0.3', lw=0.6, ls=(0, (3, 2)), zorder=0.5); bands(ax, 'R', 'R_q')
for st in MB:
    xx = np.array([xs(s) for s in st['steps']])
    if len(xx) > 1: ax.hlines(st['R'], xx.min(), xx.max(), color=INK, lw=0.9, zorder=4)
pts(ax, [np.mean([xs(s) for s in st['steps']]) for st in MB], [st['R'] for st in MB], [st['R_q'][0] for st in MB], [st['R_q'][1] for st in MB], open_first=True)
ax.set_ylim(-0.082 * 0.3 / 0.7, 0.082); ax.set_ylabel(r'contraction rate $-\ln\rho$', fontsize=FS_AXIS); letter(fig, ax, 'c')
hs = [plt.Line2D([], [], color=INK, marker='o', ms=3.8, mec='white', mew=0.5, lw=0), plt.Line2D([], [], color=INK, marker='o', ms=3.8, mfc='white', mec=INK, mew=0.7, lw=0), (Patch(color=UNT, alpha=0.22, lw=0), plt.Line2D([], [], color=UNT, lw=1.0)), (Patch(color=TRN, alpha=0.22, lw=0), plt.Line2D([], [], color=TRN, lw=1.0))]
axs[0].legend(hs, ['Pythia', 'Pythia, init.', 'untrained', 'trained'], loc='lower right', **{**LEG, 'labelspacing': 0.15, 'handlelength': 1.3, 'borderaxespad': 0.2})
[fig.savefig(fig_path('fig5_training' + _e), dpi=300) for _e in ('.png', '.pdf')]; print('ok')
