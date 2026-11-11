"""App. F figure: -ln rho of Pythia-1.4B binned joint fits under different binnings and linearisation points
(analysis/pythia_bins.py, pythia_bins.json): black = main bins (Fig. 5c), orange = half decade, green = one decade,
all at (0.63, 27); purple = main bins at (1.06, 79). Point = bin fit, vertical bar = 16-84 % of the two-level
bootstrap, horizontal line = the bin's step range."""
from common import *
D = json.load(open(ana('pythia_bins.json')))
def _rows(lst):
    out = []
    for r in lst:
        st = r['steps']; b = str(st[0]) if len(st) == 1 else f'{st[0]}--{st[-1]}'
        out.append(dict(bin=b, R=[r['R'], r['R_q'][1] - r['R'], r['R'] - r['R_q'][0]]))
    return out
blocks = [_rows(D[k]) for k in ('main', 'half_decade', 'one_decade', 'main_at_late')]
NAMES = ['Fig. 5(c) binning', 'half-decade binning', 'decade binning', 'Fig. 5(c) binning at (1.06, 79)']
COLS = [INK, '#d08b3e', '#7aa37a', '#a67fb8']   # no blue / red: those mark the untrained / trained groups
X0 = 1.3
def span(b):
    p = [int(x) for x in b.replace('--', '-').split('-')]; return [np.log10(v) if v > 0 else X0 for v in p]
fig, ax0 = plt.subplots(figsize=(W1, 2.35), gridspec_kw=dict(left=0.17, right=0.98, top=0.8, bottom=0.17)); axs = [ax0]
for ax, key, lab in [(axs[0], 'R', r'contraction rate $-\ln\rho$')]:
    ax.axvspan(1.0, np.log10(np.sqrt(256 * 512)), color=UNT_TINT, lw=0, zorder=0); ax.axvspan(np.log10(np.sqrt(256 * 512)), 5.35, color=TRN_TINT, lw=0, zorder=0)
    ax.axhline(0, color='0.3', lw=0.6, ls=(0, (3, 2)), zorder=0.5)
    for k, (bl, col) in enumerate(zip(blocks, COLS)):
        dx = (k - 1.5) * 0.04
        for r in bl:
            sp = span(r['bin']); x = np.mean(sp) + dx; v, hi, lo = r[key]
            if len(sp) > 1 and sp[1] > sp[0]: ax.plot(sp, [v, v], color=col, lw=0.7, alpha=0.7, zorder=2)
            z0 = r['bin'] == '0'   # step 0 (initialisation): open marker, as in Figs. 5 and A
            ax.errorbar([x], [v], yerr=[[lo], [hi]], fmt='o', ms=3.2, color=col, mfc='white' if z0 else col, mec=col if z0 else 'white', mew=0.6 if z0 else 0.4, elinewidth=0.7, capsize=1.5, zorder=3)
    ax.set_xlim(1.0, 5.35); ax.set_xticks([X0, 2, 3, 4, 5]); ax.set_xticklabels(['0', r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$'])
    ax.set_xlabel('training step', fontsize=FS_AXIS); ax.set_ylabel(lab, fontsize=FS_AXIS); ax.tick_params(labelsize=FS_KEY)
    for sp_ in ['top', 'right']: ax.spines[sp_].set_visible(False)
    kw = dict(transform=ax.get_xaxis_transform(), color=INK, lw=0.5, clip_on=False, zorder=10)   # axis break between step 0 and 64
    for ddx in (-0.025, 0.025): ax.plot([1.65 + ddx - 0.025, 1.65 + ddx + 0.025], [-0.025, 0.025], **kw)
    XB = np.log10(np.sqrt(256 * 512)); top = key == 'R'
    ax.text((1.0 + XB) / 2, 0.97 if top else 0.03, 'drifting', transform=ax.get_xaxis_transform(), ha='center', va='top' if top else 'bottom', fontsize=FS_TEXT, color=UNTD)
    ax.text((XB + 5.35) / 2, 0.97, 'working', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=FS_TEXT, color=TRND)
axs[0].set_ylim(-0.04, 0.085)
hs = [plt.Line2D([], [], color=c, marker='o', ms=3.2, lw=0) for c in COLS]
fig.legend(hs, NAMES, loc="upper center", ncol=2, fontsize=FS_KEY - 0.5, frameon=False, bbox_to_anchor=(0.53, 1.0), handletextpad=0.2, columnspacing=0.9)
[fig.savefig(fig_path('figF_bins' + _e), dpi=300) for _e in ('.png', '.pdf')]; print('ok', [len(b) for b in blocks])
