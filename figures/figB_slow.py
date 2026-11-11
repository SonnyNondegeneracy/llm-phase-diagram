"""App. B figure: untrained flows slow down with depth. Per-layer displacement |x_t - x_{t-1}| of the document-median
trajectory in (log10 u, log10 d_C) against layer index t; median over the 156 untrained runs (line) and 16-84 % across
runs (band), shown where >= 20 runs reach that layer; trained models (8, >= 3 reaching) for comparison;
dotted: slope -1 (1/t)."""
from common import *
sys.path.insert(0, ANALYSIS)
from runs import fam
S = {'u': [], 't': []}
for f in sorted(glob.glob(f'{RAW}/*.npz')):
    fm, a = fam(f)
    if fm is None: continue
    Z = np.load(f); X = np.stack([np.median(L10(Z['u']), 1), np.median(L10(Z['dC']), 1)], 1)
    S['t' if fm == 'trained' else 'u'].append(np.linalg.norm(np.diff(X, axis=0), axis=1))
fig, ax = plt.subplots(figsize=(W1, 1.9)); fig.subplots_adjust(left=0.16, right=0.97, top=0.95, bottom=0.22)
for key, col, lab in [('u', UNT, 'untrained (156)'), ('t', TRN, 'trained (8)')]:
    T = max(len(s) for s in S[key]); M = np.full((len(S[key]), T), np.nan)
    for i, s in enumerate(S[key]): M[i, :len(s)] = s
    t = np.arange(1, T + 1); ok = np.sum(np.isfinite(M), 0) >= (3 if key == 't' else 20)
    med = np.nanmedian(M, 0); lo, hi = np.nanpercentile(M, [16, 84], 0)
    ax.fill_between(t[ok], lo[ok], hi[ok], color=col, alpha=0.18, lw=0); ax.plot(t[ok], med[ok], color=col, lw=1.1, label=lab)
tt = np.array([4, 40]); ax.plot(tt, 0.25 * 4 / tt, ls=(0, (1, 1.5)), color=INK, lw=0.9); ax.text(9, 0.25 * 4 / 9 * 1.25, r'$\propto 1/t$', fontsize=FS_KEY)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'layer $t$', fontsize=FS_AXIS); ax.set_ylabel('per-layer displacement (dex)', fontsize=FS_AXIS)
ax.set_xticks([1, 3, 10, 30]); ax.set_xticklabels(['1', '3', '10', '30']); ax.tick_params(labelsize=FS_KEY); ax.legend(fontsize=FS_KEY, frameon=False, loc='lower left')
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
[fig.savefig(fig_path('figB_slow' + _e), dpi=300) for _e in ('.png', '.pdf')]; print('ok')
