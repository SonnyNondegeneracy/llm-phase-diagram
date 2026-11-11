"""App. A figure: Pythia-1.4B along training. Per checkpoint, the late-half (relative depth >= 1/2) median of each
document, then the median over the 96 documents; bars = 16-84 % of a document bootstrap (200 draws, seed 0).
(a) d_C; (b) E = u / (sqrt(d_C) + u). Background: phases of Fig. 5 (boundary between steps 256 and 512); step 0 (open
marker) is drawn left of the axis break."""
import re
from common import *
step = lambda f: int(re.search(r'step(\d+)', os.path.basename(f)).group(1))
rng = np.random.default_rng(0); out = []
for f in sorted(glob.glob(f'{RAW}/pythia__step*__0.npz'), key=step):
    st = step(f); Z = np.load(f); n = Z['dC'].shape[0]; k = int(np.ceil(0.5 * (n - 1) - 1e-9))
    d = np.median(Z['dC'][k:], 0); u = np.median(Z['u'][k:], 0); e = u / (np.sqrt(d) + u)
    res = []
    for arr in (d, e):
        b = [np.median(arr[rng.integers(0, len(arr), len(arr))]) for _ in range(200)]; res.append((np.median(arr), *np.percentile(b, [16, 84])))
    out.append((st, res))
X0 = 1.3; XB = np.log10(np.sqrt(256 * 512))
fig, axs = plt.subplots(1, 2, figsize=(W1, 1.7), gridspec_kw=dict(wspace=0.5, left=0.13, right=0.98, top=0.88, bottom=0.25))
for ax, j, lab, log in [(axs[0], 0, r'$d_C$', True), (axs[1], 1, r'$E=u/(\sqrt{d_C}+u)$', False)]:
    ax.axvspan(1.0, XB, color=UNT_TINT, lw=0, zorder=0); ax.axvspan(XB, 5.35, color=TRN_TINT, lw=0, zorder=0)
    ax.text((1.0 + XB) / 2 + 0.05, 0.97, 'drifting', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=FS_TEXT, color=UNTD)   # in the free corners: drifting top, working bottom
    ax.text((XB + 5.35) / 2, 0.03, 'working', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=FS_TEXT, color=TRND)
    for st, res in out:
        m, lo, hi = res[j]; x = np.log10(st) if st > 0 else X0
        ax.errorbar([x], [m], yerr=[[m - lo], [hi - m]], fmt='o', ms=3, color=INK, mfc='white' if st == 0 else INK, mec=INK if st == 0 else 'white', mew=0.6 if st == 0 else 0.4, elinewidth=0.7, capsize=1.3, zorder=3)
    if log: ax.set_yscale('log'); ax.set_ylim(7, 120); ax.set_yticks([10, 30, 100]); ax.set_yticklabels(['10', '30', '100']); ax.minorticks_off()
    ax.set_xlim(1.0, 5.35); ax.set_xticks([X0, 2, 3, 4, 5]); ax.set_xticklabels(['0', r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$'])
    ax.set_xlabel('training step', fontsize=FS_AXIS, labelpad=1); ax.set_ylabel(lab, fontsize=FS_AXIS, labelpad=1); ax.tick_params(labelsize=FS_KEY - 0.5)
    kw = dict(transform=ax.get_xaxis_transform(), color=INK, lw=0.5, clip_on=False, zorder=10)
    for dx in (-0.04, 0.04): ax.plot([1.65 + dx - 0.04, 1.65 + dx + 0.04], [-0.04, 0.04], **kw)   # axis break between step 0 and 64
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
letter(fig, axs[0], 'a'); letter(fig, axs[1], 'b')
[fig.savefig(fig_path('figA_pythia' + _e), dpi=300) for _e in ('.png', '.pdf')]
for st, res in out: print(st, '%.0f' % res[0][0], '%.3f' % res[1][0])
