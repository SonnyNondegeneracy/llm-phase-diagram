"""Fig. 4: effective dynamics in the fit range 0.3 <= u <= 3, 30 <= d_C <= 400 (an enlargement of Fig. 3; light red
lines: its coexistence band), (a) untrained vs (b) trained, identical axes and scales.
Decomposition v_{t+1} = alpha v_t + F(x) + eta with one alpha per group (analysis/eta_stats.py, eta_stats.json).
Arrows: F per 0.1-dex cell (median over runs of per-run cell medians of v_{t+1} - alpha v_t; >= 5 transitions per run
and cell, >= 3 runs); arrow length proportional to sqrt(|F|), one scale for both panels; key: 0.1 and 0.01 per layer.
Background: Peclet number Pe = sqrt((F_u/s_u)^2 + (F_d/s_d)^2), s_i = robust SD of eta along axis i; cells without
enough data are hatched.
At P*: eigen-directions of the group's joint J (analysis/fig4_J.py), inward = restoring, outward = growing, visible
length proportional to sqrt(|lambda|); fixed-point type written above (untrained: saddle; trained: stable node)."""
from common import *
import matplotlib.colors as mcol
from matplotlib.colors import LinearSegmentedColormap
JJ = json.load(open(ana('fig4_J.json')))
E = json.load(open(ana('eta_stats.json')))
H, SC = 0.1, 0.27                                     # cell width (dex); arrow length = SC * sqrt(|F|) (dex)
# Pe colour scale: light greys, reversed so that low Pe (noise-dominated) is the deepest and cannot be mistaken for an
# empty (white) cell
PE = LinearSegmentedColormap.from_list('pe', ['#ffffff', '#f2f3f5', '#e1e4e8', '#c8cdd4'][::-1]); PN = mcol.LogNorm(0.25, 4)
NODATA_HATCH = '0.72'; BX = (-0.5, 0.5, 1.5, 2.6)   # axes snapped to the 0.1-dex cell edges that cover the fit range
plt.rcParams['hatch.linewidth'] = 0.4
fig = plt.figure(figsize=(W, 3.0))
axs = [fig.add_axes([0.065, 0.13, 0.4, 0.8]), fig.add_axes([0.505, 0.13, 0.4, 0.8])]; cax = fig.add_axes([0.925, 0.13, 0.012, 0.8])
QK = dict(angles='xy', scale_units='xy', scale=1, width=0.0042, headwidth=3.6, headlength=4.2, headaxislength=3.8, minlength=0.1, pivot='tail')
for c, n in enumerate(['untrained', 'trained']):
    ax = axs[c]; d = E[n]
    ax.add_patch(plt.Rectangle((BX[0], BX[2]), BX[1] - BX[0], BX[3] - BX[2], fc='white', ec=NODATA_HATCH, lw=0, hatch='//////', zorder=0.3))   # no-data cells: hatched, distinct from the near-white high-Pe cells
    for k, b in d['bins'].items():
        ctr = np.array([int(i) for i in k.split(',')]) * H + H / 2; F = np.array(b['F'])
        ax.add_patch(plt.Rectangle(ctr - H / 2, H, H, fc=PE(PN(np.clip(b['ratio'], 0.25, 4))), ec=PE(PN(np.clip(b['ratio'], 0.25, 4))), lw=0.3, zorder=0.5))
        Fv = F / np.linalg.norm(F) * SC * np.sqrt(np.linalg.norm(F)); ax.quiver(*ctr, *Fv, color=INK, zorder=4, **QK)
    phase_band(ax, fill=False, lw=0.6, color='#dd9a8a', zorder=3.5)               # the coexistence band of Fig. 3, so that this figure reads as its enlargement
    pstar(ax, ms=8); frame_win(ax); ax.set_xlim(BX[0], BX[1]); ax.set_ylim(BX[2], BX[3]); ax.set_xticks(np.log10([0.5, 1, 2])); ax.set_xticklabels(['0.5', '1', '2']); ax.set_yticks(np.log10([50, 100, 200])); ax.set_yticklabels(['50', '100', '200'])
    letter(fig, ax, 'ab'[c])
    if c == 1: ax.set_ylabel(''); ax.set_yticklabels([])
    # eigen-directions of the group's J at P* (arrows: inward = restoring, outward = growing; length ~ sqrt(|lambda|));
    # fixed-point type written above P*
    AC = UNT if c == 0 else TRN                     # accent follows the population: blue = untrained, red = trained
    j = JJ[n]; J_ = np.array(j['J']); ev, V = np.linalg.eig(J_); ev, V = ev.real, V.real; P_ = np.array(PSTAR)
    vs = [V[:, i] / np.linalg.norm(V[:, i]) for i in range(2)]
    Ls = [0.84 * abs(l) ** 0.5 for l in ev]
    for i in range(2):
        v = vs[i]; lam = ev[i]; Lh = 0.84 * abs(lam) ** 0.5
        for sg in (1, -1):
            tip, tail = P_ + sg * 0.025 * v, P_ + sg * Lh * v
            a_, b_ = (tail, tip) if lam < 0 else (tip, tail)
            ax.annotate('', xy=b_, xytext=a_, arrowprops=dict(arrowstyle='-|>', color=AC, lw=1.4, mutation_scale=8, shrinkA=0, shrinkB=0), zorder=8)
    kind = 'saddle' if ev.min() < 0 < ev.max() else ('stable node' if ev.max() < 0 else 'unstable node')
    ax.text(P_[0], P_[1] + max(Ls) * 0.75 + 0.06, kind, color=AC, fontsize=FS_TEXT, ha='center', va='bottom', zorder=10, bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.85))
x0, y0 = np.log10(1.72), np.log10(320)               # arrow key (trained panel, upper-right corner)
_kt, _kq = [], []
for i, f in enumerate([0.1, 0.01]):
    _kq.append(axs[1].quiver(x0, y0 - 0.07 * i, SC * np.sqrt(f), 0, color=INK, zorder=9, **QK)); _kt.append(axs[1].text(x0 + SC * np.sqrt(f) + 0.015, y0 - 0.07 * i, f'{f:g}', fontsize=FS_KEY, ha='left', va='center', zorder=10))
_kt.append(axs[1].text(x0, y0 + 0.04, r'$|F|$ per layer', fontsize=FS_KEY, ha='left', va='bottom', zorder=10))
fig.canvas.draw(); _tr = axs[1].transData; _inv = _tr.inverted(); _pt = fig.dpi / 72
_bb = [t.get_window_extent() for t in _kt]                             # key content in display units (texts; the arrows start at x0)
_cx0, _cx1 = _tr.transform((x0, 0))[0], max(b.x1 for b in _bb); _cy0, _cy1 = min(b.y0 for b in _bb), max(b.y1 for b in _bb)
_pad, _m = 3.5 * _pt, 4 * _pt                                          # padding inside the box (same on all sides) | gap to the top and right frame (same on both)
_ax = axs[1].get_window_extent(); _dx, _dy = (_ax.x1 - _m - _pad) - _cx1, (_ax.y1 - _m - _pad) - _cy1
_s = _inv.transform((_dx, _dy)) - _inv.transform((0, 0))               # the same shift in data units
for a in _kt: a.set_position(np.array(a.get_position()) + _s)
for q in _kq: q.set_offsets(q.get_offsets() + _s)
_b0 = _inv.transform((_cx0 + _dx - _pad, _cy0 + _dy - _pad)); _b1 = _inv.transform((_cx1 + _dx + _pad, _cy1 + _dy + _pad))
axs[1].add_patch(plt.Rectangle(_b0, *(_b1 - _b0), fc='white', ec='none', alpha=0.85, zorder=8.5))   # opaque backing: no contour line through the key
cb = fig.colorbar(plt.cm.ScalarMappable(norm=PN, cmap=PE), cax=cax); cb.set_label('Péclet number Pe', fontsize=FS_AXIS, labelpad=2)
cb.ax.minorticks_off(); cb.set_ticks([0.25, 0.5, 1, 2, 4]); cb.set_ticklabels(['0.25', '0.5', '1', '2', '4']); cb.ax.tick_params(labelsize=FS_KEY, length=2); cb.outline.set_linewidth(0.4)
[fig.savefig(fig_path('fig4_dynamics' + _e), dpi=300) for _e in ('.png', '.pdf')]; print('ok')
