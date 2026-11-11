"""Fig. 2: the coordinates and how they behave in trained models.
(a) geometry: offset delta of a document's token mean from the reference (mean of the other documents) and the
    within-document covariance C; u and d_C are the rotation- and scale-invariant combinations.
(b), (c) u and d_C against relative depth for the 8 trained models (document medians; moments/trained__*.npz):
    thin = models, thick = median over models.
(d) the same in the (u, d_C) plane: thin = models, thick = median path over models (the thick lines of (b), (c)),
    open circle / dot = its first / last layer; star: P*."""
from common import *
from matplotlib.patches import Ellipse
MOD = ['qwen1.5', 'qwen7', 'qwen14', 'mistral7', 'falcon3', 'granite', 'qwen35_4b', 'qwen35_9b']
Z = {m: np.load(f'{RAW}/trained__{m}__0.npz') for m in MOD}
grid = np.linspace(0, 1, 41)
def ondepth(y): t = np.linspace(0, 1, len(y)); return np.interp(grid, t, y)
fig = plt.figure(figsize=(W, 2.2))
gs = fig.add_gridspec(1, 4, width_ratios=[0.9, 1, 1, 1], wspace=0.45, left=0.02, right=0.985, top=0.9, bottom=0.2)
FS = FS_TEXT
# (a)
ax = fig.add_subplot(gs[0]); ax.set_xlim(-3, 3); ax.set_ylim(-2.6, 2.6); ax.set_aspect('equal'); ax.axis('off'); rng = np.random.default_rng(1)
for ctr in [(-1.9, -1.0), (1.9, -1.5), (-1.7, 1.5), (0.2, -2.0)]:
    p = rng.multivariate_normal(ctr, np.eye(2) * 0.09, 35); ax.scatter(p[:, 0], p[:, 1], s=1, color='0.78', lw=0)
ref = np.array([-0.3, -0.5]); ax.plot(*ref, marker='+', color=INK, ms=7, mew=1.1, zorder=5)
ax.text(ref[0] + 0.1, ref[1] - 0.25, 'reference', fontsize=FS, ha='center', va='top', zorder=4)
mu = np.array([1.3, 1.2]); cov = [[0.42, 0.26], [0.26, 0.3]]; p = rng.multivariate_normal(mu, cov, 90); ax.scatter(p[:, 0], p[:, 1], s=1.6, color='0.55', lw=0, zorder=3)
w, V = np.linalg.eigh(cov); ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
ax.add_patch(Ellipse(mu, 4 * np.sqrt(w[1]), 4 * np.sqrt(w[0]), angle=ang, fill=False, ec='0.4', lw=0.7))
ax.annotate('', xy=mu, xytext=ref, arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.3, mutation_scale=8, shrinkA=0, shrinkB=0), zorder=6)
ax.text(0.3, 0.6, r'$\delta$', color=INK, fontsize=FS_AXIS, ha='right', va='center', zorder=7, bbox=dict(fc='white', ec='none', alpha=0.8, pad=0.3))
ax.text(2.55, 1.9, r'$C$', color=INK, fontsize=FS_AXIS, ha='left', va='center')
ax.text(0, -2.75, r'$u=\dfrac{|\delta|^2}{\sqrt{\mathrm{tr}\,C^2}}$    $d_C=\dfrac{(\mathrm{tr}\,C)^2}{\mathrm{tr}\,C^2}$', fontsize=FS_AXIS + 0.5, ha='center', va='top')
AXA = ax
# (b), (c)
for i, (key, lab, yl, yt) in enumerate([('u', r'$u$', (np.log10(0.05), np.log10(3)), [0.1, 0.3, 1, 3]), ('dC', r'$d_C$', (np.log10(20), np.log10(400)), [30, 100, 300])]):
    ax = fig.add_subplot(gs[1 + i]); Y = []
    if i == 0: AXB = ax
    for m in MOD:
        y = ondepth(L10(np.median(Z[m][key], 1))); Y.append(y); ax.plot(grid, y, color=TRAINED, lw=0.6, zorder=2)
    ax.plot(grid, np.median(Y, 0), color=INK, lw=1.5, zorder=3)
    ax.set_ylim(*yl); ax.set_yticks(np.log10(yt)); ax.set_yticklabels([f'{v:g}' for v in yt]); ax.minorticks_off()
    ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_xlabel('relative depth'); ax.set_ylabel(lab, labelpad=1)
    letter(fig, ax, 'bc'[i])
# (d) the same trajectories in the (u, d_C) plane: thin = models (per-layer document medians), thick = median path
ax = fig.add_subplot(gs[3]); XS, YS = [], []
for m in MOD:
    x = L10(np.median(Z[m]['u'], 1)); y = L10(np.median(Z[m]['dC'], 1)); XS.append(x); YS.append(y); ax.plot(x, y, color=TRAINED, lw=0.6, zorder=2)
MX = np.median([ondepth(x) for x in XS], 0); MY = np.median([ondepth(y) for y in YS], 0)
ax.plot(MX, MY, color=INK, lw=1.5, zorder=3); ax.plot(MX[0], MY[0], 'o', ms=4, mfc='white', mec=INK, mew=0.9, zorder=4); ax.plot(MX[-1], MY[-1], 'o', ms=3.6, color=INK, mew=0, zorder=4)
pstar(ax, ms=8)
ax.set_xlim(np.log10(0.05), np.log10(3)); ax.set_ylim(np.log10(20), np.log10(400))          # same ranges and ticks as (b) and (c)
ax.set_xticks(np.log10([0.1, 0.3, 1, 3])); ax.set_xticklabels(['0.1', '0.3', '1', '3']); ax.set_yticks(np.log10([30, 100, 300])); ax.set_yticklabels(['30', '100', '300']); ax.minorticks_off()
ax.set_xlabel(r'$u$'); ax.set_ylabel(r'$d_C$', labelpad=1)
LK = LEG
_lg = ax.legend([plt.Line2D([], [], marker='o', ms=4, mfc='white', mec=INK, mew=0.9, lw=0), plt.Line2D([], [], marker='o', ms=3.6, color=INK, lw=0),
           plt.Line2D([], [], marker='*', ms=7, color=PSTAR_FILL, mec=PSTAR_EDGE, mew=0.6, lw=0)], ['first layer', 'last layer', r'$P^*$'], loc='lower right', bbox_to_anchor=(0.985, 0.15), **{**LK, 'frameon': True, 'framealpha': 0.85, 'borderpad': 0.25, 'borderaxespad': 0.0})   # translucent white backing over the data
_lg.get_frame().set(edgecolor='none', facecolor='white', boxstyle='round,pad=0.1')
AXB.legend([plt.Line2D([], [], color=TRAINED, lw=0.7), plt.Line2D([], [], color=INK, lw=1.5)], ['model', 'median'], loc='lower right', **LK)
letter(fig, ax, 'd'); letter(fig, AXA, 'a', dy=AXB.get_position().y1 - AXA.get_position().y1)   # (a) has no frame: align with the others
[fig.savefig(fig_path('fig2_coordinates' + _e), dpi=300) for _e in ('.png', '.pdf')]; print('ok')
