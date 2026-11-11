"""App. B figure: fates of the 156 untrained runs (analysis/fates_share.py, fates_share.json). The deep points of a run
fall into three classes (working region; collapse or noise side of the line through P* along the restoring
eigen-direction of the untrained group's J), and the run takes the class with the largest share.
(a) the late-half positions coloured by fate, with the working phase (light red, smoothed P > 0.5) and the dividing
    line (dashed).
(b) share of runs in each fate against the initialisation scale sigma, F and G families pooled, each run weighted
    equally, Gaussian kernel (sd 0.15) in log10 sigma; stacked areas; ticks at the sampled sigma."""
from common import *
from scipy.ndimage import gaussian_filter
B = np.load(ana('phase_band_final.npz')); gx, gy = B['gx'], B['gy']
_J = json.load(open(ana('fig4_J.json')))['untrained']; _V = np.array(_J['V']); _vs = _V[:, int(np.argmin(_J['ev_re']))]
_Ps = np.array([np.log10(1.08), np.log10(117)])
R = [(np.array(r['median']), r['sigma'], r['fate'], r['family']) for r in json.load(open(ana('fates_share.json')))['runs'].values()]
COL = dict(collapse='#D9B45F', working=TRN, noise='#7FAF76'); FILL = dict(collapse='#EBD49F', working=PB_TINT_W, noise='#C9DEC3')
fig, axs = plt.subplots(1, 2, figsize=(W1, 1.75), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.42, left=0.11, right=0.98, top=0.88, bottom=0.22))
ax = axs[0]
m_ = np.isfinite(B['P']); w_ = gaussian_filter(m_.astype(float), 1.5)
Ps_ = np.where(w_ > 0.5, gaussian_filter(np.where(m_, B['P'], 0.0), 1.5) / np.maximum(w_, 1e-9), np.nan)
ax.contourf(gx, gy, Ps_, levels=[0.5, 2], colors=[PB_TINT_W], zorder=0)
t = np.linspace(-3, 3, 10); ax.plot(_Ps[0] + t * _vs[0], _Ps[1] + t * _vs[1], ls=(0, (3, 2)), color='0.35', lw=0.7, zorder=1)
for m, sg, c, _ in sorted(R, key=lambda r: r[2] == 'working'): ax.plot(*m, 'o', ms=3.2 if c == 'working' else 2.4, color=COL[c], mec='white' if c == 'working' else 'none', mew=0.4, zorder=8 if c == 'working' else 3)
pstar(ax, ms=5.5)
ax.set_xlim(-2.2, 1.45); ax.set_ylim(0, 3.6)
ax.set_xticks([-2, -1, 0, 1]); ax.set_xticklabels([r'$10^{-2}$', r'$10^{-1}$', '1', '10']); ax.set_yticks([0, 1, 2, 3]); ax.set_yticklabels(['1', '10', r'$10^2$', r'$10^3$'])
ax.set_xlabel(r'$u$', fontsize=FS_AXIS, labelpad=1); ax.set_ylabel(r'$d_C$', fontsize=FS_AXIS, labelpad=1); ax.tick_params(labelsize=FS_KEY)
ax.text(-0.25, 0.35, 'collapse', color=COL['collapse'], fontsize=FS_KEY, ha='center'); ax.text(1.3, 3.3, 'noise', color=COL['noise'], fontsize=FS_KEY, ha='right')
ax.annotate('working', xy=(-0.3, 1.3), xytext=(-1.1, 0.75), fontsize=FS_KEY, color=TRND, ha='center', arrowprops=dict(arrowstyle='-', color=TRND, lw=0.5))
ax = axs[1]; NB = {sg: sum(1 for r in R if r[1] == sg) for sg in set(r[1] for r in R)}
ls = np.log10([r[1] for r in R]); xs = np.linspace(-2.4, 0.1, 300); K = np.exp(-0.5 * ((xs[:, None] - ls[None]) / 0.15) ** 2)
sh = {c: (K * np.array([r[2] == c for r in R])[None]).sum(1) / K.sum(1) for c in ('collapse', 'working', 'noise')}
ax.stackplot(xs, sh['collapse'], sh['working'], sh['noise'], colors=[FILL['collapse'], FILL['working'], FILL['noise']], lw=0)
ax.text(-2.3, 0.3, 'collapse', color='#8A6414', fontsize=FS_KEY); ax.text(-0.45, 0.6, 'noise', color='#4E6B47', fontsize=FS_KEY)
ax.annotate('working', xy=(-1.0, 0.505), xytext=(-1.05, 0.8), fontsize=FS_KEY, color=TRND, arrowprops=dict(arrowstyle='-', color=TRND, lw=0.5))
for sgv in sorted(set(r[1] for r in R)): ax.plot([np.log10(sgv)] * 2, [-0.04, 0], color='0.3', lw=0.6, clip_on=False)
ax.set_xlim(-2.4, 0.1); ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1]); ax.set_xticks([-2, -1, 0]); ax.set_xticklabels(['0.01', '0.1', '1'])
ax.set_xlabel(r'initialization scale $\sigma$', fontsize=FS_AXIS, labelpad=3); ax.set_ylabel('fraction of runs', fontsize=FS_AXIS, labelpad=1); ax.tick_params(labelsize=FS_KEY)
for a_ in axs:
    for sp in ['top', 'right']: a_.spines[sp].set_visible(False)
letter(fig, axs[0], 'a'); letter(fig, axs[1], 'b')
[fig.savefig(fig_path('figB_fates' + _e), dpi=300) for _e in ('.png', '.pdf')]; print('ok', {c: sum(r[2] == c for r in R) for c in COL}, NB)
for fa in 'FG':
    print(fa, [(sg, {c: sum(1 for r in R if r[3] == fa and r[1] == sg and r[2] == c) for c in COL}) for sg in sorted(set(r[1] for r in R if r[3] == fa))])
