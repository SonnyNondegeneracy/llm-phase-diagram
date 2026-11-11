"""App. F figure: choice of the neighborhood size k of the phase map (analysis/phase_k_choice.py, phase_k_choice.json):
leave-one-architecture-out log loss (class-balanced, mean over 8 architectures; bars = SD / sqrt(8) across them)."""
from common import *
D = json.load(open(ana('phase_k_choice.json')))
K = np.array(sorted(int(k) for k in D['loss'])); L = np.array([D['loss'][str(k)] for k in K])
SE = np.array([np.std(D['loss_per_arch'][str(k)]) / np.sqrt(len(D['loss_per_arch'][str(k)])) for k in K])
fig, ax = plt.subplots(figsize=(W1, 1.7), gridspec_kw=dict(left=0.15, right=0.97, top=0.93, bottom=0.25))
ax.axvline(2000, color='0.55', lw=0.6, ls=(0, (3, 2)), zorder=0.5)
ax.errorbar(K, L, yerr=SE, fmt='none', ecolor=GREY, elinewidth=0.8, capsize=0, zorder=1)
ax.plot(K, L, '-', color=INK, lw=0.9, zorder=2)
ax.plot(K, L, 'o', ms=3.6, color=INK, mec='white', mew=0.6, zorder=3)
ax.text(2000 * 1.06, 0.278, r'$k=2000$', ha='left', va='top', fontsize=FS_KEY, color=MUTED)
ax.set_xscale('log'); ax.set_xlim(38, 4200); ax.set_ylim(0.1, 0.29); ax.set_yticks([0.1, 0.15, 0.2, 0.25])
ax.set_xlabel(r'neighborhood size $k$', fontsize=FS_AXIS, labelpad=1); ax.set_ylabel('held-out log loss', fontsize=FS_AXIS)
ax.tick_params(labelsize=FS_KEY)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
[fig.savefig(fig_path('figF_k' + _e), dpi=300) for _e in ('.png', '.pdf')]; print('ok')
