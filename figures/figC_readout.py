"""App. C figure: identification ability rises with E. 5 trained models, blocks 1..L-1 (155 blocks).
x: E per block (median over the 96 documents; moments/trained__<model>__0.npz). y: sensitivity index d' from the 96-way
logistic-probe test accuracy P (probe/probe_<model>.json, regularisation selected on the validation split), with
P(s) = int phi(z) Phi(z + s)^95 dz (equal-variance Gaussian model).
Dashed: least-squares power law d' = c E^b over all blocks."""
from common import *
from scipy.stats import norm
from scipy.optimize import brentq
z = np.linspace(-8, 8, 4001); dz = z[1] - z[0]
def P_of_s(s): return np.sum(norm.pdf(z) * norm.cdf(z + s) ** 95) * dz
def s_of_P(P): return brentq(lambda s: P_of_s(s) - P, -2, 20) if P > 1 / 96 + 1e-6 else np.nan
MOD = [('qwen7', 'Qwen2.5-7B'), ('qwen35_9b', 'Qwen3.5-9B'), ('mistral7', 'Mistral-7B-v0.3'), ('falcon3', 'Falcon3-7B'), ('granite', 'Granite-3.1-8B')]
fig, ax = plt.subplots(figsize=(W1, 2.3)); fig.subplots_adjust(left=0.16, right=0.97, top=0.96, bottom=0.2)
allE, allS = [], []
cols = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00']   # categorical (Okabe-Ito), one hue per model
for (m, lab), c in zip(MOD, cols):
    Z = np.load(f'{RAW}/trained__{m}__0.npz'); E = np.median(Z['E'], 1)
    P = {L['state']: L['logistic']['selected']['acc_test'] for L in json.load(open(f'{PROBE}/probe_{m}.json'))['layers']}
    S = np.array([s_of_P(P[str(i + 1)]) for i in range(len(E))]); ok = np.isfinite(S) & (E > 0)
    ax.scatter(E[ok], S[ok], s=7, color=c, lw=0, label=lab, zorder=3); allE += list(E[ok]); allS += list(S[ok])
allE, allS = np.array(allE), np.array(allS)
b, a = np.polyfit(np.log(allE), np.log(allS), 1); r2 = 1 - np.var(np.log(allS) - (a + b * np.log(allE))) / np.var(np.log(allS))
xx = np.geomspace(allE.min(), allE.max(), 50); ax.plot(xx, np.exp(a) * xx ** b, ls=(0, (3, 2)), color=INK, lw=0.8, zorder=2)
ax.text(0.04, 0.93, r"$d^{\prime}\propto E^{%.2f}$, $R^2=%.2f$" % (b, r2), transform=ax.transAxes, fontsize=FS_TEXT, va='top')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'$E$', fontsize=FS_AXIS); ax.set_ylabel(r"sensitivity index $d^{\prime}$", fontsize=FS_AXIS)
from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
for A, T in [(ax.xaxis, [0.01, 0.02, 0.05, 0.1, 0.2]), (ax.yaxis, [1.5, 2, 3, 4])]:
    A.set_major_locator(FixedLocator(T)); A.set_major_formatter(FixedFormatter([f'{t:g}' for t in T])); A.set_minor_formatter(NullFormatter())
ax.tick_params(labelsize=FS_KEY); ax.legend(fontsize=FS_KEY - 1, frameon=False, loc='lower right', handletextpad=0.2, markerscale=1.6)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
[fig.savefig(fig_path('figC_readout' + _e), dpi=300) for _e in ('.png', '.pdf')]; print('ok b=%.3f R2=%.3f n=%d' % (b, r2, len(allE)))
