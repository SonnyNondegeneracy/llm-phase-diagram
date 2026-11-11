"""Fig. 3: the phase diagram, measured points only. Each dot = one layer of one run (document median of (u, d_C));
colour depth = relative depth (light: first layer, dark: last layer; blue = untrained, red = trained).
(a) untrained: 156 runs (F0.02, G0.02, G1 on 8 architectures x 3 seeds, and the initialisation-scale sweep of 84 runs);
    two arrows = the two drift directions (collapse, noise).
(b) trained: 8 models.
Background (both panels): working (light red) / drifting (light blue) phase from analysis/phase_band_final.py; the two
red lines bound the coexistence band 0.16 < P(working) < 0.84; white = no deep-layer data; star: P*.
Grey: d_C > 5120, above the largest D; hatched: 1536 < d_C < 5120, unreachable for some models only."""
from common import *
from matplotlib.colors import LinearSegmentedColormap
fig, axs = plt.subplots(1, 2, figsize=(W, 3.05), sharey=True, gridspec_kw=dict(wspace=0.05, left=0.065, right=0.9, top=0.93, bottom=0.13))
XL, YL = (-2.25, 1.45), (0.0, 3.95)            # limits = data extent plus a small margin (no empty map)
CMU = LinearSegmentedColormap.from_list('du', ['#c9d8ec', UNT, '#16355e']); CMT = LinearSegmentedColormap.from_list('dt', ['#f2cfc6', TRN, '#6e2416'])   # depth: light -> dark, blue = untrained, red = trained

def base(ax):
    frame_full(ax, XL, YL); phase_band(ax)
    # unreachable region d_C > D: D differs between models (1536-5120): hatched band over that range, shown in full
    ax.axhspan(np.log10(DMIN), np.log10(DMAX), fc='none', ec=WALLL, hatch='////', lw=0, zorder=0.3, alpha=0.55)   # transition: d_C > D for some models
    ax.axhspan(np.log10(DMAX), 9, fc=WALLL, ec='none', lw=0, zorder=0.3, alpha=0.3)                               # d_C > D for every model
    ax.axhline(np.log10(DMIN), color=WALLL, lw=0.4, zorder=0.3); ax.axhline(np.log10(DMAX), color=WALLL, lw=0.4, zorder=0.3)
    ax.set_xlim(*XL); ax.set_ylim(*YL)
    ax.set_xticks([-2, -1, 0, 1]); ax.set_xticklabels([r'$10^{-2}$', r'$10^{-1}$', '1', '10'])


def dots(ax, trs, s, CM):
    X = np.concatenate([t[0] for t in trs]); Y = np.concatenate([t[1] for t in trs]); D = np.concatenate([np.linspace(0, 1, len(t[0])) for t in trs]); o = np.argsort(D)
    ax.scatter(X[o], Y[o], c=CM(D[o]), s=s, lw=0, zorder=3)
unt = []
for f in sorted(glob.glob(f'{RAW}/sweep__*.npz')): Z = np.load(f); unt.append((L10(np.median(Z['u'], 1)), L10(np.median(Z['dC'], 1))))
for g in ['native', 'init', 'unit']: unt += [(t['x'], t['y']) for t in trajectories(g)]
ax = axs[0]; base(ax); dots(ax, unt, 3.0, CMU); pstar(ax, ms=8); letter(fig, ax, 'a')
import matplotlib.patheffects as pe, math
from matplotlib.patches import FancyArrowPatch
# schematic arrows of (a): from near P* toward the median late-half position of the untrained runs that end beyond the
# fit range on the large-u / small-d_C side (collapse) and on the small-u / large-d_C side (noise), shifted sideways so
# that they cover few dots; end points in data coordinates
_xs = np.linspace(*XL, 500); _ys = np.linspace(*YL, 500); _X, _Y = np.meshgrid(_xs, _ys)
_PL = {'noise': ((-0.570, 2.293), (-1.173, 2.702)), 'collapse': ((0.310, 1.399), (0.742, 0.756))}
for t_ in ['noise', 'collapse']:
    a_, b_ = np.array(_PL[t_][0]), np.array(_PL[t_][1])
    pa = FancyArrowPatch(a_, b_, arrowstyle='simple,head_width=1.7,head_length=1.5,tail_width=0.75', mutation_scale=10, fc='none', ec='none', zorder=4)
    ax.add_patch(pa)
    L_ = np.hypot(*(b_ - a_)); u_ = (b_ - a_) / L_; tt = ((_X - a_[0]) * u_[0] + (_Y - a_[1]) * u_[1]) / L_
    img = np.zeros(_X.shape + (4,)); img[..., :3] = np.array([0x4a, 0x52, 0x5c]) / 255; img[..., 3] = np.clip(tt / 0.55, 0, 1) * 0.85  # tail fades in
    ax.imshow(img, extent=[*XL, *YL], origin='lower', zorder=4, aspect='auto', interpolation='bilinear').set_clip_path(pa)
    q0, q1 = ax.transData.transform(a_), ax.transData.transform(b_); ang = np.degrees(np.arctan2(q1[1] - q0[1], q1[0] - q0[0]))
    if ang > 90: ang -= 180
    if ang < -90: ang += 180
    nv = np.array([-(q1 - q0)[1], (q1 - q0)[0]]); nv /= np.hypot(*nv)                 # display-space normal
    side = -1 if nv[1] > 0 else 1                                                       # put the label on the lower (empty) side of the shaft
    mid = ax.transData.transform((a_ + b_) / 2 + 0.08 * u_) + side * 19 * nv           # 19 px off the shaft centre line
    ax.text(*ax.transData.inverted().transform(mid), t_, fontsize=FS_TEXT, color=INK, zorder=13, ha='center', va='center',
            rotation=ang, rotation_mode='anchor', path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
ax.set_xlim(*XL); ax.set_ylim(*YL)
ax = axs[1]; base(ax); dots(ax, [(t['x'], t['y']) for t in trajectories('trained')], 6, CMT); pstar(ax, ms=8); ax.set_ylabel(''); letter(fig, ax, 'b')
# direct labels (no legend): each D band named inside itself at the right end of (b); P* and the phases named
LB = dict(fontsize=FS_TEXT, ha='right', va='center', zorder=12)
ax.text(XL[1] - 0.06, (np.log10(DMAX) + YL[1]) / 2, r'$d_C > %d$' % DMAX, color='0.3', **LB)
ax.text(XL[1] - 0.06, (np.log10(DMIN) + np.log10(DMAX)) / 2, r'$%d < d_C \leq %d$' % (DMIN, DMAX), color='0.3', bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.85), **LB)
ax.text(PSTAR[0] + 0.12, PSTAR[1] + 0.1, r'$P^*$', color=INK, fontsize=FS_TEXT, ha='left', va='bottom', zorder=12, path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])
ax.text(-0.13, 1.25, 'working\nphase', color=TRND, fontsize=FS_TEXT, ha='left', va='center', zorder=12, linespacing=0.95, path_effects=[pe.withStroke(linewidth=2.2, foreground=PB_TINT_W)])
ax.text(-1.95, 1.55, 'drifting\nphase', color=UNTD, fontsize=FS_TEXT, ha='left', va='center', zorder=12, linespacing=0.95, bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.85))
for k_, cm_ in enumerate([CMU, CMT]):                 # two thin depth ramps side by side (blue: untrained, red: trained), one label
    cax = fig.add_axes([0.915 + 0.011 * k_, 0.13, 0.009, 0.8]); cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cm_), cax=cax); cb.outline.set_linewidth(0.3)
    if k_ == 0: cb.set_ticks([])
    else: cb.set_ticks([0, 1]); cb.set_ticklabels(['first', 'last']); cb.set_label('relative depth', fontsize=FS_AXIS); cb.ax.tick_params(labelsize=FS_KEY)
[fig.savefig(fig_path('fig3_phase_diagram' + _e), dpi=300) for _e in ('.png', '.pdf')]; print('ok')
