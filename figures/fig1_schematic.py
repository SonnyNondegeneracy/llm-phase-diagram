"""Fig. 1: schematic (not data).
(a) a transformer layer stack and the token clouds of three documents in the representation space of one block
    (rendered by fig1a_tikz.py to fig1a_panel.png).
(b) the state space (u, d_C): four edge phases at the four sides, each drawn as a glyph of three documents' token
    clouds (+ = document mean), the working phase in the centre. Every edge glyph differs from the working-phase glyph
    in one quantity only (u: separation of the means vs cloud size; d_C: cloud shape; d_C -> 1 rank collapse,
    d_C -> D isotropic).
(c) sketch around P*: the measured shape of the working phase (Fig. 3, P(working) > 0.5) scaled down, and stylised
    trajectories (one dot = one layer, light -> dark from first to last layer): trained (alpha ~ 0) relaxes to P*
    without overshoot and then jitters in place; untrained (alpha > 0) moves with persistent, near-equal steps and
    crosses the working phase without staying.
Also writes fig1_work_points.json (the token clouds of the working-phase glyph), which fig1a_tikz.py draws in (a)."""
import numpy as np
from matplotlib.patches import Circle
import matplotlib.colors as mc
from common import *
rng = np.random.default_rng(7)
# Layout: panels (b), (c) occupy a frame of width W; panel (a) adds A0 x W on the left, so the figure is KW x W wide.
# Type and line sizes are scaled by KW, so that at \textwidth they print at the same sizes as the other figures.
KW = 1.5; A0 = 0.5
for k_ in ('font.size', 'axes.labelsize', 'xtick.labelsize', 'ytick.labelsize', 'legend.fontsize'): plt.rcParams[k_] *= KW
FS_LETTER, FS_AXIS, FS_TEXT = FS_LETTER * KW, FS_AXIS * KW, FS_TEXT * KW
fig = plt.figure(figsize=(W * KW, W * 0.5)); RECT = lambda r: [(A0 + r[0]) / KW, r[1], r[2] / KW, r[3]]
AREC = [0.0, 0.02, (A0 - 0.01) / KW, 0.93]
axP = fig.add_axes(AREC); axP.axis('off')                            # (a): rendered by fig1a_tikz.py
PANEL_A = os.path.join(FIG_OUT, 'fig1a_panel.png')
if os.path.exists(PANEL_A):
    _im = plt.imread(PANEL_A)
    _ink = (_im[..., 3] > 0.02) & (_im[..., :3].min(-1) < 0.985) if _im.shape[-1] == 4 else (_im[..., :3].min(-1) < 0.985)
    _r_, _c_ = np.where(_ink); _im = _im[_r_.min():_r_.max() + 1, _c_.min():_c_.max() + 1]          # trim the white margin
    if _im.ndim == 3 and _im.shape[2] == 4: _im = _im[..., :3] * _im[..., 3:] + (1 - _im[..., 3:])   # flatten onto white: no halo at transparent edges
    axP.imshow(_im, interpolation='lanczos')
    _H, _Wd = _im.shape[:2]; axP.set_xlim(-0.10 * _Wd, _Wd); axP.set_ylim(_H, 0)      # room on the left for the residual-stream arrow
    axP.annotate('', xy=(-0.045 * _Wd, 0.03 * _H), xytext=(-0.045 * _Wd, 0.66 * _H), arrowprops=dict(arrowstyle='-|>', color=INK, lw=0.9 * KW, mutation_scale=8 * KW, shrinkA=0, shrinkB=0))
    axP.text(-0.075 * _Wd, 0.345 * _H, 'residual stream', rotation=90, ha='center', va='center', fontsize=FS_TEXT, color=INK)
else:
    print('fig1a_panel.png not found: panel (a) left empty (run fig1a_tikz.py, then this script again)')
axA = fig.add_axes(RECT([0.035, 0.075, 0.44, 0.88])); axB = fig.add_axes(RECT([0.6, 0.33, 0.36, 0.625]))        # (b) phase map, (c) sketch around P*
LIM = 1.0; UNT = '#7d8590'                      # mid colour of the untrained path (light blue -> grey -> dark blue)
def background(ax, alpha=0.85, scale=1.0, axes=True):
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_visible(False)
    if not axes: return
    akw = dict(arrowstyle='-|>', color=INK, lw=0.7, mutation_scale=7)
    ax.annotate('', xy=(1.0, 0), xytext=(0, 0), xycoords='axes fraction', arrowprops=akw, zorder=9)
    ax.annotate('', xy=(0, 1.0), xytext=(0, 0), xycoords='axes fraction', arrowprops=akw, zorder=9)
    ax.text(0.5, -0.025, r'$u$', transform=ax.transAxes, ha='center', va='top', fontsize=FS_AXIS)
    ax.text(-0.02, 0.5, r'$d_C$', transform=ax.transAxes, ha='right', va='center', rotation=90, fontsize=FS_AXIS)

def phase_blob(ax, c, k, z=0.3):
    """working phase with its measured shape (Fig. 3: P(working) > 0.5, phase_band_final.npz), outline smoothed,
    drawn about c (P* of the sketch) with log10 u and log10 d_C both divided by k"""
    from scipy.ndimage import gaussian_filter
    m = np.isfinite(PB_P); w = gaussian_filter(m.astype(float), 1.5)
    Ps = np.where(w > 0.5, gaussian_filter(np.where(m, PB_P, 0.0), 1.5) / np.maximum(w, 1e-9), 0.0)
    B = gaussian_filter((Ps > 0.5).astype(float), 7)                       # smooth outline of the measured region
    o = np.array([np.log10(1.08), np.log10(117)])
    X, Y = (PB_X - o[0]) / k + c[0], (PB_Y - o[1]) / k + c[1]
    ax.contourf(X, Y, B, levels=[0.5, 2], colors=[PB_TINT_W], zorder=z); ax.contour(X, Y, B, levels=[0.5], colors=[TRN], linewidths=0.8, zorder=z + 0.05)   # sharp boundary
# ---------------- phase map
ax = axA; background(ax)
cols = ['#1a1a1a', '#5d6168', '#9a9ea6']; CTR = np.array([[-1.25, -0.75], [1.25, -0.65], [0.0, 1.0]])
_r = np.deg2rad(28); _R = np.array([[np.cos(_r), -np.sin(_r)], [np.sin(_r), np.cos(_r)]]); CW = _R @ np.diag([0.30, 0.03]) @ _R.T
WORK_PTS = []
def glyph(cx, cy, kind, size=0.36, edge='0.55'):
    a = ax.inset_axes([cx - size / 2, cy - size / 2 * 0.8, size, size * 0.8], transform=ax.transData, zorder=7)
    a.set_xlim(-3, 3); a.set_ylim(-2.4, 2.4); a.axis('off')
    a.add_patch(plt.Rectangle((-3, -2.4), 6, 4.8, fc='#f7e3dd' if kind == 'work' else PB_TINT_D, ec=edge, lw=0.9))   # working phase: red box; the four edge limits (drifting phase): blue boxes
    if kind == 'dD':                                  # isotropic 3-D clouds, drawn in perspective (depth = dot size / shade)
        el, az = np.deg2rad(20), np.deg2rad(30)
        def proj(P):
            x, y, z = P.T; x1 = x * np.cos(az) - y * np.sin(az); y1 = x * np.sin(az) + y * np.cos(az)
            return np.c_[x1, z * np.cos(el) - y1 * np.sin(el)], y1 * np.cos(el) + z * np.sin(el)
        o = np.array([-2.6, -2.05]); L3 = 0.75                      # 3-D axis triad
        for e in np.eye(3):
            q, _ = proj(e[None] * L3); a.annotate('', xy=o + q[0], xytext=o, arrowprops=dict(arrowstyle='-|>', color='0.45', lw=0.5, mutation_scale=3.5, shrinkA=0, shrinkB=0))
        RB = 0.95; th = np.linspace(0, 2 * np.pi, 120)
        for i, c in enumerate(cols):
            ctr = CTR[i] * np.array([1.05, 0.95])
            a.add_patch(Circle(ctr, RB * 1.02, fc=mc.to_rgba(c, 0.06), ec=mc.to_rgba(c, 0.5), lw=0.4))     # sphere silhouette
            for ring in (np.c_[np.cos(th), np.sin(th), 0 * th], np.c_[np.cos(th), 0 * th, np.sin(th)], np.c_[0 * th, np.cos(th), np.sin(th)]):
                q, dep = proj(RB * ring); q += ctr; back = dep > 0
                a.plot(*np.where(back[:, None], np.nan, q).T, color=c, lw=0.55, alpha=0.85); a.plot(*np.where(back[:, None], q, np.nan).T, color=c, lw=0.35, alpha=0.3, ls=(0, (1.5, 1.5)))
            n = 40; v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1)[:, None]; P = v * RB * rng.uniform(0, 1, n)[:, None] ** (1 / 3)
            q, depth = proj(P); q = q + ctr; k = np.argsort(-depth); t = (depth[k] - depth.min()) / np.ptp(depth)
            a.scatter(q[k, 0], q[k, 1], s=0.8 + 2.4 * (1 - t), color=c, lw=0, alpha=0.5 + 0.5 * (1 - t))
            a.plot(*ctr, marker='+', color='k', ms=3.5, mew=0.7, zorder=5)
        return
    n = 45
    for i, c in enumerate(cols):
        ctr = CTR[i]
        if kind == 'work': p = rng.multivariate_normal(ctr, CW, n); WORK_PTS.append(dict(col=c, ctr=list(map(float, ctr)), pts=p.tolist()))
        if kind == 'u0': ctr = np.zeros(2); p = rng.multivariate_normal(ctr, CW, n)
        if kind == 'uinf': p = rng.multivariate_normal(ctr, CW * 0.05, n)
        if kind == 'd1': p = ctr + np.outer(rng.normal(0, 0.55, n), [np.cos(_r), np.sin(_r)])
        a.scatter(p[:, 0], p[:, 1], s=4.5 if kind == 'uinf' else 3.2, color=c, lw=0, alpha=0.9)
        a.plot(*ctr, marker='+', color='k', ms=3.5, mew=0.7, zorder=5)
G, GS = 0.7, 0.5                                     # glyph centre offset and size (larger: the map is filled)
glyph(-G, 0, 'u0', GS, UNT); glyph(G, 0, 'uinf', GS, UNT); glyph(0, -G, 'd1', GS, UNT); glyph(0, G, 'dD', GS, UNT)
lk = dict(fontsize=FS_TEXT, ha='center', va='center')
ax.text(-G, 0.255, r'$u\to0$', color=INK, **lk); 
ax.text(G, 0.255, r'$u\to\infty$', color=INK, **lk); 
ax.text(0, -G + 0.255, r'$d_C\to1$', color=INK, **lk)
ax.text(0, G + 0.255, r'$d_C\to D$', color=INK, **lk)

ZM = 0.29                                                         # half-width of the region shown in (c), in map units
ax.text(-0.93, -0.93, 'drifting phase', color=UNTD, fontsize=FS_TEXT, ha='left', va='bottom', zorder=9)   # the outer region: flows drift towards one of the four limits
glyph(0, 0.0, 'work', size=0.5, edge=EDGE); ax.text(0, ZM + 0.075, 'working phase', color=TRND, fontsize=FS_TEXT, ha='center', va='center', zorder=9)
# ---------------- (c): sketch around P*, with the same axes as the map
ax = axB; background(ax, alpha=0.85, scale=ZM, axes=False)
_ak = dict(arrowstyle='-|>', color=INK, lw=0.7, mutation_scale=7, shrinkA=0, shrinkB=0)
ax.annotate('', xy=(1.0, 0), xytext=(0, 0), xycoords='axes fraction', arrowprops=_ak, annotation_clip=False, zorder=9)
ax.annotate('', xy=(0, 1.0), xytext=(0, 0), xycoords='axes fraction', arrowprops=_ak, annotation_clip=False, zorder=9)
ax.set_xlabel(r'$u$', fontsize=FS_AXIS, labelpad=1); ax.set_ylabel(r'$d_C$', fontsize=FS_AXIS, labelpad=1)
ax.tick_params(length=2, width=0.5, labelsize=FS_TEXT, pad=1.5)
# stylised trajectories (one dot = one layer):
# trained -- enters from the left and contracts straight towards P* (no rotation), with noise; ends fluctuating near P*;
# untrained -- enters from the left and crosses the whole region with near-equal, persistent steps, leaving to the lower right.
def bez(P0, P1, P2, step):
    t = np.linspace(0, 1, 2000)[:, None]; B = (1 - t) ** 2 * P0 + 2 * t * (1 - t) * P1 + t ** 2 * P2
    s_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(B, axis=0), axis=1))]; return B[np.searchsorted(s_, np.arange(0, s_[-1], step))]
U = bez(np.array([-1.0, -0.15]), np.array([0.05, -0.35]), np.array([1.0, -0.88]), 0.11)
c0 = np.array([0.1, 0.25]); r_ = np.random.default_rng(2184); x_ = np.array([-1.0, 0.45]); T = [x_.copy()]
for k in range(15): x_ = c0 + 0.68 * (x_ - c0) + r_.normal(0, 0.06, 2); T.append(x_.copy())   # straight contraction + noise
T = np.array(T)
# fill the frame: scale both paths about one point (shapes and relative noise unchanged)
ZS = 1.85; ZC = (ZS * c0 - np.array([0.1, 0.12])) / (ZS - 1)          # P* lands near the middle
T = ZC + ZS * (T - ZC); U = ZC + ZS * (U - ZC); c0 = ZC + ZS * (c0 - ZC)
U = U + np.array([0.0, 0.32])                                          # untrained path a little higher, still clear of the trained one
T, U, c0 = T + [0, 0.22], U + [0, 0.60], c0 + [0, 0.22]              # centre the paths in the frame; the untrained path crosses the working phase
from matplotlib.colors import LinearSegmentedColormap
CMT = LinearSegmentedColormap.from_list('tr', [TRNL, TRN, TRND])      # trained: light -> deep brick red (trained phase colour)
CMU = LinearSegmentedColormap.from_list('un', [UNTL, UNT, UNTD])      # untrained: light blue -> grey -> deep blue
def chain(P, cm, lw, z):
    """measured-point style: thin line, one dot per layer (light -> dark from first to last layer), arrowheads every
    third layer, filled dot at the last layer if it lies inside the frame"""
    n = len(P) - 1
    cs = [cm(0.12 + 0.88 * i / max(n, 1)) for i in range(n + 1)]
    for i in range(n): ax.plot(*np.c_[P[i], P[i + 1]], color=cs[i + 1], lw=lw * 1.6, zorder=z, solid_capstyle='round', clip_on=True)
    ax.scatter(P[:, 0], P[:, 1], s=14, c=cs, lw=0.4, edgecolor='white', zorder=z + 0.5, clip_on=True)
    for i in range(1, n, 3):
        m = 0.55 * P[i] + 0.45 * P[i + 1]
        an = ax.annotate('', xy=m, xytext=P[i], arrowprops=dict(arrowstyle='-|>', color=cs[i + 1], lw=0, mutation_scale=11, shrinkA=0, shrinkB=0), zorder=z + 0.6, annotation_clip=False)
        an.arrow_patch.set_clip_path(ax.patch)
    if np.all(np.abs(P[-1]) < 1): ax.plot(*P[-1], 'o', ms=4.2, color=cm(1.0), mew=0, zorder=z + 1)
ax.set_xticks([]); ax.set_yticks([])                                   # schematic: no coordinate values

phase_blob(ax, c0, 1.05)                                               # working phase around P*
chain(U, CMU, 1.0, 5); chain(T, CMT, 0.95, 6)
ax.plot(*c0, marker='*', ms=10, color=PSTAR_FILL, mec=PSTAR_EDGE, mew=0.7, zorder=9)                 # P*, the point the trained path relaxes to
ax.text(c0[0] + 0.08, c0[1] - 0.09, r'$P^*$', color=INK, fontsize=FS_TEXT, ha='left', va='center', zorder=9)
# direct labels next to each path (no legend)
ax.text(-0.78, 0.68, 'trained', color=CMT(0.8), fontsize=FS_TEXT, ha='center', va='bottom', zorder=10)
_iu = int(np.argmax(U[:, 0] > -0.8)); ax.text(U[_iu, 0] - 0.1, U[_iu, 1] - 0.24, 'untrained', color=CMU(0.85), fontsize=FS_TEXT, ha='left', va='top', zorder=10)
axB.set_position(RECT([0.58, 0.075, 0.40, 0.88])); axB.set_anchor('S')   # 2026-11-14: training strip below (c) removed; (c) takes the full height, bottom aligned with the u axis of (b)
_r = RECT([0.0, 0.075, 0.44, 0.88]); fig.text(0.008, 0.995, '(a)', fontsize=FS_LETTER, ha='left', va='top'); fig.text(_r[0] - 0.035, 0.995, '(b)', fontsize=FS_LETTER, ha='left', va='top'); _rc = RECT([0.6, 0.33, 0.36, 0.625]); fig.text(_rc[0] - 0.045, 0.995, '(c)', fontsize=FS_LETTER, ha='left', va='top')
json.dump(WORK_PTS, open(os.path.join(FIG_OUT, 'fig1_work_points.json'), 'w'))   # token clouds of the working-phase glyph, reused by fig1a_tikz.py
[fig.savefig(os.path.join(FIG_OUT, 'fig1_schematic' + _e), dpi=300) for _e in ('.png', '.pdf')]; print('ok')
