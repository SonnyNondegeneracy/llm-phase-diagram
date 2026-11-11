"""App. F figure (Table F-settings): Sec. IV results of the two groups under changed analysis settings, all with the
same estimator and uncertainty (analysis/sens_unified.py, sens_unified_<task>.json): point = estimate, bar = 16-84 % of
200 two-level bootstrap draws (networks, then documents), for (a) alpha and (b) -ln rho of both groups. One column per
setting, in the order of the table, grouped (group names above panel a). 'without X' drops architecture X from both
groups; 'documents' uses the 48 even / odd documents. All columns use the absolute fit range of the main text."""
from common import *
from matplotlib.lines import Line2D
def ld(t): return json.load(open(ana(f'sens_unified_{t}.json')))
M = ld('main')
ARCH = [('qwen1.5', 'Qwen2.5-1.5B'), ('qwen7', 'Qwen2.5-7B'), ('qwen14', 'Qwen2.5-14B'), ('mistral7', 'Mistral-7B-v0.3'),
        ('falcon3', 'Falcon3-7B'), ('granite', 'Granite-3.1-8B'), ('qwen35_4b', 'Qwen3.5-4B'), ('qwen35_9b', 'Qwen3.5-9B')]
GROUPS = [('main text', [('', M, 'explicit')]),
          ('lin. point', [('last third', ld('last_third'), 'explicit')]),
          ('cell width (dex)', [('0.07', ld('cell0.07'), 'explicit'), ('0.14', ld('cell0.14'), 'explicit'), ('0.2', ld('cell0.2'), 'explicit')]),
          ('fit range (dex)', [('shrunk 0.1', ld('win-0.1'), 'explicit'), ('grown 0.1', ld('win0.1'), 'explicit')]),
          ('force at', [('midpoint', M, 'midpoint'), ('end point', M, 'implicit')]),
          ('without', [(nm, ld('loo_' + a), 'explicit') for a, nm in ARCH]),
          ('documents', [('even half', ld('half0'), 'explicit'), ('odd half', ld('half1'), 'explicit')])]
GAP = 0.8
cols, labs, spans, x = [], [], [], 0.0
for gname, items in GROUPS:
    x0 = x
    for lab, D, sch in items: cols.append((x, D, sch)); labs.append(lab); x += 1
    spans.append((gname, x0, x - 1)); x += GAP
fig, axs = plt.subplots(2, 1, figsize=(W, 3.7), sharex=True, gridspec_kw=dict(hspace=0.22, left=0.075, right=0.995, top=0.84, bottom=0.18))
for ax, key, lab in [(axs[0], 'a', r'inertia coefficient $\alpha$'), (axs[1], 'R', r'contraction rate $-\ln\rho$')]:
    ax.axhline(0, color='0.3', lw=0.6, ls=(0, (3, 2)), zorder=0.5)
    for _, a0, a1 in spans[1:]: ax.axvline(a0 - (1 + GAP) / 2, color='0.86', lw=0.5, zorder=0)
    for g, col, dx in [('untrained', UNT, -0.17), ('trained', TRN, 0.17)]:
        ax.axhline(M[g]['gamma'] if key == 'a' else M[g]['R']['explicit'], color=col, lw=0.6, alpha=0.35, zorder=0.4)
        for xx, D, sch in cols:
            if g not in D: continue
            r = D[g]; v, (lo, hi) = (r['gamma'], r['gamma_q']) if key == 'a' else (r['R'][sch], r['R_q'][sch])
            ax.plot([xx + dx] * 2, [lo, hi], color=col, lw=1.1, solid_capstyle='butt', zorder=2)
            ax.plot(xx + dx, v, 'o', ms=3.6, color=col, mfc=col, mec=col, mew=0.9, zorder=3)
    ax.set_ylabel(lab, fontsize=FS_AXIS); ax.tick_params(labelsize=FS_KEY)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
axs[0].set_ylim(-0.06, 0.43); axs[1].set_ylim(-0.042, 0.042)
axs[1].set_xticks([c[0] for c in cols]); axs[1].set_xticklabels(labs, rotation=40, ha='right', rotation_mode='anchor', fontsize=FS_KEY)
axs[1].set_xlim(-0.6, x - GAP + 0.1)
for gname, a0, a1 in spans:
    axs[0].text((a0 + a1) / 2, 1.04, gname, transform=axs[0].get_xaxis_transform(), ha='center', va='bottom', fontsize=FS_KEY, color=MUTED)
HL = [Line2D([], [], marker='o', ls='', color=UNT, ms=3.6, label='untrained'), Line2D([], [], marker='o', ls='', color=TRN, ms=3.6, label='trained'),
      Line2D([], [], color='0.45', lw=1.1, label='16%–84% bootstrap'), Line2D([], [], color='0.45', lw=0.6, alpha=0.45, label='main-text value')]
fig.legend(handles=HL, loc='upper center', bbox_to_anchor=(0.535, 1.0), ncol=5, frameon=False, fontsize=FS_KEY, handletextpad=0.3, columnspacing=1.4)
letter(fig, axs[0], 'a', dy=0.035); letter(fig, axs[1], 'b')
[fig.savefig(fig_path('figF_settings' + _e), dpi=300) for _e in ('.png', '.pdf')]; print('ok')
