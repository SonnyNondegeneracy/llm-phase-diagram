"""Fig. 1(a): a stack of layer sheets and the representation space of one block, drawn with TikZ.

One 3-D scene with one orthographic camera and one light (upper left): a stack of layer sheets (depth upward), the
tokens of one document entering it, sheet l pulled out, and a frustum from sheet l to a glass plate = the
representation space of block l; on the plate, the token clouds of three documents (+ = document mean), the same
points as the working-phase glyph of Fig. 1(b) (fig1_work_points.json, written by fig1_schematic.py).
Writes fig1a_panel.tex, compiles it with Tectonic (XeLaTeX) to fig1a_panel.pdf (drawn on a grid of 510/183 pt units)
and renders fig1a_panel.png at 600 dpi, which fig1_schematic.py places as panel (a).

Environment: TECTONIC (tectonic binary, default 'tectonic'), STIX_DIR (directory holding STIXGeneral*.ttf; default
matplotlib's mpl-data/fonts/ttf)."""
import json
import math
import os
import subprocess
import sys
import numpy as np
import matplotlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from paths import FIG_OUT as OUT

STX = os.environ.get('STIX_DIR', os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf'))
TECTONIC = os.environ.get('TECTONIC', 'tectonic')
f = lambda v: f'{v:.3f}'
L = []                     # TikZ body lines
add = L.append

pt = lambda p: f'({f(p[0])},{f(p[1])})'
poly = lambda ps: ' -- '.join(pt(p) for p in ps) + ' -- cycle'
AZ, EL, SC = math.radians(-12), math.radians(24), 24.5          # camera azimuth, elevation; scale (units per scene unit)
ORG = np.array([45.0, 9.5])
W_ = np.array([math.sin(AZ) * math.cos(EL), math.cos(AZ) * math.cos(EL), -math.sin(EL)])     # viewing direction (into page)
R_ = np.array([math.cos(AZ), -math.sin(AZ), 0.0])                                            # page-right direction
def P(p):
    x, y, z = p
    return ORG + SC * np.array([x * math.cos(AZ) - y * math.sin(AZ), (x * math.sin(AZ) + y * math.cos(AZ)) * math.sin(EL) + z * math.cos(EL)])
def box_faces(lo, hi):
    """Visible faces of an axis-aligned box as (normal, 4 corners)."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    F = [((0, 0, 1), [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]), ((0, 0, -1), [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)]),
         ((1, 0, 0), [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)]), ((-1, 0, 0), [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)]),
         ((0, 1, 0), [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)]), ((0, -1, 0), [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)])]
    return [(np.array(n, float), [np.array(c, float) for c in cs]) for n, cs in F if np.dot(n, W_) < 0]
LIGHT = np.array([-0.55, 0.25, 0.8]); LIGHT /= np.linalg.norm(LIGHT)
def shade(n, base, dark):
    """Lambert-like tone between 'dark' (facing away) and 'base' (facing the light) -> xcolor mix."""
    k = max(0.0, float(np.dot(n, LIGHT))); return f'{base}!{int(100 * (0.35 + 0.65 * k))}!{dark}'
# ---------------------------------------------------------------- stack of sheets
NB, HL, SS, TH, GAP, Z0 = 9, 5, 0.62, 0.028, 0.105, 1.30
SCX, SCY = (np.array([0.5, 0.5, 0.0]) - 1.55 * np.array([math.cos(AZ), -math.sin(AZ), 0.0]))[:2]   # stack to the left of the plate on the page
# document j's token sequence: tiles along the sheet's front edge (the sequence direction), entering the stack
NT = 7; xs_t = [SCX - SS / 2 + 0.06 + k * (SS - 0.17) / (NT - 1) for k in range(NT)]; yt = SCY - SS / 2 + 0.05; zt = Z0 - 0.26
col_seg = lambda z0_, z1_: [add(rf'\draw[docj!55, line width=0.35pt] {pt(P((x_ + 0.03, yt + 0.03, z0_)))} -- {pt(P((x_ + 0.03, yt + 0.03, z1_)))};') for x_ in xs_t]
col_seg(zt, Z0)                                                       # tokens enter the stack (one line per token)
# sheet l's box (pulled out along the page-right direction), needed before the stack so the frustum can be drawn behind it
LAY = ['9AA1AB', '5D6168', 'E4E7EB']   # selected block: base, dark and light grey
PULL = 0.13                                 # sheet l pulled slightly out of the stack
_z = Z0 + HL * GAP; _off = PULL * R_
lo = np.array([SCX - SS / 2, SCY - SS / 2, _z]) + _off; hi = np.array([SCX + SS / 2, SCY + SS / 2, _z + TH]) + _off
# ---------------------------------------------------------------- frustum from sheet l to the top of the glass plate
def hull(pts):
    pts = sorted(map(tuple, pts)); cr = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo_, up_ = [], []
    for p in pts:
        while len(lo_) >= 2 and cr(lo_[-2], lo_[-1], p) <= 0: lo_.pop()
        lo_.append(p)
    for p in reversed(pts):
        while len(up_) >= 2 and cr(up_[-2], up_[-1], p) <= 0: up_.pop()
        up_.append(p)
    return lo_[:-1] + up_[:-1]
# opens from two diagonal corners of sheet l's top face; drawn after the stack so no sheet hides it
sc = [np.array([lo[0], lo[1], hi[2]]), np.array([hi[0], hi[1], hi[2]])]   # two diagonal corners (front-left, back-right)
PZ0, PZ1, PA, PB = 0.66, 0.72, -0.12, 1.12
PSH = np.array([-13.0, 4.0])                # shift of the glass plate toward the stack (page units)
PP = lambda p: P(p) + PSH                                                     # the glass plate (representation space of block l)
cc = [np.array(p, float) for p in ((PA, PA, PZ1), (PB, PA, PZ1), (PB, PB, PZ1), (PA, PB, PZ1))]
S2 = [tuple(P(p)) for p in sc]; C2 = [tuple(PP(p)) for p in cc]
H = hull(S2 + C2)
FRUSTUM = [rf'\fill[layer, fill opacity=0.05] {poly(H)};']
for k in range(len(H)):
    a_, b_ = H[k], H[(k + 1) % len(H)]
    if (a_ in S2) != (b_ in S2):
        FRUSTUM.append(rf'\draw[layer, line width=0.35pt] {pt(a_)} -- {pt(b_)};')
for i in range(NB):
    hl = i == HL; z = Z0 + i * GAP
    off = PULL * R_ if hl else np.zeros(3)
    lo = np.array([SCX - SS / 2, SCY - SS / 2, z]) + off; hi = np.array([SCX + SS / 2, SCY + SS / 2, z + TH]) + off
    for n, cs in box_faces(lo, hi):
        col = shade(n, 'layerlight', 'layerdark') if hl else shade(n, 'white', 'slabdark')
        add(rf'\filldraw[fill={col}, draw={"layerdark" if hl else "grey"}, line width={"0.4pt" if hl else "0.3pt"}, line join=round] {poly([P(c) for c in cs])};')
    if hl:
        for x_ in xs_t:                                               # the residual stream at block l: one cell per token
            c0 = np.array([x_, yt, hi[2]]) + off
            cs = [c0 + d for d in (np.zeros(3), np.array([0.06, 0, 0]), np.array([0.06, 0.06, 0]), np.array([0, 0.06, 0]))]
            add(rf'\fill[docjdark] {poly([P(q) for q in cs])};')
for x_ in xs_t:
    cs = [np.array([x_, yt, zt]) + d for d in (np.zeros(3), np.array([0.06, 0, 0]), np.array([0.06, 0.06, 0]), np.array([0, 0.06, 0]))]
    add(rf'\filldraw[fill=docjlight, draw=docjdark, line width=0.3pt] {poly([P(q) for q in cs])};')
for c_ in FRUSTUM: add(c_)
# ---------------------------------------------------------------- the glass plate with the token clouds of three documents
fl = [PP(p) for p in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))]; cen = sum(fl) / 4
add(rf'\fill[black, fill opacity=0.05, path fading=soft] ({f(cen[0] + 2.0)},{f(cen[1] + 5.0)}) ellipse (20 and 4.5);')   # soft shadow
for n, cs in box_faces(np.array([PA, PA, PZ0]), np.array([PB, PB, PZ1])):
    if n[2] > 0: continue
    add(rf'\filldraw[fill={shade(n, "white", "slabdark")}, draw=grey, line width=0.35pt, line join=round] {poly([PP(c) for c in cs])};')
top = [PP(p) for p in ((PA, PA, PZ1), (PB, PA, PZ1), (PB, PB, PZ1), (PA, PB, PZ1))]
add(rf'\shade[shading angle=125, left color=white, right color=floorlo, fill opacity=0.92] {poly(top)};')
add(rf'\draw[grey, line width=0.45pt, line join=round] {poly(top)};')
add(rf'\draw[white, line width=0.9pt, opacity=0.9] {pt(PP((0.05, 0.08, PZ1)))} -- {pt(PP((0.05, 0.85, PZ1)))};')       # glass highlight
WP = json.load(open(os.path.join(OUT, 'fig1_work_points.json')))       # the same three clouds as the working-phase glyph of Fig. 1(b)
GX = lambda x: PA + (PB - PA) * (x + 3.0) / 6.0; GY = lambda y: PA + (PB - PA) * (y + 2.4) / 4.8    # inset coords -> plate
for k_, d_ in enumerate(WP):
    col = ('doca', 'docb', 'docc')[k_]
    for x_, y_ in d_['pts']:
        add(rf'\fill[{col}, fill opacity=0.85] {pt(PP((GX(x_), GY(y_), PZ1)))} circle (0.30);')
    m = PP((GX(d_['ctr'][0]), GY(d_['ctr'][1]), PZ1)); add(rf'\draw[ink, line width=0.7pt] ({f(m[0] - 1.1)},{f(m[1])}) -- ({f(m[0] + 1.1)},{f(m[1])}) ({f(m[0])},{f(m[1] - 1.1)}) -- ({f(m[0])},{f(m[1] + 1.1)});')

TEX = r'''\documentclass[tikz,border=0pt]{standalone}
\usepackage{fontspec}
\setmainfont{STIXGeneral.ttf}[Path=%(stix)s/, BoldFont=STIXGeneralBol.ttf, ItalicFont=STIXGeneralItalic.ttf]
\usetikzlibrary{arrows.meta,fadings}
\tikzfading[name=soft, inner color=transparent!0, outer color=transparent!100]
\definecolor{ink}{HTML}{1A1A1A}\definecolor{grey}{HTML}{8F949C}
\definecolor{layer}{HTML}{%(lay)s}\definecolor{layerdark}{HTML}{%(layd)s}\definecolor{layerlight}{HTML}{%(layl)s}\definecolor{docj}{HTML}{9AA1AB}\definecolor{docjdark}{HTML}{4F545B}\definecolor{docjlight}{HTML}{EEF0F2}\definecolor{doca}{HTML}{1A1A1A}\definecolor{docb}{HTML}{6B7078}\definecolor{docc}{HTML}{A3A7AE}
\definecolor{slabdark}{HTML}{AEB3BB}\definecolor{floorlo}{HTML}{E2E6ED}

\begin{document}
\fontsize{7.5}{8.5}\selectfont
\begin{tikzpicture}[x=%(unit)spt, y=%(unit)spt]
%(body)s
\end{tikzpicture}
\end{document}
''' % dict(stix=STX, body='\n'.join(L), lay=LAY[0], layd=LAY[1], layl=LAY[2], unit=f'{510 / 183:.5f}')

tex = os.path.join(OUT, 'fig1a_panel.tex')
open(tex, 'w').write(TEX)
subprocess.run([TECTONIC, '-X', 'compile', '--outdir', OUT, tex], check=True, capture_output=True)
import pymupdf
pg = pymupdf.open(os.path.join(OUT, 'fig1a_panel.pdf'))[0]
pix = pg.get_pixmap(dpi=600, alpha=True)                              # the standalone class crops the page to the drawing
pix.save(os.path.join(OUT, 'fig1a_panel.png'))
print('wrote fig1a_panel.png', pix.width, pix.height)
