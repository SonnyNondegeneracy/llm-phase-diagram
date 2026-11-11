"""Locations of the input data and of the outputs (the only place where paths are set).

Defaults assume the layout
    code/analysis/   analysis scripts, outputs in code/analysis/out/
    code/figures/    figure scripts, PNG and PDF figures in code/figures/out/
    data/moments/      per-run moments (*.npz): u, d_C (and E for the five probed models), layers x 96 documents
    data/probe/        linear-probe results (probe_<model>.json)
Each location can be overridden with the environment variable named next to it."""
import os

CODE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.join(CODE, 'analysis')
DATA = os.path.abspath(os.environ.get('DATA_DIR', os.path.join(CODE, '..', 'data')))
RAW = os.path.abspath(os.environ.get('RAW_DIR', os.path.join(DATA, 'moments')))
PROBE = os.path.abspath(os.environ.get('PROBE_DIR', os.path.join(DATA, 'probe')))
OUT = os.path.abspath(os.environ.get('ANALYSIS_OUT', os.path.join(ANALYSIS, 'out')))
FIG_OUT = os.path.abspath(os.environ.get('FIGURE_OUT', os.path.join(CODE, 'figures', 'out')))
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG_OUT, exist_ok=True)


def raw(name):
    """path of a per-run moments file, e.g. raw('trained__qwen7__0.npz')"""
    return os.path.join(RAW, name)


def out(name):
    """path of an analysis output, e.g. out('fig4_J.json')"""
    return os.path.join(OUT, name)
