#!/bin/sh
# Runs the whole pipeline in dependency order: analysis outputs into analysis/out/, figures into figures/out/.
# Paths are set in paths.py (overridable with DATA_DIR, RAW_DIR, PROBE_DIR, ANALYSIS_OUT, FIGURE_OUT).
set -e
PY=${PYTHON:-python}
cd "$(dirname "$0")/analysis"
$PY phase_band_final.py
$PY phase_k_choice.py
for t in main last_third cell0.07 cell0.14 cell0.2 win-0.1 win0.1 \
         loo_qwen1.5 loo_qwen7 loo_qwen14 loo_mistral7 loo_falcon3 loo_granite loo_qwen35_4b loo_qwen35_9b half0 half1; do
    $PY sens_unified.py $t
done
$PY fig4_J.py
$PY eta_stats.py
$PY rho_models_fixedwin.py
$PY fates_share.py
$PY dispersion.py
$PY depth_slowdown.py
$PY step_direction_cos.py
$PY pythia_share.py
$PY pythia_ck_fixedP.py
$PY pythia_bins.py
cd ../figures
$PY fig1_schematic.py          # first pass writes fig1_work_points.json (panel (a) is left empty if fig1a_panel.png is missing)
$PY fig1a_tikz.py              # needs tectonic; renders panel (a) to fig1a_panel.png
$PY fig1_schematic.py          # second pass with panel (a)
for f in fig2_coordinates fig3_phase_diagram fig4_dynamics fig5_training figA_pythia figB_fates figB_slow figC_readout figF_bins figF_k figF_settings; do
    $PY $f.py
done
