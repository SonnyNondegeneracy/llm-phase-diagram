# A phase diagram for training and inference in language models: code

This repository holds the code that turns the measured moments into every number and figure of the paper. The moments themselves (one file per network, with `u` and `d_C` for every layer and document) are in the data repository; see `paths.py` for where the code expects them. The forward-pass measurement scripts are in `measurement/`.

## Layout

```
LICENSE             MIT license
paths.py            locations of the data and of the outputs (overridable by environment variables)
run_all.sh          the whole pipeline in dependency order
measurement/        forward-pass scripts that produced the moments, corpus definition, run index (see measurement/README.md)
analysis/           analysis scripts; outputs in analysis/out/
figures/            figure scripts; each figure as PNG and vector PDF in figures/out/
```

Requirements: Python 3.11 or later, numpy, scipy, matplotlib, pillow; `figures/fig1a_tikz.py` also needs pymupdf and a `tectonic` binary (environment variable `TECTONIC`). Every analysis script needs less than 0.3 GB of memory. The whole pipeline takes about 30 minutes on a laptop CPU.

```
DATA_DIR=/path/to/data sh run_all.sh
```

## Data naming

`data/moments/<group>__<architecture>__<seed>.npz`. The arrays have shape (layers, 96 documents). The groups are:

| prefix | networks |
|---|---|
| `trained` | the 8 trained models |
| `pythia` | Pythia-1.4B checkpoints (`pythia__step<N>__0`) |
| `native` | untrained, F family, σ = 0.02 (the architecture's default initialization) |
| `init` | untrained, G family, σ = 0.02 |
| `unit` | untrained, G family, σ = 1 |
| `sweep` | untrained, initialization-scale sweep (`sweep__<arch>__<F|G><σ>__<seed>`, `p` = decimal point) |

## Where each result comes from

| Paper | Script → output |
|---|---|
| P* = (1.08, 117); fit range | `analysis/pythia_pstar.py` (`P0`, `OFF`) |
| Fig. 2 | `figures/fig2_coordinates.py` (moments) |
| Fig. 3: working phase, coexistence band | `analysis/phase_band_final.py` → `phase_band_final.npz`; `figures/fig3_phase_diagram.py` |
| Entry depth 0.11; 252288 / 12864 deep-layer points | `phase_band_final.json` (`first_entry_median`, `n_points`, `n_points_trained`) |
| Spread of late-half positions (0.085 dex; 0.28–1.38 dex) | `analysis/dispersion.py` → `dispersion.json` |
| Fig. 4: J, its type, force arrows, Pe | `analysis/fig4_J.py` → `fig4_J.json`; `analysis/eta_stats.py` → `eta_stats.json`; `figures/fig4_dynamics.py` |
| α and −ln ρ of the two groups (Sec. IV) | `analysis/sens_unified.py main` → `sens_unified_main.json` |
| Fig. 5(a) fraction in the working phase | `analysis/pythia_share.py` → `pythia_share.json` |
| Fig. 5(b) α per checkpoint | `analysis/pythia_ck_fixedP.py` → `pythia_ck_fixedP.json` |
| Fig. 5(c) and Fig. F-bins, binned −ln ρ | `analysis/pythia_bins.py` → `pythia_bins.json` |
| Step-0 direction cosine (App. F) | `analysis/step_direction_cos.py` → `step_direction_cos.json` |
| Fig. A | `figures/figA_pythia.py` (moments) |
| Fates of the untrained runs (App. B, Fig. B-fates) | `analysis/fates_share.py` → `fates_share.json`; `figures/figB_fates.py` |
| Slowdown with depth (App. B, Fig. B-slow) | `analysis/depth_slowdown.py` → `depth_slowdown.json`; `figures/figB_slow.py` |
| Linear readout (App. C, Fig. C) | `figures/figC_readout.py` (moments, probe results) |
| Per-model −ln ρ (App. D) | `analysis/rho_models_fixedwin.py` → `rho_models_fixedwin.json` |
| Fig. F-settings, Table IV | `analysis/sens_unified.py <task>` → `sens_unified_<task>.json`; `figures/figF_settings.py` |
| Choice of k (Fig. F-k) | `analysis/phase_k_choice.py` → `phase_k_choice.json`; `figures/figF_k.py` |
| Fig. 1 (schematic) | `figures/fig1_schematic.py`, `figures/fig1a_tikz.py` |

## Citation

Paper: [arXiv:2611.19848](https://arxiv.org/abs/2611.19848)

```bibtex
@misc{song2026phasediagramtraininginference,
      title={A phase diagram for training and inference in language models}, 
      author={Zhuo-Yang Song and Zeyu Li and Qing-Hong Cao and Ming-xing Luo and Hua Xing Zhu},
      year={2026},
      eprint={2611.19848},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2611.19848}, 
}
```

## License

The code is released under the MIT License (see `LICENSE`). The data in the data repository are released under CC BY 4.0.
