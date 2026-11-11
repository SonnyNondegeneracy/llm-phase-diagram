# Measurement

The forward-pass scripts that produced the measurements of the paper, the corpus definition, and an index of all 188 runs.

The scripts are the original files used for the runs, kept byte-identical so that their SHA-256 hashes match the hashes recorded with each run. They were written over the course of the project: their comments refer to earlier versions, and they compute more quantities than the paper uses. The paper uses only the quantities listed under "What the paper uses".

## Files

| File | Role | SHA-256 |
| --- | --- | --- |
| `dim_v4.py` | measurement script (18 runs) | `073eeafd353005bbf1e1ab311838624784882bf64c6afbb818421e940382ea12` |
| `dim_v5.py` | measurement script (29 runs); same measurement, adds the architecture's own initialization (`--faithful`) | `487838e49f74890fff8428201c950574990caa185b3b3d23a261acdc5694b0db` |
| `dim_v6.py` | measurement script (22 runs); same measurement | `a8536ddde141b6ec39905a7ea669d086b5969008541a704e1bc451dd6a443b50` |
| `dim_v7_hfinit_fixed.py` | measurement script (114 runs, including the whole initialization-scale sweep); same measurement; also imported by the probe | `f387806df80b36520aecb971ef27d3b635d0fd40def584fbab2e44a84723b7d9` |
| `dim_probe.py` | helper module imported by `dim_probe2.py` | `5e8795b494b63c8650609fcda5c5ae3fc5055a28d1d8e4a8fd47ff991634a617` |
| `dim_probe2.py` | linear-readout probe (Appendix C) | `eac7aa210576e13da62ac0c1baf984be4f1f991ba1c6e4e88c7c6f1e62a73c86` |
| `build_corpus_v4.py` | builds the 96-document corpus | `9044a7c99d3c01e5e3815cc578d4fb19640a4a32263adeacfe98c728d9307a20` |
| `corpus_v4_lock.json` | dataset revisions and selection rule of the corpus | `f78508c18fd2f2bccd599801335c1e99eba7abbf556229860cb4b2c034d5fc15` |
| `corpus_v4_manifest.json` | the 96 documents: source, revision, split, index, token counts, SHA-256 of the text | `b26c4ba034cd1fd180eb2d9c97697311c4e304e66e1dfdcff4ecb156be62d0d2` |
| `runs.csv` | one row per run (see below) | |
| `environment.md` | software environments | |
| `PROVENANCE.md` | how the runs are bound to the scripts | |

The kept-position rule and the measured quantities are identical in the four measurement scripts: in every document the first 16 positions and the last one are dropped, then positions whose residual-stream norm exceeds 8 times the document's median norm (`SKIP_FIRST = 16`, `HI_NORM = 8.0`); all moments are computed in float64 on the kept positions.

## What the paper uses

From the output of a measurement run, the paper uses, for each block output `k = 1, ..., L-1` and each document, the fields `trC`, `trC2` and, for the reference made of all other documents (`ref.all`), `delta2` and `E`. The files in `moments/` of the data repository hold

- `u = delta2 / sqrt(trC2)`,
- `dC = trC**2 / trC2`,
- `E` (five trained models only, for Appendix C),

as arrays of shape (L-1, 96). From the probe runs the paper uses the held-out accuracy of the selected logistic probe per block (`logistic.selected.acc_test`), given in `probe/` of the data repository.

## `runs.csv`

One row per run: 8 trained models, 19 Pythia-1.4B checkpoints, 72 first-batch untrained runs, 84 runs of the initialization-scale sweep and 5 probe runs.

| Column | Content |
| --- | --- |
| `group` | `trained`, `pythia`, `first_batch`, `sweep` or `probe` |
| `data_file` | the corresponding file in the data repository |
| `model_id`, `resolved_revision` | Hugging Face model and commit |
| `initialization_family`, `sigma`, `seed` | untrained runs: family F (the architecture's own initialization) or G (Gaussian), scale, seed |
| `tokens` | tokens per document |
| `script_file`, `script_sha256` | script that produced the run and its hash |
| `command` | command line where it was recorded, otherwise `unrecorded`; local paths are written as `<MODEL_DIR>`, `<CORPUS>`, `<OUT>` |
| `python`, `torch`, `transformers`, `cuda` | runtime versions where recorded, otherwise `unrecorded` |
| `raw_result_sha256` | SHA-256 of the run's raw output file (not distributed; the data repository holds the reduced moments) |

## Reproduction

Build the corpus with `build_corpus_v4.py`, then check every document against `corpus_v4_manifest.json` (the construction entry point resolves the live dataset revision and does not pin its tokenizer revisions, so the lock and the manifest are the reference). Corpus text and model weights are not distributed. Use the revision in `runs.csv` for every model. Examples:

```sh
# trained model
python dim_v4.py --model Qwen/Qwen2.5-1.5B --revision 8faed761d45a263340a0528343f099c05c9a4323 --tokens 16384 --dtype bfloat16 --corpus-jsonl <CORPUS> --out <OUT>
# Pythia-1.4B checkpoint
python dim_v4.py --model EleutherAI/pythia-1.4b --revision c42605265af8e9c36dd9a93bd4be04b73111d886 --tokens 2048 --dtype bfloat16 --corpus-jsonl <CORPUS> --out <OUT>
# untrained, family G, sigma = 0.02
python dim_v4.py --model Qwen/Qwen2.5-7B --revision d149729398750b98c0af14eb82c78cfe92750796 --tokens 16384 --dtype bfloat16 --corpus-jsonl <CORPUS> --random-blocks --seed 41 --random-std 0.02 --out <OUT>
# untrained, family F, sigma = 0.005 (scale sweep)
python dim_v7_hfinit_fixed.py --model Qwen/Qwen2.5-1.5B --revision 8faed761d45a263340a0528343f099c05c9a4323 --corpus-jsonl <CORPUS> --tokens 16384 --random-blocks --seed 41 --random-std 0.005 --faithful --out <OUT>
# linear-readout probe
python dim_probe2.py --model Qwen/Qwen2.5-7B --revision d149729398750b98c0af14eb82c78cfe92750796 --corpus-jsonl <CORPUS> --tokens 16384 --out <OUT>
```

Some comments in the scripts mention files that are not part of this release (usage notes and earlier scripts); they are not needed to run the scripts.
