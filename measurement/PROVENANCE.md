# Provenance

Each of the 188 runs in `runs.csv` is bound to the script that produced it: the script's SHA-256 was recorded with the run, and it equals the hash of the file in this directory. The binding is: `dim_v4.py` 18 runs, `dim_v5.py` 29, `dim_v6.py` 22, `dim_v7_hfinit_fixed.py` 114, `dim_probe2.py` 5.

The raw output of every run is identified by `raw_result_sha256`. The files in `moments/` of the data repository were reduced from these raw outputs (see `README.md`, "What the paper uses"); each row of `runs.csv` names its file in `data_file`, and the 183 measurement rows cover the 183 files in `moments/` one to one.

Recorded metadata: the full command line for 97 runs; Python for 145, torch for 158, transformers for 158 and CUDA for 153 runs. Where a field was not recorded it is marked `unrecorded`; the environments in `environment.md` were used throughout.

The corpus is fixed by `corpus_v4_lock.json` (dataset revisions and selection rule) and `corpus_v4_manifest.json` (the 96 documents with the SHA-256 of each text).
