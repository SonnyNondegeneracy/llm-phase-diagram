# Environments

All runs were made with one of two software environments, which differ only in the `transformers` and `huggingface-hub` versions.

| Environment | Used for | Python | torch | CUDA | transformers | huggingface-hub |
| --- | --- | --- | --- | --- | --- | --- |
| A | all models except Qwen3.5 | 3.12.3 | 2.6.0+cu124 | 12.4 | 4.56.1 | 0.34.4 |
| B | Qwen3.5-4B and Qwen3.5-9B | 3.12.3 | 2.6.0+cu124 | 12.4 | 5.2.0 | 1.33.0 |

The probe runs additionally recorded `protobuf` 5.29.5 and `sentencepiece` 0.2.1. `build_corpus_v4.py` needs `datasets` and `transformers`.

The versions recorded for each run are in `runs.csv`; fields that were not recorded for a run are marked `unrecorded`.
