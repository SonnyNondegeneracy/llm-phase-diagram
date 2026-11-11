#!/usr/bin/env python3
"""Build the v4 standard long-document corpus. Requires `pip install datasets transformers`.

Three public, versioned long-document sources (no concatenation, no cherry-picking):
  pg19       emozilla/pg19 (parquet mirror of DeepMind PG-19), test split, field `text`
  arxiv      ccdv/arxiv-summarization, config `document`, test split, field `article`
  govreport  ccdv/govreport-summarization, config `document`, test split, field `report`
Documents are scanned in stored order of the pinned revision. A document qualifies if, after truncation to
MAX_CHARS characters, it has >= MIN_TOKENS tokens under EVERY tokenizer in TOKENIZERS. The first COUNT qualifying
documents per source are kept. If the test split runs out, validation and then train are scanned (recorded per doc).

Outputs (in --out):
  corpus_v4.jsonl        {"id","source","repo","revision","split","index","text","tokens":{tokenizer: count}}
  corpus_v4_lock.json    revisions, scan counts, rule constants
  corpus_v4_manifest.json  ids, splits, indices, token counts, sha256(text)  (no texts; this one is returned)

usage: python build_corpus_v4.py --out corpus_v4
"""
import argparse, hashlib, json, os
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

MIN_TOKENS = 16384
COUNT = 32
MAX_CHARS = 400_000
SOURCES = {
    # name: (repo, config, field)
    'pg19': ('emozilla/pg19', None, 'text'),
    'arxiv': ('ccdv/arxiv-summarization', 'document', 'article'),
    'govreport': ('ccdv/govreport-summarization', 'document', 'report'),
}
SPLITS = ['test', 'validation', 'train']
TOKENIZERS = ['Qwen/Qwen2.5-7B', 'mistralai/Mistral-7B-v0.3', 'mistralai/Mistral-Nemo-Base-2407',
              'tiiuae/Falcon3-7B-Base', 'ibm-granite/granite-3.1-8b-base']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='corpus_v4')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    api = HfApi()
    toks = {t: AutoTokenizer.from_pretrained(t) for t in TOKENIZERS}
    lock = dict(rule=dict(min_tokens=MIN_TOKENS, count=COUNT, max_chars=MAX_CHARS, splits=SPLITS, tokenizers=TOKENIZERS), sources={})
    recs = []
    for name, (repo, cfg, field) in SOURCES.items():
        rev = api.dataset_info(repo).sha
        kept, scanned = [], {}
        for split in SPLITS:
            if len(kept) >= COUNT:
                break
            try:
                ds = load_dataset(repo, cfg, split=split, streaming=True, revision=rev)
            except Exception as e:  # split absent
                scanned[split] = f'unavailable: {e.__class__.__name__}'; continue
            n = 0
            for i, ex in enumerate(ds):
                n += 1
                text = (ex[field] or '')[:MAX_CHARS]
                if len(text) < 2 * MIN_TOKENS:  # cheap pre-filter: below 2 chars/token no tokenizer here reaches MIN_TOKENS
                    continue
                counts = {t: len(tk(text, add_special_tokens=False).input_ids) for t, tk in toks.items()}
                if min(counts.values()) >= MIN_TOKENS:
                    kept.append(dict(id=f'{name}-{len(kept):02d}', source=name, repo=repo, revision=rev, split=split,
                                     index=i, text=text, tokens=counts))
                    if len(kept) >= COUNT:
                        break
            scanned[split] = n
        lock['sources'][name] = dict(repo=repo, config=cfg, field=field, revision=rev, kept=len(kept), scanned=scanned)
        print(name, lock['sources'][name], flush=True)
        recs += kept
    with open(os.path.join(a.out, 'corpus_v4.jsonl'), 'w', encoding='utf-8') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    json.dump(lock, open(os.path.join(a.out, 'corpus_v4_lock.json'), 'w'), indent=1)
    man = [dict({k: r[k] for k in ('id', 'source', 'repo', 'revision', 'split', 'index', 'tokens')},
                chars=len(r['text']), sha256=hashlib.sha256(r['text'].encode('utf-8')).hexdigest()) for r in recs]
    json.dump(man, open(os.path.join(a.out, 'corpus_v4_manifest.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
