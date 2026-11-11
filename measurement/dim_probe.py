#!/usr/bin/env python3
"""Layer-wise linear decodability of document identity (probe, v8).

Question: in the plateau phase E*PR stays roughly constant. One reading is E*PR = |delta|^2 / lambda_w, i.e. the
document's shared offset measured in units of a typical direction's variance -- a squared signal-to-noise ratio of
document identity. If that reading is right, how well single token states can be assigned to their document should
rise during the lift phase and then stay roughly constant in the plateau (and degrade for shuffled input).

Point set (same construction as dim_pooled_nn.py): every document is forwarded once at full length (--tokens); at each
state the residual vectors at kept positions p >= MIN_POS on the STRIDE grid are taken (kept = dim_v7 rule: first 16 and
last position dropped, |h| > 8 x median dropped). Positions p < tokens/2 form the TRAIN half, p >= tokens/2 the TEST
half of every document, so every test token comes from a part of the document never used for fitting.

Per state, on the test half (all fitting on the train half only):
  dprime_*   for each document j: u_j = unit vector of (train mean of j - train mean of the other documents);
             d'_j = (mean of u_j.x over test tokens of j - mean over test tokens of others) / pooled s.d.;
             median and quartiles over documents.
  acc_centroid  nearest train-centroid classification accuracy (Euclidean, 1 of N documents).
  acc_logistic  multinomial logistic regression on the top-K principal components of the train half
             (components standardised, L2 = --l2, full-batch L-BFGS, fixed iterations); accuracy and the median margin
             (true-class logit minus the largest other logit) on the test half.
  chance = 1 / number of documents.
Optional --control shuffle uses dim_v7's input control (content tokens permuted within each document, seed 0 + index).
Schema dim_probe/1. One forward per document. Model loading and kept-position rule come from dim_v7_hfinit_fixed.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import torch

import dim_v7_hfinit_fixed as M

SCHEMA = 'dim_probe/1'


def select_positions(h, stride, min_pos):
    _, x, _, pos = M.kept(h)
    sel = (pos >= min_pos) & (pos % stride == 0)
    return x[sel], pos[sel]


@torch.no_grad()
def dprime(Xtr, ytr, Xte, yte, n_docs):
    """Per-document d' along the document's own mean-offset direction (fit on train, evaluated on test)."""
    mu = Xtr.mean(0)
    Xtr = Xtr - mu; Xte = Xte - mu
    cnt = torch.bincount(ytr, minlength=n_docs).to(Xtr.dtype)
    S = torch.zeros(n_docs, Xtr.shape[1], dtype=Xtr.dtype, device=Xtr.device).index_add_(0, ytr, Xtr)
    m = S / cnt[:, None]
    others = (S.sum(0)[None, :] - S) / (cnt.sum() - cnt)[:, None]
    U = m - others
    U = U / U.norm(dim=1, keepdim=True).clamp_min(1e-30)
    P = Xte @ U.T                                   # [n_test, n_docs]
    out = []
    for j in range(n_docs):
        a = P[yte == j, j]; b = P[yte != j, j]
        if a.numel() < 2:
            continue
        sd = torch.sqrt(0.5 * (a.var() + b.var())).clamp_min(1e-30)
        out.append(float((a.mean() - b.mean()) / sd))
    return out, m


@torch.no_grad()
def centroid_accuracy(m_train, mu, Xte, yte):
    Xc = Xte - mu
    d2 = (Xc * Xc).sum(1, keepdim=True) - 2 * Xc @ m_train.T + (m_train * m_train).sum(1)[None, :]
    return float((d2.argmin(1) == yte).double().mean())


def logistic(Xtr, ytr, Xte, yte, n_docs, k, l2, iters, seed):
    torch.manual_seed(seed)
    mu = Xtr.mean(0)
    A = Xtr - mu
    k = int(min(k, A.shape[0] - 1, A.shape[1]))
    _, _, V = torch.linalg.svd(A, full_matrices=False)
    V = V[:k].T
    Ztr = A @ V; Zte = (Xte - mu) @ V
    sd = Ztr.std(0).clamp_min(1e-12)
    Ztr = Ztr / sd; Zte = Zte / sd
    W = torch.zeros(k, n_docs, dtype=Ztr.dtype, device=Ztr.device, requires_grad=True)
    b = torch.zeros(n_docs, dtype=Ztr.dtype, device=Ztr.device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, history_size=20, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Ztr @ W + b, ytr) + l2 * (W * W).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        logit = Zte @ W + b
        acc = float((logit.argmax(1) == yte).double().mean())
        true = logit.gather(1, yte[:, None]).squeeze(1)
        other = logit.scatter(1, yte[:, None], float('-inf')).max(1).values
        margin = float((true - other).median())
    return acc, margin, k


def analyse(X, lab, pos, half, n_docs, k, l2, iters, seed):
    tr = pos < half; te = ~tr
    Xtr, ytr, Xte, yte = X[tr], lab[tr], X[te], lab[te]
    dp, m = dprime(Xtr, ytr, Xte, yte, n_docs)
    t = torch.tensor(dp, dtype=torch.float64)
    q = torch.quantile(t, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)).tolist() if len(dp) else [float('nan')] * 3
    acc_c = centroid_accuracy(m, Xtr.mean(0), Xte, yte)
    acc_l, margin, kk = logistic(Xtr, ytr, Xte, yte, n_docs, k, l2, iters, seed)
    return dict(n_train=int(tr.sum()), n_test=int(te.sum()), dprime_q25=q[0], dprime_median=q[1], dprime_q75=q[2],
                dprime_per_doc=dp, acc_centroid=acc_c, acc_logistic=acc_l, margin_logistic_median=margin,
                pca_k=kk, chance=1.0 / n_docs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--revision', default=None)
    ap.add_argument('--corpus-jsonl', required=True)
    ap.add_argument('--tokens', type=int, default=16384)
    ap.add_argument('--n-texts', type=int, default=None)
    ap.add_argument('--control', choices=['shuffle'], default=None)
    ap.add_argument('--stride', type=int, default=128)
    ap.add_argument('--min-pos', type=int, default=1024)
    ap.add_argument('--pca-k', type=int, default=256)
    ap.add_argument('--l2', type=float, default=1e-3)
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    ap.add_argument('--device', default=None, help='device for the analysis (default: model device)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    half = a.tokens // 2
    if a.min_pos < M.SKIP_FIRST or a.min_pos >= half or a.stride < 1:
        sys.exit('need SKIP_FIRST <= min-pos < tokens/2 and stride >= 1')
    t0 = time.time()
    cfg = M.AutoConfig.from_pretrained(a.model, revision=a.revision)
    tc = getattr(cfg, 'text_config', cfg)
    ctx = getattr(tc, 'max_position_embeddings', None)
    if ctx is not None and a.tokens > ctx:
        sys.exit(f'--tokens {a.tokens} exceeds max_position_embeddings {ctx}')
    tok = M.AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    ns = SimpleNamespace(legacy_zip=None, corpus_jsonl=a.corpus_jsonl, sources=None, n_texts=a.n_texts,
                         min_doc_tokens=None, tokens=a.tokens, control=a.control)
    ids_all, meta, skipped = M.build_inputs(ns, tok)
    if len(ids_all) < 3:
        sys.exit(f'only {len(ids_all)} documents qualify')
    model, loader, attn_impl, linfo = M.load_model(a.model, a.revision, a.dtype)
    model.eval()
    emb, blocks, norm = M.parts(model)
    keys = ['0'] + [str(i + 1) for i in range(len(blocks))] + ['norm']
    pts = {k: [] for k in keys}; labs = {k: [] for k in keys}; poss = {k: [] for k in keys}
    state = {'seen': set(), 'doc': 0}

    def capture(key):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            h = h[0].detach()
            if not torch.isfinite(h).all():
                raise RuntimeError(f'nonfinite activations at state {key}')
            x, p = select_positions(h, a.stride, a.min_pos)
            pts[key].append(x.to(h.dtype).cpu()); poss[key].append(p.cpu())
            labs[key].append(torch.full((x.shape[0],), state['doc'], dtype=torch.long))
            state['seen'].add(key)
        return hook

    hs = [emb.register_forward_hook(capture('0'))]
    hs += [b.register_forward_hook(capture(str(i + 1))) for i, b in enumerate(blocks)]
    hs.append(norm.register_forward_hook(capture('norm')))
    dev = emb.weight.device
    try:
        with torch.no_grad():
            for j, ids in enumerate(ids_all):
                state['seen'] = set(); state['doc'] = j
                model(input_ids=ids.to(dev), use_cache=False)
                if state['seen'] != set(keys):
                    raise RuntimeError(f'missing states {set(keys) - state["seen"]}')
                print(f'forward {j + 1}/{len(ids_all)} {time.time() - t0:.0f}s', file=sys.stderr, flush=True)
    finally:
        for h in hs:
            h.remove()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cdev = torch.device(a.device) if a.device else dev
    layers = []
    for key in keys:
        X = torch.cat(pts[key]).to(cdev, torch.float32)
        lab = torch.cat(labs[key]).to(cdev); pos = torch.cat(poss[key]).to(cdev)
        pts[key] = labs[key] = poss[key] = None
        rec = analyse(X, lab, pos, half, len(ids_all), a.pca_k, a.l2, a.iters, a.seed)
        rec.update(state=key, kind='embedding' if key == '0' else ('final_norm' if key == 'norm' else 'block'))
        layers.append(rec)
        print(f"state={key} dprime_median={rec['dprime_median']:.3f} acc_centroid={rec['acc_centroid']:.3f} "
              f"acc_logistic={rec['acc_logistic']:.3f}", file=sys.stderr, flush=True)
        del X
    import transformers
    out = dict(schema=SCHEMA, model=a.model, revision=a.revision, resolved_revision=getattr(cfg, '_commit_hash', None),
               loader=loader, attn_implementation=attn_impl, loading_info=linfo, control=a.control, dtype=a.dtype,
               tokens=a.tokens, corpus=a.corpus_jsonl, docs=meta, skipped=skipped, stride=a.stride, min_pos=a.min_pos,
               split='train: kept positions < tokens/2; test: >= tokens/2', pca_k=a.pca_k, l2=a.l2, iters=a.iters,
               seed=a.seed, skip_first=M.SKIP_FIRST, hi_norm=M.HI_NORM, n_blocks=len(blocks), layers=layers,
               seconds=round(time.time() - t0, 1), versions=dict(torch=torch.__version__, transformers=transformers.__version__),
               source=dict(script='dim_probe.py', sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest(),
                           measurement_module='dim_v7_hfinit_fixed.py',
                           measurement_sha256=hashlib.sha256(open(M.__file__, 'rb').read()).hexdigest()))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
    path = a.out if a.out.endswith('.json') else a.out + '.json'
    with open(path, 'w') as f:
        json.dump(out, f, allow_nan=False)
    print('wrote', path, file=sys.stderr)


if __name__ == '__main__':
    main()
