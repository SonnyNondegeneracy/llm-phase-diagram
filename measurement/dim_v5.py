#!/usr/bin/env python3
"""v5 (= v4 measurement, unchanged, plus the --faithful initialisation mode): layer-wise token correlator E and participation ratio PR, two definitions, two forward passes.

Kept positions of a document at one state: drop the first SKIP_FIRST positions (attention sinks) and the last one,
then drop positions with |h| > HI_NORM x median |h| (pre-registered rule, identical for every model).
All quantities below are computed on the kept positions x_1..x_n (float64).

Definition (i), per document (continuity with v1-v3):
  E_doc   = sum_{i!=j} x_i.x_j / ((n-1) sum_i |x_i|^2)            (uncentred)
  PR_doc  = (tr C)^2 / tr(C^2),  C = covariance about the document mean mu
Definition (ii), reference-centred: y_i = x_i - mubar, with mubar the token-weighted mean of the OTHER documents
  E_ref   = sum_{i!=j} y_i.y_j / ((n-1) sum_i |y_i|^2)
  PR_ref  = (tr M)^2 / tr(M^2),  M = (1/n) sum_i y_i y_i^T
  computed for two references: 'all' (all other documents) and 'src' (other documents of the same source).
  With delta = mu - mubar:  tr M = trC + |delta|^2,  tr M^2 = trC2 + 2 delta^T C delta + |delta|^4,
  E_ref = (n |delta|^2 / tr M - 1) / (n - 1).  Pass 1 stores every document mean; pass 2 reruns the forward pass and
  evaluates delta^T C delta.  Doc 0 is also checked by explicit centring at every state (self-check).
Pass 2 also stores, per document and state, B = U^T C U and A = U^T mu_j for an orthonormal basis U of the span of
all document means, so that any other reference built from document means can be evaluated offline exactly.

States (forward hooks): 0 embedding output, 1..L raw block outputs, 'norm' final-norm output.

usage:
  python dim_v4.py --model Qwen/Qwen2.5-7B --corpus-jsonl corpus_v4/corpus_v4.jsonl --tokens 16384 --out runs4/A_qwen2.5-7b
  python dim_v4.py --model Qwen/Qwen2.5-7B --legacy-zip dimension.zip --n-texts 5 --tokens 8192 --out runs4/gate_qwen2.5-7b
"""
import argparse, hashlib, json, sys, time, zipfile

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

SKIP_FIRST = 16
HI_NORM = 8.0
LEGACY_ROWS = [93, 106, 300, 324, 420, 452, 867, 940, 1060, 1163, 1186, 1230, 1489, 1631, 1632, 1694, 1993, 2008, 2020, 2159]


def parts(model):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        m = model.model
        return m.embed_tokens, m.layers, m.norm
    if hasattr(model, 'gpt_neox'):
        m = model.gpt_neox
        return m.embed_in, m.layers, m.final_layer_norm
    raise SystemExit('unsupported architecture: add it to parts()')


def randomize_blocks(blocks, seed, std=None, faithful=False):
    """Replace every block parameter. The random stream is identical in all modes (one N(0,1) draw per parameter, in
    parameter order, generator seeded per block), so for a given seed every matrix is the SAME random matrix up to scale.
      unit      (std None)            : every parameter <- w
      scaled    (std s)               : every parameter <- s * w   (v4 'init-std' mode: norm gains and biases too)
      faithful  (std s, faithful=True): matrices (ndim>=2) <- s * w; 1-D '...bias' <- 0; other 1-D (norm gains) <- 1
    Returns a summary of the parameter classes for the record."""
    summ = {}
    for b, block in enumerate(blocks):
        g = torch.Generator(device='cpu').manual_seed(seed * 100003 + b)
        for name, p in block.named_parameters():
            w = torch.randn(p.shape, generator=g, dtype=torch.float32)
            if faithful and p.dim() == 1:
                cls = 'bias' if name.endswith('bias') else 'gain'
                w = torch.zeros_like(w) if cls == 'bias' else torch.ones_like(w)
            else:
                cls = 'matrix' if p.dim() >= 2 else 'vector'
                w = w if std is None else w * std
            p.data.copy_(w.to(p.dtype))
            if b == 0:
                v = p.data.float()
                summ.setdefault(cls, []).append(dict(name=name, shape=list(p.shape), mean=float(v.mean()), std=float(v.std()) if v.numel() > 1 else 0.0))
    return summ


def kept(h):
    x = h[SKIP_FIRST:-1].double()
    nrm = x.norm(dim=1)
    med = nrm.median()
    keep = nrm <= HI_NORM * med
    return x, x[keep], int((~keep).sum())


def e_uncentred(x):
    n = x.shape[0]; n2 = (x * x).sum(1); s = x.sum(0)
    return float(((s @ s) - n2.sum()) / ((n - 1) * n2.sum()))


def pr_centred(x):
    xc = x - x.mean(0)
    C = xc.T @ xc / x.shape[0]
    return float(torch.trace(C) ** 2 / (C * C).sum())


def pass1_stats(h):
    xall, x, n_hi = kept(h)
    n = x.shape[0]
    mu = x.mean(0)
    xc = x - mu
    C = xc.T @ xc / n
    trC = float(torch.trace(C)); trC2 = float((C * C).sum())
    rec = dict(n=n, n_hi=n_hi, m2=float(mu @ mu), trC=trC, trC2=trC2,
               E_doc=e_uncentred(x), PR_doc=trC * trC / trC2,
               E_doc_unclipped=e_uncentred(xall), PR_doc_unclipped=pr_centred(xall))
    return rec, mu


def ref_from(delta, q, n, trC, trC2):
    dd = float(delta @ delta); trM = trC + dd
    return dict(E=(n * dd / trM - 1) / (n - 1), PR=trM * trM / (trC2 + 2 * q + dd * dd), delta2=dd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--revision', default=None)
    ap.add_argument('--corpus-jsonl', default=None)
    ap.add_argument('--legacy-zip', default=None, help='gate only: the v1-v3 dimension.zip texts')
    ap.add_argument('--n-texts', type=int, default=None)
    ap.add_argument('--sources', nargs='*', default=None, help='restrict to these corpus sources (e.g. pg19)')
    ap.add_argument('--min-doc-tokens', type=int, default=None, help='keep only documents with at least this many tokens under this tokenizer')
    ap.add_argument('--tokens', type=int, default=16384)
    ap.add_argument('--allow-overflow', action='store_true', help='permit tokens > max_position_embeddings (never for registered runs)')
    ap.add_argument('--random-blocks', action='store_true')
    ap.add_argument('--seed', type=int, default=41)
    ap.add_argument('--random-std', type=float, default=None)
    ap.add_argument('--faithful', action='store_true', help='with --random-blocks --random-std: matrices N(0,std^2), norm gains 1, biases 0')
    ap.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    ap.add_argument('--out', required=True, help='output prefix: writes <out>.json and <out>.proj.pt')
    a = ap.parse_args()
    if a.faithful and not (a.random_blocks and a.random_std):
        sys.exit('--faithful requires --random-blocks and --random-std')

    cfg = AutoConfig.from_pretrained(a.model, revision=a.revision)
    ctx = getattr(cfg, 'max_position_embeddings', None)
    if ctx is not None and a.tokens > ctx and not a.allow_overflow:
        sys.exit(f'--tokens {a.tokens} exceeds max_position_embeddings {ctx}')
    if getattr(cfg, 'use_sliding_window', False) or (getattr(cfg, 'sliding_window', None) and not hasattr(cfg, 'use_sliding_window') and cfg.sliding_window < a.tokens):
        sys.exit('sliding-window attention shorter than --tokens: not permitted')

    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    if a.legacy_zip:
        with zipfile.ZipFile(a.legacy_zip) as z:
            rows = json.loads(z.read('dimension/data/input_texts.json'))
        idx = LEGACY_ROWS[:a.n_texts or 5]
        docs = [dict(id=str(i), source='legacy', text=rows[i][:131072]) for i in idx]
    else:
        docs = [json.loads(l) for l in open(a.corpus_jsonl, encoding='utf-8') if l.strip()]
        if a.sources:
            docs = [d for d in docs if d['source'] in a.sources]
        if a.n_texts:
            docs = docs[:a.n_texts]
    ids_all, meta, skipped = [], [], []
    for d in docs:
        ids = tok(d['text'], return_tensors='pt', truncation=True, max_length=a.tokens).input_ids
        need = max(a.tokens, a.min_doc_tokens or 0)
        if a.min_doc_tokens and len(tok(d['text']).input_ids) < a.min_doc_tokens or ids.shape[1] < a.tokens:
            skipped.append(dict(id=d['id'], tokens=int(ids.shape[1]), required=need)); continue
        ids_all.append(ids); meta.append(dict(id=d['id'], source=d['source'], tokens=int(ids.shape[1])))
    N = len(ids_all)
    if N < 3:
        sys.exit(f'only {N} documents qualify')
    print(f'{N} documents, {len(skipped)} skipped', file=sys.stderr)

    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, torch_dtype=getattr(torch, a.dtype),
                                                 device_map='auto', attn_implementation='sdpa')
    model.eval()
    emb, blocks, norm = parts(model)
    init_summary = None
    if a.random_blocks:
        init_summary = randomize_blocks(blocks, a.seed, a.random_std, a.faithful)
    L = len(blocks)
    keys = ['0'] + [str(i) for i in range(1, L + 1)] + ['norm']
    dev = model.get_input_embeddings().weight.device

    state = dict(fn=None, doc=None)
    def hook(key):
        def f(mod, inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            state['fn'](key, h[0].detach())
        return f
    hs = [emb.register_forward_hook(hook('0'))] + [b.register_forward_hook(hook(str(i + 1))) for i, b in enumerate(blocks)]
    hs.append(norm.register_forward_hook(hook('norm')))

    # ---- pass 1: per-document statistics and means
    P1 = {k: [None] * N for k in keys}; MU = {k: [None] * N for k in keys}; digest1, nonfinite = [], []
    def f1(key, h):
        j = state['doc']
        if not torch.isfinite(h).all():
            nonfinite.append([meta[j]['id'], key]); return
        P1[key][j], MU[key][j] = pass1_stats(h)
        if key == '0':
            digest1.append(hashlib.sha256(h.float().cpu().numpy().tobytes()).hexdigest()[:16])
    state['fn'] = f1
    t0 = time.time()
    with torch.no_grad():
        for j, ids in enumerate(ids_all):
            state['doc'] = j; model(ids.to(dev), use_cache=False)
            print(f'pass1 {j + 1}/{N} {time.time() - t0:.0f}s', file=sys.stderr)
    if nonfinite:
        sys.exit(f'non-finite activations: {nonfinite[:5]}')

    # references and span basis per state
    src = [m['source'] for m in meta]
    REF, BASIS = {}, {}
    for k in keys:
        Mu = torch.stack(MU[k])                               # [N, D] float64 on device
        w = torch.tensor([P1[k][j]['n'] for j in range(N)], dtype=Mu.dtype, device=Mu.device)
        tot = (w[:, None] * Mu).sum(0); wt = w.sum()
        refs = []
        for j in range(N):
            r = {'all': (tot - w[j] * Mu[j]) / (wt - w[j])}
            same = [i for i in range(N) if src[i] == src[j] and i != j]
            if same:
                ws = w[same]; r['src'] = (ws[:, None] * Mu[same]).sum(0) / ws.sum()
            refs.append(r)
        REF[k] = refs
        U, _ = torch.linalg.qr(Mu.T)                          # [D, N] orthonormal basis of span{mu_j}
        BASIS[k] = U

    # ---- pass 2: delta^T C delta, projected covariance, self-check
    P2 = {k: [None] * N for k in keys}; PROJ_B = {k: [None] * N for k in keys}; digest2, check = [], []
    def f2(key, h):
        j = state['doc']
        _, x, _ = kept(h)
        n = x.shape[0]; mu = MU[key][j]; xc = x - mu
        p1 = P1[key][j]; out = {}
        for name, mubar in REF[key][j].items():
            delta = mu - mubar
            v = xc @ delta
            out[name] = ref_from(delta, float(v @ v) / n, n, p1['trC'], p1['trC2'])
            if j == 0:  # explicit centring
                y = x - mubar
                e_dir = e_uncentred(y); M = y.T @ y / n; pr_dir = float(torch.trace(M) ** 2 / (M * M).sum())
                check.append(dict(state=key, ref=name, E_rel=abs(e_dir - out[name]['E']) / max(abs(e_dir), 1e-30),
                                  PR_rel=abs(pr_dir - out[name]['PR']) / pr_dir))
        P2[key][j] = out
        Pj = xc @ BASIS[key]
        PROJ_B[key][j] = (Pj.T @ Pj / n).float().cpu()
        if key == '0':
            digest2.append(hashlib.sha256(h.float().cpu().numpy().tobytes()).hexdigest()[:16])
    state['fn'] = f2
    with torch.no_grad():
        for j, ids in enumerate(ids_all):
            state['doc'] = j; model(ids.to(dev), use_cache=False)
            print(f'pass2 {j + 1}/{N} {time.time() - t0:.0f}s', file=sys.stderr)
    for h in hs:
        h.remove()

    layers = []
    for k in keys:
        per = [dict(P1[k][j], ref=P2[k][j]) for j in range(N)]
        layers.append(dict(state=k, kind='embedding' if k == '0' else ('final_norm' if k == 'norm' else 'block'), per_doc=per))
    out = dict(schema='dim_v5/1', model=a.model, revision=a.revision, resolved_revision=getattr(model.config, '_commit_hash', None),
               mode=('trained' if not a.random_blocks else 'random_blocks_unit_gaussian' if a.random_std is None else f'random_blocks_faithful_std{a.random_std:g}' if a.faithful else f'random_blocks_gaussian_std{a.random_std:g}'),
               random_std=a.random_std, faithful=a.faithful, init_summary=init_summary,
               seed=a.seed if a.random_blocks else None, dtype=a.dtype, n_blocks=L, hidden=int(cfg.hidden_size),
               max_position_embeddings=ctx, tokens=a.tokens, skip_first=SKIP_FIRST, hi_norm=HI_NORM,
               corpus=a.corpus_jsonl or a.legacy_zip, docs=meta, skipped=skipped,
               digest_pass1=digest1, digest_pass2=digest2, deterministic=digest1 == digest2,
               self_check=check, self_check_max=dict(E=max(c['E_rel'] for c in check), PR=max(c['PR_rel'] for c in check)),
               layers=layers, seconds=round(time.time() - t0, 1))
    json.dump(out, open(a.out + '.json', 'w'))
    torch.save(dict(keys=keys, docs=meta,
                    A={k: (BASIS[k].T @ torch.stack(MU[k]).T).T.cpu() for k in keys},   # [N, N]: coords of each doc mean
                    B={k: torch.stack(PROJ_B[k]) for k in keys}),                         # [N, N, N]: U^T C_j U per doc
               a.out + '.proj.pt')
    print('wrote', a.out, 'deterministic', out['deterministic'], 'self_check', out['self_check_max'], file=sys.stderr)


if __name__ == '__main__':
    main()
