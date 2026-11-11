#!/usr/bin/env python3
"""Classifier meaning of E and E*d (probe v8.1). Schema dim_probe/2.

Point set: identical to dim_probe.py (v8). Every document is forwarded once at full length (--tokens); at each state the
residual vectors at kept positions p >= MIN_POS on the STRIDE grid are taken (dim_v7 kept rule).

ONE split for every classifier (main methodology): the positions of each document are cut into blocks of --block tokens
(default 1024). The usable blocks (those at or after --min-pos) of each document are randomly assigned, with a seed
fixed per document and identical for every state, to TRAIN / VALID / TEST in the proportion --split (default 9/3/3 of
the 15 usable blocks at 16384 tokens). TRAIN provides contexts / fits, VALID chooses hyper-parameters or thresholds,
TEST is used once for the reported numbers.

Readouts per state (all on the same split):
  logistic     96-way multinomial logistic regression (dim_probe.logistic: top-K principal components of the fit set,
               standardised, L2, full-batch L-BFGS). Grid K x L2 fitted on TRAIN, chosen by VALID accuracy, refitted on
               TRAIN+VALID, reported on TEST. The v8 default (K=256, L2=1e-3, refitted on TRAIN+VALID) is also reported.
  membership   binary context-membership test for every document j: context = mean of m TRAIN tokens of j (spread over
               distinct blocks), centre c_j = mean of the TRAIN tokens of the other documents (the centre of E);
               score variants
                 dot : S(x) = (x - c_j) . (ybar - c_j)                          (main)
                 cos : S(x) = dot / |x - c_j|                                    (ablation)
                 dist: S(x) = -|x - ybar|^2                                      (ablation)
               positives = TEST tokens of j, negatives = TEST tokens of the other documents. Per document:
                 dprime2 = [(mean_pos - mean_neg)^2 - var_pos/n_pos - var_neg/n_neg] / (0.5 (var_pos + var_neg))
                           (score SNR, power ratio; the subtracted terms remove the small-sample bias)
                 auc     = Mann-Whitney AUC;  da2 = 2 * Phi^-1(auc)^2              (distribution-free SNR)
                 bacc    = balanced accuracy on TEST at the threshold maximising balanced accuracy on VALID
               m in --m-list (plus 'all' = every TRAIN token of j); --draws random contexts per m (1 for 'all').
               dprime2 per document = (draw-averaged numerator) / (draw-averaged denominator), i.e. the SNR of the score
               over queries and contexts (a per-draw ratio would be biased upward by the spread of the context norm);
               auc and bacc per document = mean over draws. Median and quartiles over documents.
  multiclass   96-way context classification with the same contexts: dot = argmax_j (x - mu).(ybar_j - mu),
               dist = argmin_j |x - ybar_j|^2 (mu = mean of all TRAIN tokens). TEST accuracy, the largest share of
               predictions taken by one document and the Gini coefficient of the prediction counts.
  geometry     on the TRAIN tokens of each document, relative to c_j, unbiased U-statistics over pairs i != k:
                 E_u = mean(y_i.y_k) / mean|y|^2,  d_u = (mean|y|^2)^2 / mean((y_i.y_k)^2),  Ed_u = E_u * d_u;
               align = delta' C delta / (|delta|^2 lambda_w)  (align = 1 <=> generic alignment, d'^2 = E*d), with
                 |delta|^2 = mean(y_i.y_k) (U-statistic), lambda_w = mean((y_i.y_k)^2) / mean|y|^2, and delta' C delta
                 estimated without bias by split halves: the TRAIN blocks of j are split into two halves giving
                 u1, u2 (half means - c_j); delta' C delta = covariance over the TEST tokens of j of (x.u1, x.u2).
  legacy       dim_probe.analyse on the same points with the v8 front/back half split (gate: must equal the v8 run).
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

import dim_probe as P
import dim_v7_hfinit_fixed as M

SCHEMA = 'dim_probe/2'
GRID_K = (128, 256, 512)
GRID_L2 = (1e-1, 1e-2, 1e-3, 1e-4)      # strongest regularisation first: ties go to the simpler model


def assign_blocks(n_docs, tokens, block, min_pos, split, seed):
    """{doc: {block_id: 'train'|'valid'|'test'}}; depends only on the document index and the seed."""
    usable = list(range(-(-min_pos // block), tokens // block))
    nt, nv, ns = split
    if nt + nv + ns != len(usable):
        sys.exit(f'--split {split} must add up to the {len(usable)} usable blocks')
    out = []
    for j in range(n_docs):
        g = torch.Generator(device='cpu').manual_seed(seed * 100003 + j)
        perm = [usable[i] for i in torch.randperm(len(usable), generator=g, device='cpu').tolist()]
        out.append({b: ('train' if i < nt else 'valid' if i < nt + nv else 'test') for i, b in enumerate(perm)})
    return out


def parts_of(lab, pos, assign, block):
    blk = (pos // block).tolist(); labs = lab.tolist()
    names = [assign[j].get(b, 'none') for j, b in zip(labs, blk)]
    return {s: torch.tensor([n == s for n in names], device=lab.device) for s in ('train', 'valid', 'test')}


def context_order(train_idx, train_blk, gen):
    """Order of a document's TRAIN tokens such that consecutive picks come from different blocks."""
    blocks = sorted(set(train_blk.tolist()))
    blocks = [blocks[i] for i in torch.randperm(len(blocks), generator=gen, device='cpu').tolist()]
    per = []
    for b in blocks:
        ii = train_idx[train_blk == b]
        per.append(ii[torch.randperm(len(ii), generator=gen, device='cpu')].tolist())
    order = []
    for r in range(max(len(p) for p in per)):
        order += [p[r] for p in per if r < len(p)]
    return order


def auc_cols(S, pos_mask):
    """Mann-Whitney AUC per column: S [n, J] scores, pos_mask [n, J] bool."""
    r = S.argsort(0).argsort(0).to(torch.float64) + 1
    npos = pos_mask.sum(0).to(torch.float64); nneg = S.shape[0] - npos
    return ((r * pos_mask).sum(0) - npos * (npos + 1) / 2) / (npos * nneg)


def bacc_threshold(Sv, posv, St, post):
    """Threshold maximising balanced accuracy on VALID (per column), applied to TEST."""
    out = []
    for j in range(Sv.shape[1]):
        s, o = Sv[:, j].sort(descending=True); p = posv[o, j].to(torch.float64)
        tp = p.cumsum(0); fp = (1 - p).cumsum(0)
        ba = 0.5 * (tp / p.sum() + 1 - fp / (1 - p).sum())
        i = int(ba.argmax()); t = float(s[i]) if i + 1 >= len(s) else 0.5 * float(s[i] + s[i + 1])
        pr = St[:, j] >= t; q = post[:, j]
        out.append(0.5 * (float((pr & q).sum()) / max(1, int(q.sum())) + float((~pr & ~q).sum()) / max(1, int((~q).sum()))))
    return torch.tensor(out, dtype=torch.float64, device='cpu')


def snr_cols(S, pos_mask):
    """Numerator and denominator of the score SNR per column. Numerator bias-corrected:
    E[(mp_hat - mn_hat)^2] = (mp - mn)^2 + vp/np + vn/nn."""
    p = pos_mask.to(S.dtype); n = 1 - p
    np_, nn_ = p.sum(0), n.sum(0)
    mp = (S * p).sum(0) / np_; mn = (S * n).sum(0) / nn_
    vp = ((S - mp) ** 2 * p).sum(0) / (np_ - 1); vn = ((S - mn) ** 2 * n).sum(0) / (nn_ - 1)
    return (mp - mn) ** 2 - vp / np_ - vn / nn_, 0.5 * (vp + vn), vp


def gini(counts):
    c = counts.sort().values.to(torch.float64); n = len(c)
    return float((2 * torch.arange(1, n + 1, dtype=torch.float64, device=c.device) - n - 1).mul(c).sum() / (n * c.sum()))


def q3(t):
    t = torch.as_tensor(t, dtype=torch.float64, device='cpu') if not torch.is_tensor(t) else t.detach().to('cpu', torch.float64)
    t = t[torch.isfinite(t)]
    if t.numel() == 0:
        return dict(q25=None, median=None, q75=None)
    q = torch.quantile(t, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64, device='cpu')).tolist()
    return dict(q25=q[0], median=q[1], q75=q[2])


@torch.no_grad()
def context_readouts(X, lab, part, blk, n_docs, m_list, draws, seed):
    tr, va, te = part['train'], part['valid'], part['test']
    dev = X.device
    idx = torch.arange(X.shape[0], device=dev)
    S_tr = torch.zeros(n_docs, X.shape[1], dtype=X.dtype, device=dev).index_add_(0, lab[tr], X[tr])
    n_tr = torch.bincount(lab[tr], minlength=n_docs).to(X.dtype)
    C = (S_tr.sum(0)[None] - S_tr) / (n_tr.sum() - n_tr)[:, None]          # c_j: centre of E (other documents)
    mu = S_tr.sum(0) / n_tr.sum()                                         # common centre for multiclass
    Xv, Xt = X[va], X[te]; lv, lt = lab[va], lab[te]
    J = torch.arange(n_docs, device=dev)
    posv = lv[:, None] == J[None]; post = lt[:, None] == J[None]
    nv2 = (Xv * Xv).sum(1, keepdim=True); nt2 = (Xt * Xt).sum(1, keepdim=True)
    cc = (C * C).sum(1)[None]
    normv = (nv2 - 2 * Xv @ C.T + cc).clamp_min(1e-30).sqrt(); normt = (nt2 - 2 * Xt @ C.T + cc).clamp_min(1e-30).sqrt()
    orders = []
    for j in range(n_docs):
        ii = idx[tr & (lab == j)]
        orders.append([context_order(ii.cpu(), blk[ii].cpu(), torch.Generator(device='cpu').manual_seed(seed * 7919 + j * 131 + r))
                       for r in range(draws)])
    n_min = int(n_tr.min())
    ms = [m for m in m_list if m <= n_min] + ['all']
    mem = {v: {} for v in ('dot', 'cos', 'dist')}; mc = {v: {} for v in ('dot', 'dist')}
    keep_all = {}
    for m in ms:
        R = 1 if m == 'all' else draws
        acc = {v: dict(num=[], den=[], auc=[], bacc=[], vpos=[]) for v in mem}
        mca = {v: dict(acc=[], max_share=[], gini=[]) for v in mc}
        for r in range(R):
            if m == 'all':
                Y = S_tr / n_tr[:, None]
            else:
                Y = torch.stack([X[torch.tensor(orders[j][r][:m], device=dev)].mean(0) for j in range(n_docs)])
            U = Y - C                                                      # ybar_j - c_j
            off = (C * U).sum(1)[None]
            dv = Xv @ U.T - off; dt = Xt @ U.T - off
            yy = (Y * Y).sum(1)[None]
            sc = dict(dot=(dv, dt), cos=(dv / normv, dt / normt),
                      dist=(-(nv2 - 2 * Xv @ Y.T + yy), -(nt2 - 2 * Xt @ Y.T + yy)))
            for v, (Sv, St) in sc.items():
                nu, de, vp = snr_cols(St.double(), post)
                acc[v]['num'].append(nu); acc[v]['den'].append(de); acc[v]['auc'].append(auc_cols(St, post))
                acc[v]['bacc'].append(bacc_threshold(Sv, posv, St, post)); acc[v]['vpos'].append(vp)
            for v, Sm in (('dot', (Xt - mu) @ (Y - mu).T), ('dist', -(nt2 - 2 * Xt @ Y.T + yy))):
                pred = Sm.argmax(1); cnt = torch.bincount(pred, minlength=n_docs)
                mca[v]['acc'].append(float((pred == lt).double().mean()))
                mca[v]['max_share'].append(float(cnt.max()) / len(pred)); mca[v]['gini'].append(gini(cnt))
        key = str(m)
        for v in mem:
            # SNR of the score over queries AND contexts: ratio of the draw-averaged numerator and denominator
            d2 = torch.stack(acc[v]['num']).mean(0) / torch.stack(acc[v]['den']).mean(0)
            au = torch.stack(acc[v]['auc']).mean(0)
            ba = torch.stack(acc[v]['bacc']).mean(0)
            da2 = 2 * torch.special.ndtri(au.clamp(1e-6, 1 - 1e-6)) ** 2 * torch.sign(au - 0.5)
            mem[v][key] = dict(m=m, draws=R, dprime2=q3(d2), auc=q3(au), da2=q3(da2), bacc=q3(ba))
            if m == 'all' and v == 'dot':
                keep_all = dict(U=U, vpos=acc[v]['vpos'][0], dprime2=d2, auc=au)
        for v in mc:
            mc[v][key] = dict(m=m, draws=R, acc=sum(mca[v]['acc']) / R, max_share=sum(mca[v]['max_share']) / R,
                              gini=sum(mca[v]['gini']) / R)
    # geometry: U-statistics on TRAIN tokens of each document relative to c_j
    geo = dict(E_u=[], d_u=[], Ed_u=[], align=[])
    for j in range(n_docs):
        Yj = (X[tr & (lab == j)] - C[j]).double()
        G = Yj @ Yj.T; n = G.shape[0]; off = ~torch.eye(n, dtype=torch.bool, device=dev)
        m1 = float(G.diagonal().mean()); p1 = float(G[off].mean()); p2 = float((G[off] ** 2).mean())
        E_u = p1 / m1; d_u = m1 * m1 / p2; lam_w = p2 / m1
        ii = idx[tr & (lab == j)]; bj = blk[ii]; ub = sorted(set(bj.tolist()))
        h1 = torch.tensor([b in ub[0::2] for b in bj.tolist()], device=dev)
        u1 = (X[ii[h1]].mean(0) - C[j]).double(); u2 = (X[ii[~h1]].mean(0) - C[j]).double()
        Tj = Xt[lt == j].double(); a1 = Tj @ u1; a2 = Tj @ u2
        dCd = float(((a1 - a1.mean()) * (a2 - a2.mean())).sum() / (len(a1) - 1))
        geo['E_u'].append(E_u); geo['d_u'].append(d_u); geo['Ed_u'].append(E_u * d_u)
        geo['align'].append(dCd / (p1 * lam_w))
    geometry = {k: q3(v) for k, v in geo.items()}
    per_doc = dict(dprime2_all=keep_all['dprime2'].tolist(), auc_all=keep_all['auc'].tolist(), **{k: v for k, v in geo.items()})
    return mem, mc, geometry, per_doc, ms


def logistic_block(X, lab, part, n_docs, iters, seed, grid):
    tr, va, te = part['train'], part['valid'], part['test']
    tv = tr | va
    res = dict(grid=[])
    best = None
    if grid:
        for l2 in GRID_L2:
            for k in GRID_K:
                a, _, kk = P.logistic(X[tr], lab[tr], X[va], lab[va], n_docs, k, l2, iters, seed)
                res['grid'].append(dict(k=kk, l2=l2, acc_valid=a))
                if best is None or a > best[0]:
                    best = (a, k, l2)
        a, mg, kk = P.logistic(X[tv], lab[tv], X[te], lab[te], n_docs, best[1], best[2], iters, seed)
        res['selected'] = dict(k=kk, l2=best[2], acc_valid=best[0], acc_test=a, margin_test=mg)
    a, mg, kk = P.logistic(X[tv], lab[tv], X[te], lab[te], n_docs, 256, 1e-3, iters, seed)
    res['default'] = dict(k=kk, l2=1e-3, acc_test=a, margin_test=mg)
    return res


def analyse_state(X, lab, pos, a, assign, n_docs):
    half = a.tokens // 2
    blk = pos // a.block
    part = parts_of(lab, pos, assign, a.block)
    rec = dict(n_train=int(part['train'].sum()), n_valid=int(part['valid'].sum()), n_test=int(part['test'].sum()),
               chance=1.0 / n_docs)
    leg = P.analyse(X, lab, pos, half, n_docs, a.pca_k, a.l2, a.iters, a.seed)
    rec['legacy'] = {k: leg[k] for k in ('n_train', 'n_test', 'dprime_q25', 'dprime_median', 'dprime_q75',
                                         'acc_centroid', 'acc_logistic', 'margin_logistic_median', 'pca_k')}
    rec['logistic'] = logistic_block(X, lab, part, n_docs, a.iters, a.seed, not a.no_grid)
    mem, mc, geo, per_doc, ms = context_readouts(X, lab, part, blk, n_docs, a.m_list, a.draws, a.seed)
    rec.update(membership=mem, multiclass=mc, geometry=geo, per_doc=per_doc, m_used=[str(m) for m in ms])
    return rec


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
    ap.add_argument('--block', type=int, default=1024)
    ap.add_argument('--split', type=int, nargs=3, default=[9, 3, 3], metavar=('TRAIN', 'VALID', 'TEST'))
    ap.add_argument('--m-list', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument('--draws', type=int, default=20)
    ap.add_argument('--pca-k', type=int, default=256, help='legacy (v8) readout only')
    ap.add_argument('--l2', type=float, default=1e-3, help='legacy (v8) readout only')
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--no-grid', action='store_true', help='skip the logistic hyper-parameter grid (tests only)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    ap.add_argument('--device', default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    if a.min_pos < M.SKIP_FIRST or a.min_pos >= a.tokens // 2 or a.stride < 1 or a.block % a.stride:
        sys.exit('need SKIP_FIRST <= min-pos < tokens/2, stride >= 1, block a multiple of stride')
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
    n_docs = len(ids_all)
    if n_docs < 3:
        sys.exit(f'only {n_docs} documents qualify')
    assign = assign_blocks(n_docs, a.tokens, a.block, a.min_pos, a.split, a.seed)
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
            x, p = P.select_positions(h, a.stride, a.min_pos)
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
                print(f'forward {j + 1}/{n_docs} {time.time() - t0:.0f}s', file=sys.stderr, flush=True)
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
        rec = analyse_state(X, lab, pos, a, assign, n_docs)
        rec.update(state=key, kind='embedding' if key == '0' else ('final_norm' if key == 'norm' else 'block'))
        layers.append(rec)
        mm = rec['membership']['dot']['all']
        sel = rec['logistic'].get('selected', rec['logistic']['default'])
        print(f"state={key} legacy_acc={rec['legacy']['acc_logistic']:.3f} logistic_test={sel['acc_test']:.3f} "
              f"member_dprime2={mm['dprime2']['median']:.2f} auc={mm['auc']['median']:.3f} "
              f"Ed_u={rec['geometry']['Ed_u']['median']:.2f} align={rec['geometry']['align']['median']:.2f} "
              f"{time.time() - t0:.0f}s", file=sys.stderr, flush=True)
        del X
    import transformers
    out = dict(schema=SCHEMA, model=a.model, revision=a.revision, resolved_revision=getattr(cfg, '_commit_hash', None),
               loader=loader, attn_implementation=attn_impl, loading_info=linfo, control=a.control, dtype=a.dtype,
               tokens=a.tokens, corpus=a.corpus_jsonl, docs=meta, skipped=skipped, stride=a.stride, min_pos=a.min_pos,
               block=a.block, split=a.split, block_assignment=[{str(b): s for b, s in d.items()} for d in assign],
               m_list=a.m_list, draws=a.draws, grid=dict(k=list(GRID_K), l2=list(GRID_L2)) if not a.no_grid else None,
               iters=a.iters, seed=a.seed, legacy=dict(pca_k=a.pca_k, l2=a.l2, split='front/back half (v8)'),
               skip_first=M.SKIP_FIRST, hi_norm=M.HI_NORM, n_blocks=len(blocks), layers=layers,
               seconds=round(time.time() - t0, 1), versions=dict(torch=torch.__version__, transformers=transformers.__version__),
               source=dict(script='dim_probe2.py', sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest(),
                           probe_module='dim_probe.py', probe_sha256=hashlib.sha256(open(P.__file__, 'rb').read()).hexdigest(),
                           measurement_module='dim_v7_hfinit_fixed.py',
                           measurement_sha256=hashlib.sha256(open(M.__file__, 'rb').read()).hexdigest()))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
    path = a.out if a.out.endswith('.json') else a.out + '.json'
    with open(path, 'w') as f:
        json.dump(out, f, allow_nan=False)
    print('wrote', path, file=sys.stderr)


if __name__ == '__main__':
    main()
