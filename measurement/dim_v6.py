#!/usr/bin/env python3
"""v6 = v5 measurement (unchanged definitions) + sub-layer states, causal interventions, position windows, input controls.

Measurement (identical to v4/v5). Kept positions of a document at one state: drop the first SKIP_FIRST positions and the
last one, then drop positions with |h| > HI_NORM x median |h|. On the kept positions x_1..x_n (float64):
  definition (i):  E_doc uncentred, PR_doc about the document mean;
  definition (ii): E_ref, PR_ref on y = x - mubar, mubar = token-weighted mean of the OTHER documents ('all') or of the
                   other documents of the same source ('src'); evaluated exactly via delta = mu - mubar.
States: '0' embedding output; for every block k: 'k.a' (with --sublayer: the residual stream after the attention
sub-layer, i.e. the input of the block's post-attention norm) and 'k' (block output); 'norm' final-norm output.

New in v6
  --sublayer            add the 'k.a' states. Validity is checked on document 0 for every block: the block output minus
                        'k.a' must equal mlp(post_attention_layernorm('k.a')) (times residual_multiplier when the config
                        has one) on the first 256 positions the measurement keeps (after SKIP_FIRST, without high-norm
                        tokens); the relative error is stored in sublayer_check. The same error on raw positions 0..255
                        is stored in sublayer_check_first256 for diagnosis only: in bf16 it is dominated by rounding on
                        the attention-sink tokens, whose norm is up to ~100x the median (fp32 gives ~1e-7 there too).
  --zero-attn B         zero the attention sub-layer output in blocks B (e.g. 1-5).
  --zero-mlp B          zero the MLP sub-layer output in blocks B.
  --local-window W --window-blocks B
                        in blocks B, each token attends only to itself and the previous W-1 tokens (causal band mask).
                        Only softmax-attention modules are affected; others are listed in intervention.skipped_modules.
  position windows      always on: in pass 2, E_ref/PR_ref ('all' reference) are also computed on the kept positions
                        inside absolute-position windows POS_WINDOWS, with the same reference mean.
  --control mix         input control: per source, sequence j is made of 16 chunks of T/16 tokens, chunk c taken from
                        document (j + c) mod N_source at token range [c*T/16, (c+1)*T/16). Every document contributes
                        one chunk to 16 different sequences, so the token multiset over all sequences is unchanged.
  --control shuffle     input control: the T content tokens of each document are permuted (seed 0 + document index).
  --attn-kind K         with --zero-attn: 'all' (default), 'linear' (only linear-attention modules, e.g. Qwen3.5
                        GatedDeltaNet) or 'full' (only softmax self-attention modules); other blocks in the range are
                        listed in intervention.skipped_modules.
  --hf-init             with --random-blocks: re-initialise every block with the model's own _init_weights (torch seed =
                        --seed), i.e. the library's training-start initialisation, including special parameters.
  --inspect             print the architecture summary (classes, layers, attention modules, config) and exit.
  --no-proj             do not write <out>.proj.pt.
usage examples are in HANDOFF.md.
"""
import argparse, hashlib, json, sys, time, zipfile

import torch
from transformers import AutoConfig, AutoTokenizer
import transformers

SKIP_FIRST = 16
HI_NORM = 8.0
POS_WINDOWS = [(16, 1024), (1024, 4096), (4096, None)]
LEGACY_ROWS = [93, 106, 300, 324, 420, 452, 867, 940, 1060, 1163, 1186, 1230, 1489, 1631, 1632, 1694, 1993, 2008, 2020, 2159]
ATTN_NAMES = ('self_attn', 'linear_attn', 'attention', 'attn')


def load_model(name, revision, dtype, random_blocks=False):
    """Try the causal-LM loader first, then the image-text-to-text loader (natively multimodal checkpoints such as
    Qwen3.5), then the bare model. A loader is rejected if any decoder-layer weight is missing from the checkpoint
    (a mismatched class would otherwise silently initialise those weights at random)."""
    kw = dict(revision=revision, dtype=getattr(torch, dtype), device_map='auto', output_loading_info=True)
    errs = []
    for cls in ('AutoModelForCausalLM', 'AutoModelForImageTextToText', 'AutoModel'):
        C = getattr(transformers, cls, None)
        if C is None:
            continue
        for attn in ('sdpa', None):
            try:
                m, info = C.from_pretrained(name, **kw, **({'attn_implementation': attn} if attn else {}))
            except Exception as e:  # try the next loader
                errs.append(f'{cls}/{attn}: {e.__class__.__name__}: {str(e)[:160]}'); continue
            missing = [k for k in info.get('missing_keys', []) if '.layers.' in k or 'embed' in k]
            if missing:
                errs.append(f'{cls}/{attn}: {len(missing)} missing layer/embedding weights, e.g. {missing[:3]}')
                del m; continue
            return m, cls, attn or 'default', dict(missing=list(info.get('missing_keys', []))[:20], unexpected=len(info.get('unexpected_keys', [])))
    raise SystemExit('could not load model:\n' + '\n'.join(errs))


def parts(model):
    if hasattr(model, 'model') and hasattr(model.model, 'layers') and hasattr(model.model, 'norm'):
        m = model.model
        return model.get_input_embeddings(), m.layers, m.norm
    if hasattr(model, 'gpt_neox'):
        m = model.gpt_neox
        return m.embed_in, m.layers, m.final_layer_norm
    # generic: the longest ModuleList of decoder layers; its parent must hold the final norm
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 2 and type(mod[0]).__name__.endswith('Layer'):
            if best is None or len(mod) > len(best[1]):
                best = (name, mod)
    if best is None:
        raise SystemExit('no decoder layer list found: run --inspect and extend parts()')
    parent = model.get_submodule(best[0].rsplit('.', 1)[0]) if '.' in best[0] else model
    for nn_ in ('norm', 'final_layer_norm', 'ln_f', 'final_norm'):
        if hasattr(parent, nn_):
            return model.get_input_embeddings(), best[1], getattr(parent, nn_)
    raise SystemExit(f'final norm not found next to {best[0]}: run --inspect and extend parts()')


def attn_module(layer):
    for n in ATTN_NAMES:
        if hasattr(layer, n):
            return n, getattr(layer, n)
    return None, None


def parse_blocks(spec, L):
    if not spec:
        return []
    out = set()
    for part in spec.split(','):
        lo, _, hi = part.partition('-')
        lo = int(lo); hi = int(hi) if hi else lo
        out.update(range(lo, hi + 1))
    bad = [b for b in out if not 1 <= b <= L]
    if bad:
        raise SystemExit(f'block index out of range 1..{L}: {bad}')
    return sorted(out)


def randomize_blocks(blocks, seed, std=None, faithful=False):
    """Unchanged from v5 (same random stream in all modes)."""
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
    pos = torch.arange(SKIP_FIRST, h.shape[0] - 1, device=h.device)[keep]
    return x, x[keep], int((~keep).sum()), pos


def e_uncentred(x):
    n = x.shape[0]; n2 = (x * x).sum(1); s = x.sum(0)
    return float(((s @ s) - n2.sum()) / ((n - 1) * n2.sum()))


def pr_centred(x):
    xc = x - x.mean(0)
    C = xc.T @ xc / x.shape[0]
    return float(torch.trace(C) ** 2 / (C * C).sum())


def pass1_stats(h):
    xall, x, n_hi, _ = kept(h)
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


def window_stats(xw, mubar):
    n = xw.shape[0]
    if n < 32:
        return None
    mu = xw.mean(0); xc = xw - mu; C = xc.T @ xc / n
    trC = float(torch.trace(C)); trC2 = float((C * C).sum())
    delta = mu - mubar; v = xc @ delta
    r = ref_from(delta, float(v @ v) / n, n, trC, trC2); r['n'] = n
    return r


def build_inputs(a, tok):
    """Returns ids_all (list of [1, T] LongTensors), meta, skipped."""
    if a.legacy_zip:
        with zipfile.ZipFile(a.legacy_zip) as z:
            rows = json.loads(z.read('dimension/data/input_texts.json'))
        docs = [dict(id=str(i), source='legacy', text=rows[i][:131072]) for i in LEGACY_ROWS[:a.n_texts or 5]]
    else:
        docs = [json.loads(l) for l in open(a.corpus_jsonl, encoding='utf-8') if l.strip()]
        if a.sources:
            docs = [d for d in docs if d['source'] in a.sources]
        if a.n_texts:
            docs = docs[:a.n_texts]
    ids_all, meta, skipped = [], [], []
    if not a.control:
        for d in docs:
            ids = tok(d['text'], return_tensors='pt', truncation=True, max_length=a.tokens).input_ids
            need = max(a.tokens, a.min_doc_tokens or 0)
            if a.min_doc_tokens and len(tok(d['text']).input_ids) < a.min_doc_tokens or ids.shape[1] < a.tokens:
                skipped.append(dict(id=d['id'], tokens=int(ids.shape[1]), required=need)); continue
            ids_all.append(ids); meta.append(dict(id=d['id'], source=d['source'], tokens=int(ids.shape[1])))
        return ids_all, meta, skipped
    # controls: content tokens without special tokens; a BOS is prepended iff the tokenizer adds one by default
    with_sp = tok('a').input_ids; no_sp = tok('a', add_special_tokens=False).input_ids
    bos = with_sp[:len(with_sp) - len(no_sp)] if len(with_sp) > len(no_sp) and with_sp[-len(no_sp):] == no_sp else []
    T = a.tokens - len(bos)
    content = {}
    for d in docs:
        c = tok(d['text'], add_special_tokens=False).input_ids
        if len(c) < T:
            skipped.append(dict(id=d['id'], tokens=len(c), required=T)); continue
        content[d['id']] = (d['source'], c[:T])
    if a.control == 'shuffle':
        for i, (did, (s, c)) in enumerate(content.items()):
            g = torch.Generator().manual_seed(0 + i)
            perm = torch.randperm(T, generator=g)
            seq = torch.tensor(bos + [c[p] for p in perm.tolist()])[None]
            ids_all.append(seq); meta.append(dict(id=did + '-shuffle', source=s, tokens=a.tokens))
    elif a.control == 'mix':
        ch = T // 16  # content length per sequence is 16*ch (<= tokens - BOS)
        by_src = {}
        for did, (s, c) in content.items():
            by_src.setdefault(s, []).append((did, c))
        for s, lst in by_src.items():
            m = len(lst)
            if m < 16:
                raise SystemExit(f'--control mix needs >= 16 documents per source; {s} has {m}')
            for j in range(m):
                seq, src_ids = list(bos), []
                for c_ in range(16):
                    did, c = lst[(j + c_) % m]
                    seq += c[c_ * ch:(c_ + 1) * ch]; src_ids.append(did)
                ids_all.append(torch.tensor(seq)[None]); meta.append(dict(id=f'mix-{s}-{j:02d}', source=s, tokens=len(seq), chunks=src_ids))
    return ids_all, meta, skipped


def inspect(model, cfg, loader, attn_impl):
    emb, blocks, norm = parts(model)
    tc = getattr(cfg, 'text_config', cfg)
    kinds = {}
    for i, b in enumerate(blocks):
        n, m = attn_module(b)
        key = f'{type(b).__name__}:{n}:{type(m).__name__ if m is not None else None}:mlp={type(getattr(b, "mlp", None)).__name__}:post_ln={hasattr(b, "post_attention_layernorm")}'
        kinds.setdefault(key, []).append(i + 1)
    info = dict(model_class=type(model).__name__, loader=loader, attn_implementation=attn_impl, model_type=getattr(cfg, 'model_type', None),
                text_model_type=getattr(tc, 'model_type', None), hidden_size=getattr(tc, 'hidden_size', None), n_blocks=len(blocks),
                max_position_embeddings=getattr(tc, 'max_position_embeddings', None), sliding_window=getattr(tc, 'sliding_window', None),
                use_sliding_window=getattr(tc, 'use_sliding_window', None), layer_types=getattr(tc, 'layer_types', None),
                initializer_range=getattr(tc, 'initializer_range', None), residual_multiplier=getattr(tc, 'residual_multiplier', None),
                embedding=type(emb).__name__, final_norm=type(norm).__name__, block_kinds={k: f'{v[0]}..{v[-1]} ({len(v)})' for k, v in kinds.items()})
    print(json.dumps(info, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--revision', default=None)
    ap.add_argument('--corpus-jsonl', default=None)
    ap.add_argument('--legacy-zip', default=None)
    ap.add_argument('--n-texts', type=int, default=None)
    ap.add_argument('--sources', nargs='*', default=None)
    ap.add_argument('--min-doc-tokens', type=int, default=None)
    ap.add_argument('--tokens', type=int, default=16384)
    ap.add_argument('--allow-overflow', action='store_true')
    ap.add_argument('--random-blocks', action='store_true')
    ap.add_argument('--seed', type=int, default=41)
    ap.add_argument('--random-std', type=float, default=None)
    ap.add_argument('--faithful', action='store_true')
    ap.add_argument('--sublayer', action='store_true')
    ap.add_argument('--zero-attn', default=None)
    ap.add_argument('--zero-mlp', default=None)
    ap.add_argument('--local-window', type=int, default=None)
    ap.add_argument('--window-blocks', default=None)
    ap.add_argument('--control', choices=['mix', 'shuffle'], default=None)
    ap.add_argument('--attn-kind', choices=['all', 'linear', 'full'], default='all')
    ap.add_argument('--hf-init', action='store_true')
    ap.add_argument('--inspect', action='store_true')
    ap.add_argument('--no-proj', action='store_true')
    ap.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    if a.faithful and not (a.random_blocks and a.random_std):
        sys.exit('--faithful requires --random-blocks and --random-std')
    if a.hf_init and not a.random_blocks or a.hf_init and (a.faithful or a.random_std):
        sys.exit('--hf-init requires --random-blocks and excludes --faithful/--random-std')
    if (a.local_window is None) != (a.window_blocks is None):
        sys.exit('--local-window and --window-blocks go together')

    cfg = AutoConfig.from_pretrained(a.model, revision=a.revision)
    tc = getattr(cfg, 'text_config', cfg)
    if a.inspect:
        model, loader, attn_impl, linfo = load_model(a.model, a.revision, a.dtype)
        inspect(model, cfg, loader, attn_impl); print('loading_info', json.dumps(linfo)); return
    if not a.out:
        sys.exit('--out is required')
    ctx = getattr(tc, 'max_position_embeddings', None)
    if ctx is not None and a.tokens > ctx and not a.allow_overflow:
        sys.exit(f'--tokens {a.tokens} exceeds max_position_embeddings {ctx}')
    sw = getattr(tc, 'sliding_window', None)
    uses_sw = getattr(tc, 'use_sliding_window', None)
    lt = getattr(tc, 'layer_types', None) or []
    if (uses_sw or (uses_sw is None and sw) or any('sliding' in str(t) for t in lt)) and sw and sw < a.tokens:
        sys.exit('sliding-window attention shorter than --tokens: not permitted')

    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    ids_all, meta, skipped = build_inputs(a, tok)
    N = len(ids_all)
    if N < 3:
        sys.exit(f'only {N} documents qualify')
    print(f'{N} sequences, {len(skipped)} skipped', file=sys.stderr)

    model, loader, attn_impl, linfo = load_model(a.model, a.revision, a.dtype)
    model.eval()
    emb, blocks, norm = parts(model)
    L = len(blocks)
    init_summary = None
    if a.random_blocks and a.hf_init:
        torch.manual_seed(a.seed)
        with torch.no_grad():
            for block in blocks:
                block.apply(model._init_weights)
        init_summary = {}
        for name, p in blocks[0].named_parameters():
            v = p.data.float()
            init_summary.setdefault('matrix' if p.dim() >= 2 else 'vector', []).append(
                dict(name=name, shape=list(p.shape), mean=float(v.mean()), std=float(v.std()) if v.numel() > 1 else 0.0))
    elif a.random_blocks:
        init_summary = randomize_blocks(blocks, a.seed, a.random_std, a.faithful)
    dev = model.get_input_embeddings().weight.device

    # ---- interventions
    iv = dict(zero_attn=parse_blocks(a.zero_attn, L), attn_kind=a.attn_kind, zero_mlp=parse_blocks(a.zero_mlp, L),
              local_window=a.local_window, window_blocks=parse_blocks(a.window_blocks, L), skipped_modules=[])
    hs = []
    def zero_out(mod, inp, out):
        if isinstance(out, (tuple, list)):
            return (torch.zeros_like(out[0]),) + tuple(out[1:])
        return torch.zeros_like(out)
    for b in iv['zero_attn']:
        n, m = attn_module(blocks[b - 1])
        if m is None:
            sys.exit(f'block {b}: no attention module')
        if a.attn_kind == 'linear' and n != 'linear_attn' or a.attn_kind == 'full' and n not in ('self_attn', 'attention'):
            iv['skipped_modules'].append(dict(block=b, module=n, reason=f'attn_kind={a.attn_kind}')); continue
        hs.append(m.register_forward_hook(zero_out))
    for b in iv['zero_mlp']:
        m = getattr(blocks[b - 1], 'mlp', None)
        if m is None:
            sys.exit(f'block {b}: no mlp module')
        hs.append(m.register_forward_hook(zero_out))
    MASK = {}
    def band_mask(T, device):
        if (T, device) not in MASK:
            i = torch.arange(T, device=device)
            d = i[:, None] - i[None, :]
            MASK[(T, device)] = ((d >= 0) & (d < a.local_window))[None, None]
        return MASK[(T, device)]
    def window_pre(mod, args, kwargs):
        hsx = kwargs.get('hidden_states', args[0] if args else None)
        T = hsx.shape[1]
        m = band_mask(T, hsx.device)
        if 'attention_mask' in kwargs or len(args) < 3:
            kwargs['attention_mask'] = m
        else:
            args = tuple(args[:2]) + (m,) + tuple(args[3:])
        return args, kwargs
    for b in iv['window_blocks']:
        n, m = attn_module(blocks[b - 1])
        if n != 'self_attn' and n != 'attention':
            iv['skipped_modules'].append(dict(block=b, module=n)); continue
        hs.append(m.register_forward_pre_hook(window_pre, with_kwargs=True))

    # ---- state hooks
    keys = ['0']
    for i in range(1, L + 1):
        if a.sublayer:
            keys.append(f'{i}.a')
        keys.append(str(i))
    keys.append('norm')
    state = dict(fn=None, doc=None, mid={})
    mult = float(getattr(tc, 'residual_multiplier', 1.0) or 1.0)
    sub_check = {}; sub_check_first = {}
    def hook(key):
        def f(mod, inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            h = h[0].detach()
            if a.sublayer and key not in ('0', 'norm') and state['doc'] == 0 and key in state['mid'] and state['pass'] == 1:
                b = blocks[int(key) - 1]; mid = state['mid'][key]
                idx = kept(h)[3][:256]          # the positions the measurement keeps (no sink / high-norm tokens)
                state['checking'] = True
                try:
                    with torch.no_grad():
                        pred = b.mlp(b.post_attention_layernorm(mid[None, idx]))[0] * mult
                        pred0 = b.mlp(b.post_attention_layernorm(mid[None, :256]))[0] * mult
                finally:
                    state['checking'] = False
                r = (h[idx] - mid[idx]).float(); pred = pred.float()
                sub_check[key] = float((r - pred).norm() / r.norm().clamp_min(1e-30))
                r0 = (h[:256] - mid[:256]).float()
                sub_check_first[key] = float((r0 - pred0.float()).norm() / r0.norm().clamp_min(1e-30))
            state['fn'](key, h)
        return f
    def mid_hook(key):
        def f(mod, args):
            if state.get('checking'):
                return
            h = args[0][0].detach()
            state['mid'][key.split('.')[0]] = h
            state['fn'](key, h)
        return f
    hs += [emb.register_forward_hook(hook('0'))] + [b.register_forward_hook(hook(str(i + 1))) for i, b in enumerate(blocks)]
    hs.append(norm.register_forward_hook(hook('norm')))
    if a.sublayer:
        for i, b in enumerate(blocks):
            if not hasattr(b, 'post_attention_layernorm') or not hasattr(b, 'mlp'):
                sys.exit(f'--sublayer: block {i + 1} has no post_attention_layernorm/mlp')
            hs.append(b.post_attention_layernorm.register_forward_pre_hook(mid_hook(f'{i + 1}.a')))

    # ---- pass 1
    P1 = {k: [None] * N for k in keys}; MU = {k: [None] * N for k in keys}; digest1, nonfinite = [], []
    def f1(key, h):
        j = state['doc']
        if not torch.isfinite(h).all():
            nonfinite.append([meta[j]['id'], key]); return
        P1[key][j], MU[key][j] = pass1_stats(h)
        if key == '0':
            digest1.append(hashlib.sha256(h.float().cpu().numpy().tobytes()).hexdigest()[:16])
    state['fn'] = f1; state['pass'] = 1
    t0 = time.time()
    with torch.no_grad():
        for j, ids in enumerate(ids_all):
            state['doc'] = j; state['mid'] = {}; model(input_ids=ids.to(dev), use_cache=False)
            print(f'pass1 {j + 1}/{N} {time.time() - t0:.0f}s', file=sys.stderr)
    if nonfinite:
        sys.exit(f'non-finite activations: {nonfinite[:5]}')

    src = [m['source'] for m in meta]
    REF, BASIS = {}, {}
    for k in keys:
        Mu = torch.stack(MU[k])
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
        if not a.no_proj:
            U, _ = torch.linalg.qr(Mu.T); BASIS[k] = U

    # ---- pass 2
    P2 = {k: [None] * N for k in keys}; POS = {k: [None] * N for k in keys}; PROJ_B = {k: [None] * N for k in keys}; digest2, check = [], []
    def f2(key, h):
        j = state['doc']
        _, x, _, pos = kept(h)
        n = x.shape[0]; mu = MU[key][j]; xc = x - mu
        p1 = P1[key][j]; out = {}
        for name, mubar in REF[key][j].items():
            delta = mu - mubar
            v = xc @ delta
            out[name] = ref_from(delta, float(v @ v) / n, n, p1['trC'], p1['trC2'])
            if j == 0:
                y = x - mubar
                e_dir = e_uncentred(y); M = y.T @ y / n; pr_dir = float(torch.trace(M) ** 2 / (M * M).sum())
                check.append(dict(state=key, ref=name, E_rel=abs(e_dir - out[name]['E']) / max(abs(e_dir), 1e-30),
                                  PR_rel=abs(pr_dir - out[name]['PR']) / pr_dir))
        P2[key][j] = out
        pw = {}
        for lo, hi in POS_WINDOWS:
            sel = (pos >= lo) & (pos < hi if hi is not None else torch.ones_like(pos, dtype=torch.bool))
            r = window_stats(x[sel], REF[key][j]['all'])
            if r is not None:
                pw[f'{lo}-{hi if hi is not None else "end"}'] = r
        POS[key][j] = pw
        if not a.no_proj:
            Pj = xc @ BASIS[key]; PROJ_B[key][j] = (Pj.T @ Pj / n).float().cpu()
        if key == '0':
            digest2.append(hashlib.sha256(h.float().cpu().numpy().tobytes()).hexdigest()[:16])
    state['fn'] = f2; state['pass'] = 2
    with torch.no_grad():
        for j, ids in enumerate(ids_all):
            state['doc'] = j; state['mid'] = {}; model(input_ids=ids.to(dev), use_cache=False)
            print(f'pass2 {j + 1}/{N} {time.time() - t0:.0f}s', file=sys.stderr)
    for h in hs:
        h.remove()

    layers = []
    for k in keys:
        per = [dict(P1[k][j], ref=P2[k][j], pos=POS[k][j]) for j in range(N)]
        kind = 'embedding' if k == '0' else ('final_norm' if k == 'norm' else ('post_attention' if k.endswith('.a') else 'block'))
        layers.append(dict(state=k, kind=kind, per_doc=per))
    mode = ('trained' if not a.random_blocks else 'random_blocks_hfinit' if a.hf_init else 'random_blocks_unit_gaussian' if a.random_std is None else
            f'random_blocks_faithful_std{a.random_std:g}' if a.faithful else f'random_blocks_gaussian_std{a.random_std:g}')
    out = dict(schema='dim_v6/1', model=a.model, revision=a.revision, resolved_revision=getattr(model.config, '_commit_hash', None),
               loader=loader, attn_implementation=attn_impl, loading_info=linfo, mode=mode, random_std=a.random_std, faithful=a.faithful, init_summary=init_summary,
               seed=a.seed if a.random_blocks else None, dtype=a.dtype, n_blocks=L, hidden=int(getattr(tc, 'hidden_size')),
               max_position_embeddings=ctx, tokens=a.tokens, skip_first=SKIP_FIRST, hi_norm=HI_NORM, pos_windows=POS_WINDOWS,
               control=a.control, intervention=iv, sublayer=a.sublayer, sublayer_check=sub_check,
               sublayer_check_max=max(sub_check.values()) if sub_check else None,
               sublayer_check_first256=sub_check_first,
               sublayer_check_first256_max=max(sub_check_first.values()) if sub_check_first else None,
               corpus=a.corpus_jsonl or a.legacy_zip, docs=meta, skipped=skipped,
               digest_pass1=digest1, digest_pass2=digest2, deterministic=digest1 == digest2,
               self_check=check, self_check_max=dict(E=max(c['E_rel'] for c in check), PR=max(c['PR_rel'] for c in check)),
               layers=layers, seconds=round(time.time() - t0, 1))
    json.dump(out, open(a.out + '.json', 'w'))
    if not a.no_proj:
        torch.save(dict(keys=keys, docs=meta, A={k: (BASIS[k].T @ torch.stack(MU[k]).T).T.cpu() for k in keys},
                        B={k: torch.stack(PROJ_B[k]) for k in keys}), a.out + '.proj.pt')
    print('wrote', a.out, 'deterministic', out['deterministic'], 'self_check', out['self_check_max'],
          'sublayer_check_max', out['sublayer_check_max'], file=sys.stderr)


if __name__ == '__main__':
    main()
