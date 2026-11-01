#!/usr/bin/env python3
"""
Multi-seed pruning grid for mamba-130m  (Part 1 follow-up: the "full experiment").

One run = (seed, scoring method, sparsity):  score -> prune (per-layer top-k, identical to the H4-H15 code)
          -> evaluate -> fine-tune -> evaluate.  Each run is written to its own JSON file, so the grid is
          resumable and can be split across Colab accounts (e.g. one seed per account) and merged afterwards.

Scoring methods (D = 1536 channels per layer, in_proj output = [content | gate]):
  Random                       uniform random
  LRP-CO / LRP-GI              MambaLRP relevance at the in_proj output: content half / content+gate halves  (H10, H15)
  LRP-Out                      MambaLRP relevance at the out_proj input, i.e. AFTER the gate        (NEW)
  GxI-CO / GxI-GI              plain gradient x input at the in_proj output (same sites as LRP, no LRP rules)  (NEW as GI)
  GxA-Out                      plain gradient x activation at the out_proj input (same site as LRP-Out)        (NEW)
  GradMag-CO / GradMag-GI      Taylor |w * dL/dw| on the in_proj rows: content half / content+gate halves   (H15)
  Act-CO / Act-GI              mean |in_proj output| content half / both halves  (H12, H15; pre-gate)
  Act-Out                      mean |out_proj input|, i.e. post-gate activation                               (NEW)
  Dense                        control: the unpruned model given the same fine-tuning

Protocols
  main    WikiText-103 train for calibration + fine-tuning, FULL WikiText test set for evaluation, longer
          fine-tuning with warm-up + cosine schedule, calibration sample and fine-tuning data order depend on the seed.
  legacy  exactly the H15 setting (WikiText-2, 256 first texts, 30 constant-lr steps, 40x256 eval tokens).
          Use it once to confirm this runner reproduces the numbers in the report (e.g. LRP-GI @30% ~ 58.7).

Note: the WikiText-103 and WikiText-2 test sets are the same text, so "WikiText-103" only changes the
calibration / fine-tuning data (much larger than WikiText-2's), not the evaluation text.
"""
import argparse, copy, json, math, os, random, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ALL_SCORES = ["LRP-CO", "LRP-GI", "LRP-Out", "Act-CO", "Act-GI", "Act-Out",           # pass A (MambaLRP-wrapped model)
              "GradMag-CO", "GradMag-GI", "GxI-CO", "GxI-GI", "GxA-Out"]             # pass B (stock HF model)
PASS_A = {"LRP-CO", "LRP-GI", "LRP-Out", "Act-CO", "Act-GI", "Act-Out"}
PASS_B = {"GradMag-CO", "GradMag-GI", "GxI-CO", "GxI-GI", "GxA-Out"}
HYBRIDS = ("Hyb-L0GI", "Hyb-L0CO")
MAGS = ("Mag-CO", "Mag-GI")      # weight-magnitude baselines, computed from the weights alone (no data, no scoring stage)
DEFAULT_METHODS = ["Random", "LRP-CO", "LRP-GI", "LRP-Out", "Act-GI", "Act-Out",
                   "GradMag-GI", "GxI-GI", "GxA-Out", "Dense"]


# ══════════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════════
def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def atomic_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def lm_logits(model, ids, hf=True):
    out = model(input_ids=ids, use_cache=False) if hf else model(input_ids=ids)
    return out.logits if hasattr(out, "logits") else out


def token_loss(logits, ids, mask=None):
    """Sum of next-token cross-entropy over real (unmasked) positions, as in the H-scripts."""
    ce = F.cross_entropy(logits[:, :-1].contiguous().transpose(1, 2), ids[:, 1:].contiguous(), reduction="none")
    if mask is not None:
        ce = ce * mask[:, 1:].float()
    return ce.sum()


# ══════════════════════════════════════════════════════════════════════════════
# data
# ══════════════════════════════════════════════════════════════════════════════
class Data:
    """Holds everything that depends on the protocol (not on the seed)."""
    pass


def load_data(args, tokenizer):
    from datasets import load_dataset
    D = Data()
    if args.protocol == "legacy":
        cfg = "wikitext-2-raw-v1"
        test = load_dataset("Salesforce/wikitext", cfg, split="test")
        train = load_dataset("Salesforce/wikitext", cfg, split="train")
        join = lambda ds: " ".join(t for t in ds["text"] if len(t.strip()) > 20)
        enc = lambda s: tokenizer(s, return_tensors="pt")["input_ids"][0]
        ev = enc(join(test))[: 256 * 40 + 1]
        D.eval_chunks = torch.stack([ev[i:i + 256] for i in range(0, len(ev) - 256, 256)][:40])
        D.train_tokens = enc(join(train))[: 128 * 200]
        D.calib_pool = [t for t in train["text"] if len(t.strip()) > 80][:256]
    else:
        test = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
        ev = tokenizer(" ".join(t for t in test["text"] if len(t.strip()) > 20), return_tensors="pt")["input_ids"][0]
        if args.eval_max_tokens:
            ev = ev[: args.eval_max_tokens + 1]
        n = (len(ev) // 256)
        D.eval_chunks = ev[: n * 256].view(n, 256)
        train = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=f"train[:{args.train_rows}]")
        lines = [t for t in train["text"] if len(t.strip()) > 20]
        toks = []
        for i in range(0, len(lines), 1000):                               # piecewise: tokenising 15 MB at once is slow
            toks.append(tokenizer(" ".join(lines[i:i + 1000]), return_tensors="pt")["input_ids"][0])
        D.train_tokens = torch.cat(toks)
        D.calib_pool = [t for t in train["text"] if len(t.strip()) > 80][: args.calib_pool]
    log(f"data[{args.protocol}]: eval chunks {tuple(D.eval_chunks.shape)} | train tokens {len(D.train_tokens):,} "
        f"| calibration pool {len(D.calib_pool):,} texts")
    return D


def make_tiny(args):
    """Offline plumbing test: random tiny Mamba + synthetic tokens (no downloads)."""
    from transformers import MambaConfig, MambaForCausalLM
    torch.manual_seed(0)
    cfg = MambaConfig(vocab_size=97, hidden_size=32, state_size=4, num_hidden_layers=3, expand=2, conv_kernel=4,
                      time_step_rank=4, use_cache=False)
    model = MambaForCausalLM(cfg).eval()
    g = torch.Generator().manual_seed(123)
    D = Data()
    D.eval_chunks = torch.randint(0, 97, (6, 64), generator=g)
    D.train_tokens = torch.randint(0, 97, (4000,), generator=g)
    D.calib_pool = None
    D.tiny_calib = [(torch.randint(0, 97, (4, 24), generator=g), torch.ones(4, 24, dtype=torch.long)) for _ in range(6)]
    return model, None, D


# ══════════════════════════════════════════════════════════════════════════════
# scoring
# ══════════════════════════════════════════════════════════════════════════════
def calib_batches(args, D, tokenizer, seed, device):
    """List of (input_ids, attention_mask) batches of 4 -- the same batching as the H-scripts."""
    if getattr(D, "tiny_calib", None) is not None:
        return [(i.to(device), m.to(device)) for i, m in D.tiny_calib]
    if args.protocol == "legacy":
        texts = D.calib_pool[: args.calib_n]
    else:
        texts = random.Random(1000 + seed).sample(D.calib_pool, args.calib_n)
    out = []
    for s in range(0, len(texts), 4):
        e = tokenizer(texts[s:s + 4], return_tensors="pt", padding=True, truncation=True, max_length=args.calib_len)
        out.append((e["input_ids"].to(device), e["attention_mask"].to(device)))
    return out


def scores_pass_A(base, batches, args, device, need):
    """MambaLRP-wrapped model: LRP-CO/GI/Out and activation scores in one forward/backward sweep."""
    sys.path.insert(0, args.mambalrp_path)
    from mamba_lrp.model.mamba_huggingface import ModifiedMambaForCausalLM
    wrapped = ModifiedMambaForCausalLM(base, is_fast_forward_available=False).to(device).eval()
    L, Dm = base.config.num_hidden_layers, base.config.intermediate_size
    S = {k: torch.zeros(L, Dm) for k in ("LRP-CO", "LRP-GI", "LRP-Out", "Act-CO", "Act-GI", "Act-Out")}
    cap_in, cap_out, fired = {}, {}, {"in": 0, "out": 0}

    def in_hook(i):
        def h(m, a, out):
            out.retain_grad(); cap_in[i] = out; fired["in"] += 1
        return h

    def out_hook(i):
        def h(m, a):
            a[0].retain_grad(); cap_out[i] = a[0]; fired["out"] += 1
        return h

    handles = []
    for i in range(L):
        mx = wrapped.backbone.layers[i].mixer
        handles.append(mx.in_proj.register_forward_hook(in_hook(i)))
        handles.append(mx.out_proj.register_forward_pre_hook(out_hook(i)))
    try:
        mb = max(1, getattr(args, "score_micro", 4))
        mbatches = [(i_[c:c + mb], m_[c:c + mb]) for i_, m_ in batches for c in range(0, i_.shape[0], mb)]
        for ids, mask in mbatches:
            wrapped.zero_grad(set_to_none=True)
            logits = lm_logits(wrapped, ids, hf=False)
            token_loss(logits, ids, mask).backward()
            m_ = mask.unsqueeze(-1).float()
            for i in range(L):
                h, a = cap_in[i], cap_out[i]
                hv, hg = h.detach(), h.grad
                av, ag = a.detach(), a.grad
                con = lambda x: x[..., :Dm]
                gat = lambda x: x[..., Dm:]
                s_con = (con(hv) * con(hg) * m_).abs().sum((0, 1)).cpu()
                s_gat = (gat(hv) * gat(hg) * m_).abs().sum((0, 1)).cpu()
                S["LRP-CO"][i] += s_con
                S["LRP-GI"][i] += s_con + s_gat
                S["LRP-Out"][i] += (av * ag * m_).abs().sum((0, 1)).cpu()
                a_con = (con(hv) * m_).abs().sum((0, 1)).cpu()
                a_gat = (gat(hv) * m_).abs().sum((0, 1)).cpu()
                S["Act-CO"][i] += a_con
                S["Act-GI"][i] += a_con + a_gat
                S["Act-Out"][i] += (av * m_).abs().sum((0, 1)).cpu()
            cap_in.clear(); cap_out.clear()
    finally:
        for h in handles:
            h.remove()
    assert fired["in"] > 0 and fired["out"] > 0, \
        f"scoring hooks did not fire ({fired}); the model is not using the slow MambaLRP path"
    del wrapped
    if device == "cuda":
        torch.cuda.empty_cache()
    return {k: v for k, v in S.items() if k in need}


def scores_pass_B(base, batches, args, device, need):
    """Stock HF model: weight-gradient Taylor scores, plain gradient x input, plain gradient x activation."""
    model = copy.deepcopy(base).to(device)
    model.eval(); model.zero_grad(set_to_none=True)      # eval(): Mamba has no dropout, and train mode may take a fused path
    for p in model.parameters():
        p.requires_grad_(True)
    L, Dm = model.config.num_hidden_layers, model.config.intermediate_size
    S = {k: torch.zeros(L, Dm) for k in ("GxI-CO", "GxI-GI", "GxA-Out")}
    cap_in, cap_out, fired = {}, {}, {"in": 0, "out": 0}

    def in_hook(i):
        def h(m, a, out):
            out.retain_grad(); cap_in[i] = out; fired["in"] += 1
        return h

    def out_hook(i):
        def h(m, a):
            a[0].retain_grad(); cap_out[i] = a[0]; fired["out"] += 1
        return h

    handles = []
    for i, blk in enumerate(model.backbone.layers):
        handles.append(blk.mixer.in_proj.register_forward_hook(in_hook(i)))
        handles.append(blk.mixer.out_proj.register_forward_pre_hook(out_hook(i)))
    try:
        mb = max(1, getattr(args, "score_micro", 4))
        for ids_full, mask_full in batches:                 # no zero_grad between batches: weight grads accumulate
          for c0 in range(0, ids_full.shape[0], mb):        # loss is a SUM over tokens, so micro-batches give identical gradients
            ids, mask = ids_full[c0:c0 + mb], mask_full[c0:c0 + mb]
            token_loss(lm_logits(model, ids), ids, mask).backward()
            m_ = mask.unsqueeze(-1).float()
            for i in range(L):
                h, a = cap_in[i], cap_out[i]
                hv, hg, av, ag = h.detach(), h.grad, a.detach(), a.grad
                s_con = (hv[..., :Dm] * hg[..., :Dm] * m_).abs().sum((0, 1)).cpu()
                s_gat = (hv[..., Dm:] * hg[..., Dm:] * m_).abs().sum((0, 1)).cpu()
                S["GxI-CO"][i] += s_con
                S["GxI-GI"][i] += s_con + s_gat
                S["GxA-Out"][i] += (av * ag * m_).abs().sum((0, 1)).cpu()
            cap_in.clear(); cap_out.clear()
    finally:
        for h in handles:
            h.remove()
    assert fired["in"] > 0 and fired["out"] > 0, f"scoring hooks did not fire ({fired}); the model took a fused path"
    gm_co, gm_gi = torch.zeros(L, Dm), torch.zeros(L, Dm)
    for i, blk in enumerate(model.backbone.layers):
        w, g = blk.mixer.in_proj.weight.data, blk.mixer.in_proj.weight.grad
        con = (w[:Dm] * g[:Dm]).abs().mean(1).cpu()
        gat = (w[Dm:] * g[Dm:]).abs().mean(1).cpu()
        gm_co[i], gm_gi[i] = con, con + gat
    S["GradMag-CO"], S["GradMag-GI"] = gm_co, gm_gi
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {k: v for k, v in S.items() if k in need}


def get_scores(args, base, D, tokenizer, seed, methods, device, allow_compute=True):
    """All requested score tensors for this seed; cached on disk (computed once per seed, shared by all runs)."""
    path = os.path.join(args.out, "scores", f"{args.protocol}_seed{seed}.pt")
    cache = torch.load(path) if os.path.exists(path) else {}
    need = {m for m in methods if m in ALL_SCORES}
    if any(m in HYBRIDS for m in methods):
        need |= {"LRP-CO", "LRP-GI"}                       # the layer-0 hybrids are built from these two cached scores
    missing = need - set(cache)
    if missing and not allow_compute:
        sys.exit(f"stage 'run' needs cached scores for seed {seed} but {sorted(missing)} are missing in {path}. "
                 f"Run `--stage score` for this seed first (in the slow-path runtime), and copy/keep the scores folder.")
    if missing:
        batches = calib_batches(args, D, tokenizer, seed, device)
        t0 = time.time()
        if missing & PASS_A:
            log(f"seed {seed}: scoring pass A (MambaLRP-wrapped) for {sorted(missing & PASS_A)}")
            cache.update(scores_pass_A(base, batches, args, device, PASS_A))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(cache, path + ".tmp"); os.replace(path + ".tmp", path)      # keep pass A even if pass B fails
        if missing & PASS_B:
            log(f"seed {seed}: scoring pass B (stock model) for {sorted(missing & PASS_B)}")
            cache.update(scores_pass_B(base, batches, args, device, PASS_B))
        log(f"seed {seed}: scores done in {time.time() - t0:.0f}s")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(cache, path + ".tmp"); os.replace(path + ".tmp", path)
    L, Dm = base.config.num_hidden_layers, base.config.intermediate_size
    g = torch.Generator().manual_seed(42 if args.protocol == "legacy" else 10_000 + seed)
    cache["Random"] = torch.rand(L, Dm, generator=g)
    inp = torch.stack([blk.mixer.in_proj.weight.data.float().abs().cpu() for blk in base.backbone.layers])   # L x 2D x d_model
    cache["Mag-CO"] = inp[:, :Dm].mean(dim=2)                  # L1-mean magnitude of the content rows (the H3 definition)
    cache["Mag-GI"] = 0.5 * (inp[:, :Dm].mean(dim=2) + inp[:, Dm:].mean(dim=2))   # content + gate rows
    if "LRP-CO" in cache and "LRP-GI" in cache:               # derived causal-test scores (per-layer top-k only uses within-layer ranks)
        h = cache["LRP-CO"].clone(); h[0] = cache["LRP-GI"][0]; cache["Hyb-L0GI"] = h     # content-only everywhere, gate-aware in layer 0
        h = cache["LRP-GI"].clone(); h[0] = cache["LRP-CO"][0]; cache["Hyb-L0CO"] = h     # gate-aware everywhere, content-only in layer 0
    return cache


# ══════════════════════════════════════════════════════════════════════════════
# pruning  (copied verbatim from the H-scripts: per-layer top-k over all components that share the channel axis)
# ══════════════════════════════════════════════════════════════════════════════
def slice_linear(l, ko=None, ki=None):
    of = ko.numel() if ko is not None else l.out_features
    inf = ki.numel() if ki is not None else l.in_features
    nl = nn.Linear(inf, of, bias=l.bias is not None, device=l.weight.device, dtype=l.weight.dtype)
    w = l.weight.data
    if ko is not None: w = w.index_select(0, ko.to(w.device))
    if ki is not None: w = w.index_select(1, ki.to(w.device))
    nl.weight.data.copy_(w)
    if l.bias is not None:
        b = l.bias.data
        if ko is not None: b = b.index_select(0, ko.to(b.device))
        nl.bias.data.copy_(b)
    return nl


def slice_conv(c, keep):
    keep = keep.to(c.weight.device); n = keep.numel()
    nc = nn.Conv1d(n, n, kernel_size=c.kernel_size, groups=n, padding=c.padding, bias=c.bias is not None,
                   device=c.weight.device, dtype=c.weight.dtype)
    nc.weight.data.copy_(c.weight.data.index_select(0, keep))
    if c.bias is not None:
        nc.bias.data.copy_(c.bias.data.index_select(0, keep))
    return nc


def kept_indices(scores, keep_ratio, D, multiple=1):
    n_keep = n_keep_for(D, keep_ratio, multiple)
    return [torch.topk(scores[i], k=n_keep, largest=True).indices.sort().values for i in range(scores.shape[0])]


def n_keep_for(D, keep_ratio, multiple=1):
    n = int(round(D * keep_ratio))
    return n if multiple <= 1 else max(multiple, int(round(n / multiple)) * multiple)


def prune_model(src, scores, keep_ratio, device, multiple=1):
    m = copy.deepcopy(src).to(device)
    D = src.config.intermediate_size
    n_keep = n_keep_for(D, keep_ratio, multiple)
    for i, block in enumerate(m.backbone.layers):
        mx = block.mixer
        keep = torch.topk(scores[i], k=n_keep, largest=True).indices.sort().values.to(device)
        kip = torch.cat([keep, keep + D])
        mx.in_proj = slice_linear(mx.in_proj, ko=kip)
        mx.conv1d = slice_conv(mx.conv1d, keep)
        mx.x_proj = slice_linear(mx.x_proj, ki=keep)
        mx.dt_proj = slice_linear(mx.dt_proj, ko=keep)
        mx.out_proj = slice_linear(mx.out_proj, ki=keep)
        mx.A_log = nn.Parameter(mx.A_log.data.index_select(0, keep))
        mx.D = nn.Parameter(mx.D.data.index_select(0, keep))
        mx.intermediate_size = n_keep
    m.config.intermediate_size = n_keep
    return m.eval()


# ══════════════════════════════════════════════════════════════════════════════
# evaluation and fine-tuning
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def perplexity(model, chunks, device, bs=8):
    """exp(mean next-token NLL) over non-overlapping 256-token chunks (same convention as compute_perplexity)."""
    model.eval()
    nll, n = 0.0, 0
    for i in range(0, len(chunks), bs):
        ids = chunks[i:i + bs].to(device)
        nll += token_loss(lm_logits(model, ids), ids).item()
        n += ids.shape[0] * (ids.shape[1] - 1)
    return float(math.exp(nll / n))


def fine_tune(model, D, args, seed, device, eval_cb=None):
    """Legacy: constant lr, sequential 4x128 windows. Main: warm-up + cosine, random windows (seeded, same for all methods)."""
    legacy = args.protocol == "legacy"
    steps, bs, seq, lr = (30, 4, 128, 2e-4) if legacy else (args.ft_steps, args.ft_batch, args.ft_seq, args.ft_lr)
    micro = bs if legacy else max(1, min(bs, args.ft_micro))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    toks = D.train_tokens
    if legacy:
        chunks = [toks[i:i + seq] for i in range(0, len(toks) - seq, seq) if len(toks[i:i + seq]) == seq]
    else:
        g = torch.Generator().manual_seed(7919 * seed + 13)
        starts = torch.randint(0, len(toks) - seq - 1, (steps, bs), generator=g)
    losses, mids = [], {}
    for step in range(steps):
        if legacy:
            s = (step * bs) % max(1, len(chunks) - bs)
            ids = torch.stack(chunks[s:s + bs])
        else:
            ids = torch.stack([toks[j:j + seq] for j in starts[step].tolist()])
            warm = min(1.0, (step + 1) / args.ft_warmup)
            cos = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps))
            for pg in opt.param_groups:
                pg["lr"] = lr * warm * cos
        ids = ids.to(device)
        opt.zero_grad(set_to_none=True)
        denom, total = ids.shape[0] * (ids.shape[1] - 1), 0.0
        for part in ids.split(micro):                          # gradient accumulation: same maths, T4-safe memory
            loss = token_loss(lm_logits(model, part), part) / denom
            loss.backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(total)
        if eval_cb is not None and (step + 1) in args.eval_at_list and (step + 1) != steps:
            mids[step + 1] = eval_cb(model); model.train()
    model.eval()
    k = max(1, steps // 20)
    curve = [float(np.mean(losses[i:i + k])) for i in range(0, steps, k)]
    return model, curve, mids


# ══════════════════════════════════════════════════════════════════════════════
# grid
# ══════════════════════════════════════════════════════════════════════════════
def run_path(args, seed, method, sp):
    return os.path.join(args.out, "runs", args.protocol, f"seed{seed}_{method}_sp{int(round(sp * 100)):02d}.json")


def one_run(args, base, D, scores, seed, method, sp, device):
    t0 = time.time()
    evalf = lambda m: perplexity(m, D.eval_chunks, device)
    if method == "Dense":
        model = copy.deepcopy(base).to(device)
    else:
        model = prune_model(base, scores[method], 1.0 - sp, device, getattr(args, 'keep_multiple', 1))
    n_params = sum(p.numel() for p in model.parameters())
    pre = evalf(model)
    torch.manual_seed(seed)
    model, curve, mids = fine_tune(model, D, args, seed, device, eval_cb=evalf)
    post = evalf(model)
    rec = dict(protocol=args.protocol, seed=seed, method=method, sparsity=0.0 if method == "Dense" else sp,
               ppl_pre=pre, ppl_post=post, ppl_mid={str(k): v for k, v in mids.items()}, train_loss=curve,
               n_params=n_params, seconds=round(time.time() - t0, 1),
               ft=dict(steps=30 if args.protocol == "legacy" else args.ft_steps,
                       batch=4 if args.protocol == "legacy" else args.ft_batch,
                       seq=128 if args.protocol == "legacy" else args.ft_seq,
                       lr=2e-4 if args.protocol == "legacy" else args.ft_lr),
               eval_tokens=int(D.eval_chunks.numel()), calib_n=args.calib_n,
               torch=torch.__version__, time=time.strftime("%Y-%m-%d %H:%M:%S"))
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="grid_out")
    ap.add_argument("--protocol", choices=["main", "legacy"], default="main")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    ap.add_argument("--sparsities", default="0.3,0.5,0.7")
    ap.add_argument("--model", default="state-spaces/mamba-130m-hf")
    ap.add_argument("--score-micro", type=int, default=4, help="micro-batch size inside the scoring passes (identical result; lower it if scoring runs out of memory)")
    ap.add_argument("--mambalrp-path", default="/content/MambaLRP")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--calib-n", type=int, default=256)
    ap.add_argument("--calib-len", type=int, default=128)
    ap.add_argument("--calib-pool", type=int, default=20000, help="main: size of the WikiText-103 text pool calibration is sampled from")
    ap.add_argument("--train-rows", type=int, default=60000, help="main: number of WikiText-103 train rows used for fine-tuning data")
    ap.add_argument("--ft-steps", type=int, default=500)
    ap.add_argument("--ft-batch", type=int, default=8)
    ap.add_argument("--ft-seq", type=int, default=128)
    ap.add_argument("--ft-lr", type=float, default=2e-4)
    ap.add_argument("--ft-micro", type=int, default=4, help="micro-batch size for gradient accumulation (lower it if you hit OOM)")
    ap.add_argument("--ft-warmup", type=int, default=20)
    ap.add_argument("--eval-at", default="100,250", help="extra evaluations during fine-tuning (recovery curve); '' to disable")
    ap.add_argument("--eval-max-tokens", type=int, default=0, help="main: cap on evaluation tokens (0 = full test set)")
    ap.add_argument("--stage", choices=["all", "score", "run"], default="all",
                    help="score: only compute+cache scores (needs the slow-path runtime, no mamba_ssm); "
                         "run: prune+fine-tune+evaluate from cached scores (may use mamba_ssm kernels); all: both")
    ap.add_argument("--keep-multiple", type=int, default=1,
                    help="round the number of kept channels to a multiple of this (e.g. 8) if the CUDA kernels need aligned sizes")
    ap.add_argument("--tiny", action="store_true", help="offline plumbing test with a random tiny Mamba")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    args.eval_at_list = [int(x) for x in args.eval_at.split(",") if x.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]
    methods = [m for m in args.methods.split(",") if m]
    sps = [float(s) for s in args.sparsities.split(",")]
    bad = [m for m in methods if m not in ALL_SCORES + ["Random", "Dense"] + list(HYBRIDS) + list(MAGS)]
    assert not bad, f"unknown methods {bad}; choose from {ALL_SCORES + ['Random', 'Dense'] + list(HYBRIDS) + list(MAGS)}"
    device = args.device
    try:
        import mamba_ssm  # noqa: F401
        if device == "cuda" and args.stage != "run":
            sys.exit("mamba_ssm is installed in this runtime: Hugging Face would take the fused-kernel path, which bypasses the "
                     "scoring hooks. Use a fresh Colab runtime WITHOUT the Part-2 install for --stage score/all "
                     "(Runtime -> Disconnect and delete runtime, then reconnect); --stage run is the one meant for the kernel runtime.")
    except ImportError:
        pass

    plan = [(s, m, sp) for s in seeds for m in methods for sp in ([0.0] if m == "Dense" else sps)]
    todo = [p for p in plan if not os.path.exists(run_path(args, p[0], p[1], p[2]))]
    log(f"protocol={args.protocol} | {len(plan)} runs planned, {len(plan) - len(todo)} already done, {len(todo)} to do")
    if args.dry_run:
        for p in todo: print(p)
        return
    if not todo:
        return

    if args.tiny:
        base, tok, D = make_tiny(args)
        base = base.to(device)
    else:
        from transformers import AutoTokenizer, MambaForCausalLM
        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base = MambaForCausalLM.from_pretrained(args.model).to(device).eval()
        D = load_data(args, tok)
    log(f"model: {sum(p.numel() for p in base.parameters()):,} params | device {device}")

    dense_path = os.path.join(args.out, f"dense_baseline_{args.protocol}.json")
    if not os.path.exists(dense_path):
        atomic_json(dense_path, dict(ppl=perplexity(base, D.eval_chunks, device), protocol=args.protocol,
                                     eval_tokens=int(D.eval_chunks.numel())))
    log(f"unpruned model perplexity: {json.load(open(dense_path))['ppl']:.3f}")

    done = 0
    t_start = time.time()
    for seed in seeds:
        seed_todo = [p for p in todo if p[0] == seed]
        if not seed_todo:
            continue
        scores = get_scores(args, base, D, tok, seed, [p[1] for p in seed_todo], device, allow_compute=(args.stage != "run"))
        if args.stage == "score":
            continue
        for (_, method, sp) in seed_todo:
            rec = one_run(args, base, D, scores, seed, method, sp, device)
            atomic_json(run_path(args, seed, method, sp), rec)
            done += 1
            eta = (time.time() - t_start) / done * (len(todo) - done) / 60
            log(f"[{done}/{len(todo)}] seed {seed} {method:11s} sp={sp:.0%}: pre {rec['ppl_pre']:.2f} -> "
                f"post {rec['ppl_post']:.2f}  ({rec['seconds']:.0f}s, ETA {eta:.0f} min)")
    log("scores cached" if args.stage == "score" else "grid finished")


if __name__ == "__main__":
    main()
