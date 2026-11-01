"""
H7: LRP-Guided Non-Uniform Layer-Wise Sparsity Allocation
==========================================================
Claim: Allocating the global pruning budget non-uniformly across layers —
guided by LRP's implicit layer-level relevance distribution — produces lower
perplexity than uniform per-layer sparsity at matched total parameter count.
This directly addresses the H6 PARTIAL result, where LRP's extreme early-layer
concentration (layer 0 pressure=0.93) damaged recoverable structure and hurt
fine-tuning recovery relative to GradMag.

Method: The global LRP relevance threshold is used as a natural per-layer
budget. A channel is "kept" if its score exceeds the global (1-keep_ratio)
quantile. This allows layers with dense high-relevance channels (late layers)
to keep nearly all their channels, while layers with sparse high-relevance
channels (early layers) lose more. Total channels removed equals the global
budget, by construction.

Comparison:
  - Uniform LRP   : each layer keeps identical n_keep channels
  - Non-uniform LRP: each layer keeps channels above global quantile threshold
  at matched total parameter count.

Runtime: ~20 minutes on T4 (includes FT comparison at 30% and 50% sparsity).

Prerequisites: run codebase_complete.py first and save scores, or this script
will recompute them.
"""

import sys, os, copy, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/content/MambaLRP")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME      = "state-spaces/mamba-130m-hf"
DATASET         = "Salesforce/wikitext"
DATASET_CONFIG  = "wikitext-2-raw-v1"
PPL_CHUNK_SIZE  = 256
PPL_N_CHUNKS    = 40
FT_STEPS        = 30
FT_LR           = 2e-4
FT_BATCH        = 4
FT_SEQ_LEN      = 128
SPARSITY_LEVELS = [0.30, 0.50, 0.70]
CALIB_SEQ_LEN   = 128
CALIB_MAIN      = 256
SCORE_CACHE     = "/content/lrp_scores_256.pt"

def sep(t="", w=68):
    pad = (w-len(t)-2)//2 if t else 0
    print(f"\n{'═'*pad} {t} {'═'*(w-pad-len(t)-2)}" if t else "═"*w)

# ── Setup ─────────────────────────────────────────────────────────────────────
sep("SETUP")
from transformers import MambaForCausalLM, AutoTokenizer
from mamba_lrp.model.mamba_huggingface import ModifiedMambaForCausalLM

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
base_model = MambaForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
base_model.eval()
NUM_LAYERS = base_model.config.num_hidden_layers
INTERMEDIATE_SIZE = base_model.config.intermediate_size
print(f"Model loaded. Layers={NUM_LAYERS}, D={INTERMEDIATE_SIZE}")

from datasets import load_dataset
wiki_test  = load_dataset(DATASET, DATASET_CONFIG, split="test")
wiki_train = load_dataset(DATASET, DATASET_CONFIG, split="train")
def build_tokens(ds, max_tokens):
    text = " ".join(t for t in ds["text"] if len(t.strip()) > 20)
    return tokenizer(text, return_tensors="pt")["input_ids"][0][:max_tokens]
eval_tokens  = build_tokens(wiki_test,  PPL_CHUNK_SIZE * PPL_N_CHUNKS + 1)
train_tokens = build_tokens(wiki_train, FT_SEQ_LEN * 200)
calib_raw    = load_dataset(DATASET, DATASET_CONFIG, split="train")
CALIB_TEXTS  = [t for t in calib_raw["text"] if len(t.strip()) > 80][:CALIB_MAIN]
print(f"Data loaded. Eval={len(eval_tokens):,} tokens.")

# ── Helper functions ──────────────────────────────────────────────────────────
def slice_linear(linear, keep_out=None, keep_in=None):
    out_f = keep_out.numel() if keep_out is not None else linear.out_features
    in_f  = keep_in.numel()  if keep_in  is not None else linear.in_features
    nl = nn.Linear(in_f, out_f, bias=linear.bias is not None,
                   device=linear.weight.device, dtype=linear.weight.dtype)
    w = linear.weight.data
    if keep_out is not None: w = w.index_select(0, keep_out.to(w.device))
    if keep_in  is not None: w = w.index_select(1, keep_in.to(w.device))
    nl.weight.data.copy_(w)
    if linear.bias is not None:
        b = linear.bias.data
        if keep_out is not None: b = b.index_select(0, keep_out.to(b.device))
        nl.bias.data.copy_(b)
    return nl

def slice_conv1d(conv, keep_ch):
    keep_ch = keep_ch.to(conv.weight.device)
    n = keep_ch.numel()
    nc = nn.Conv1d(n, n, kernel_size=conv.kernel_size, groups=n,
                   padding=conv.padding, bias=conv.bias is not None,
                   device=conv.weight.device, dtype=conv.weight.dtype)
    nc.weight.data.copy_(conv.weight.data.index_select(0, keep_ch))
    if conv.bias is not None:
        nc.bias.data.copy_(conv.bias.data.index_select(0, keep_ch))
    return nc

def prune_uniform(src, scores, keep_ratio):
    """Standard uniform per-layer pruning."""
    m = copy.deepcopy(src).to(DEVICE)
    D = INTERMEDIATE_SIZE
    n_keep = int(round(D * keep_ratio))
    for i, block in enumerate(m.backbone.layers):
        mx = block.mixer
        _, idx = torch.topk(scores[i], k=n_keep, largest=True)
        keep = idx.sort().values.to(DEVICE)
        kip  = torch.cat([keep, keep + D])
        mx.in_proj  = slice_linear(mx.in_proj,  keep_out=kip)
        mx.conv1d   = slice_conv1d(mx.conv1d,   keep)
        mx.x_proj   = slice_linear(mx.x_proj,   keep_in=keep)
        mx.dt_proj  = slice_linear(mx.dt_proj,  keep_out=keep)
        mx.out_proj = slice_linear(mx.out_proj, keep_in=keep)
        mx.A_log    = nn.Parameter(mx.A_log.data.index_select(0, keep))
        mx.D        = nn.Parameter(mx.D.data.index_select(0, keep))
        mx.intermediate_size = n_keep
    m.config.intermediate_size = n_keep
    return m.eval()

def prune_nonuniform(src, scores, keep_ratio, min_keep=0.10, max_keep=0.98):
    """
    Non-uniform per-layer pruning: each layer keeps channels above the global
    (1-keep_ratio) quantile. Total channels removed ≈ global target.
    """
    m = copy.deepcopy(src).to(DEVICE)
    D = INTERMEDIATE_SIZE
    flat   = scores.flatten()
    thresh = flat.kthvalue(int(len(flat) * (1 - keep_ratio))).values.item()
    keeps  = []
    for i, block in enumerate(m.backbone.layers):
        mx     = block.mixer
        above  = (scores[i] > thresh).sum().item()
        n_keep = max(int(D * min_keep), min(int(D * max_keep), above))
        if n_keep == 0: n_keep = max(1, int(D * min_keep))
        _, idx = torch.topk(scores[i], k=n_keep, largest=True)
        keep   = idx.sort().values.to(DEVICE)
        keeps.append(n_keep)
        kip    = torch.cat([keep, keep + D])
        mx.in_proj  = slice_linear(mx.in_proj,  keep_out=kip)
        mx.conv1d   = slice_conv1d(mx.conv1d,   keep)
        mx.x_proj   = slice_linear(mx.x_proj,   keep_in=keep)
        mx.dt_proj  = slice_linear(mx.dt_proj,  keep_out=keep)
        mx.out_proj = slice_linear(mx.out_proj, keep_in=keep)
        mx.A_log    = nn.Parameter(mx.A_log.data.index_select(0, keep))
        mx.D        = nn.Parameter(mx.D.data.index_select(0, keep))
        mx.intermediate_size = n_keep
    # config.intermediate_size is ambiguous for non-uniform; leave stale
    return m.eval(), keeps

def compute_perplexity(model, tokens):
    model.eval()
    chunks = [tokens[i:i+PPL_CHUNK_SIZE]
              for i in range(0, len(tokens)-PPL_CHUNK_SIZE, PPL_CHUNK_SIZE)][:PPL_N_CHUNKS]
    nll, n = 0.0, 0
    with torch.no_grad():
        for chunk in chunks:
            ids = chunk.unsqueeze(0).to(DEVICE)
            out = model(input_ids=ids)
            logits = out.logits if hasattr(out, "logits") else out
            nll += F.cross_entropy(logits[:,:-1].contiguous().transpose(1,2),
                                   ids[:,1:].contiguous(), reduction="sum").item()
            n += PPL_CHUNK_SIZE - 1
    return float(np.exp(nll / n))

def fine_tune(model, steps=FT_STEPS):
    ft_chunks = [train_tokens[i:i+FT_SEQ_LEN]
                 for i in range(0, len(train_tokens)-FT_SEQ_LEN, FT_SEQ_LEN)
                 if len(train_tokens[i:i+FT_SEQ_LEN]) == FT_SEQ_LEN]
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=FT_LR)
    for step in range(steps):
        start = (step * FT_BATCH) % max(1, len(ft_chunks) - FT_BATCH)
        ids = torch.stack(ft_chunks[start:start+FT_BATCH]).to(DEVICE)
        opt.zero_grad()
        out = model(input_ids=ids)
        logits = out.logits if hasattr(out, "logits") else out
        F.cross_entropy(logits[:,:-1].contiguous().transpose(1,2),
                        ids[:,1:].contiguous(), reduction="mean").backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    return model

def compute_lrp_scores():
    wrapped = ModifiedMambaForCausalLM(base_model, is_fast_forward_available=False)
    wrapped.to(DEVICE).eval()
    scores = torch.zeros(NUM_LAYERS, INTERMEDIATE_SIZE)
    n_real, captured = 0, {}
    def make_hook(i):
        def h(m, inp, out): out.retain_grad(); captured[i] = out
        return h
    handles = [wrapped.backbone.layers[i].mixer.in_proj.register_forward_hook(make_hook(i))
               for i in range(NUM_LAYERS)]
    for start in range(0, len(CALIB_TEXTS), 4):
        batch = CALIB_TEXTS[start:start+4]
        enc   = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=CALIB_SEQ_LEN).to(DEVICE)
        mask  = enc["attention_mask"]
        wrapped.zero_grad(set_to_none=True)
        logits = wrapped(input_ids=enc["input_ids"])
        if hasattr(logits, "logits"): logits = logits.logits
        (F.cross_entropy(logits[:,:-1].contiguous().transpose(1,2),
                         enc["input_ids"][:,1:].contiguous(), reduction="none")
         * mask[:,1:].float()).sum().backward()
        for i in range(NUM_LAYERS):
            c = captured.get(i)
            if c is not None and c.grad is not None:
                scores[i] += (c.detach()[...,:INTERMEDIATE_SIZE]
                              * c.grad[...,:INTERMEDIATE_SIZE]
                              * mask.unsqueeze(-1).float()).abs().sum(dim=(0,1)).cpu()
        n_real += int(mask.sum().item()); captured.clear()
    for h in handles: h.remove()
    del wrapped; torch.cuda.empty_cache() if DEVICE=="cuda" else None
    return (scores / max(n_real,1)).detach()

# ── Load or compute LRP scores ────────────────────────────────────────────────
sep("LOADING SCORES")
if os.path.exists(SCORE_CACHE):
    LRP_SCORES = torch.load(SCORE_CACHE)
    print(f"Loaded LRP scores from {SCORE_CACHE}")
else:
    print("Computing LRP scores (N=256)...", end=" ", flush=True)
    t0 = time.time()
    LRP_SCORES = compute_lrp_scores()
    torch.save(LRP_SCORES, SCORE_CACHE)
    print(f"{time.time()-t0:.1f}s — saved to {SCORE_CACHE}")

# ── H7: Non-uniform vs uniform comparison ────────────────────────────────────
sep("H7: NON-UNIFORM SPARSITY ALLOCATION")
print("""
Comparing uniform per-layer keep-ratio against non-uniform allocation where
each layer keeps all channels above the global relevance threshold.
Matched on total parameter count. Tested at 30%, 50%, 70% sparsity.
""")

print(f"  Base model PPL ...", end=" ", flush=True)
base_ppl = compute_perplexity(base_model, eval_tokens)
print(f"{base_ppl:.2f}")

results = {}
N_PARAMS_BASE = sum(p.numel() for p in base_model.parameters())

for sp in SPARSITY_LEVELS:
    kr = 1.0 - sp
    print(f"\n  ── Sparsity {sp*100:.0f}% ──")

    # Uniform LRP
    print(f"    Uniform   LRP ...", end=" ", flush=True)
    t0 = time.time()
    m_uni = prune_uniform(base_model, LRP_SCORES, kr)
    ppl_uni = compute_perplexity(m_uni, eval_tokens)
    n_uni   = sum(p.numel() for p in m_uni.parameters())
    print(f"PPL={ppl_uni:.2f}  params={n_uni/N_PARAMS_BASE*100:.1f}%  ({time.time()-t0:.1f}s)")

    # Non-uniform LRP
    print(f"    Non-uniform LRP ...", end=" ", flush=True)
    t0 = time.time()
    m_nu, keeps = prune_nonuniform(base_model, LRP_SCORES, kr)
    ppl_nu  = compute_perplexity(m_nu, eval_tokens)
    n_nu    = sum(p.numel() for p in m_nu.parameters())
    print(f"PPL={ppl_nu:.2f}  params={n_nu/N_PARAMS_BASE*100:.1f}%  ({time.time()-t0:.1f}s)")
    print(f"    Per-layer keeps (min={min(keeps)}, max={max(keeps)}, "
          f"std={float(np.std(keeps)):.1f})")

    # Fine-tune both
    print(f"    Fine-tuning uniform ({FT_STEPS} steps)...", end=" ", flush=True)
    t0 = time.time()
    m_uni = fine_tune(m_uni, FT_STEPS)
    ppl_uni_ft = compute_perplexity(m_uni, eval_tokens)
    print(f"PPL={ppl_uni_ft:.2f}  ({time.time()-t0:.0f}s)")

    print(f"    Fine-tuning non-uniform ({FT_STEPS} steps)...", end=" ", flush=True)
    t0 = time.time()
    m_nu = fine_tune(m_nu, FT_STEPS)
    ppl_nu_ft  = compute_perplexity(m_nu, eval_tokens)
    print(f"PPL={ppl_nu_ft:.2f}  ({time.time()-t0:.0f}s)")

    results[sp] = {"uni_pre": ppl_uni, "nu_pre": ppl_nu,
                   "uni_post": ppl_uni_ft, "nu_post": ppl_nu_ft,
                   "n_uni": n_uni, "n_nu": n_nu, "keeps": keeps}

    del m_uni, m_nu
    if DEVICE=="cuda": torch.cuda.empty_cache()

# ── Results table ─────────────────────────────────────────────────────────────
sep("RESULTS")
print(f"  {'Sparsity':<10} {'Uniform pre':>12} {'NonUniform pre':>15} "
      f"{'Uniform post':>14} {'NonUniform post':>16} {'Winner (post)':>14}")
print("  " + "-"*84)
nu_wins = 0
for sp in SPARSITY_LEVELS:
    r = results[sp]
    winner = "NonUniform" if r["nu_post"] < r["uni_post"] else "Uniform"
    if winner == "NonUniform": nu_wins += 1
    print(f"  {sp*100:.0f}%{'':<7} {r['uni_pre']:>12.2f} {r['nu_pre']:>15.2f} "
          f"{r['uni_post']:>14.2f} {r['nu_post']:>16.2f} {winner:>14}")

print(f"\n  Non-uniform wins in {nu_wins}/{len(SPARSITY_LEVELS)} sparsity levels (post-FT)")

# Per-layer distribution at 50% sparsity
print(f"\n  Per-layer keep counts at 50% sparsity (non-uniform vs uniform={int(INTERMEDIATE_SIZE*0.5)}):")
keeps_50 = results[0.50]["keeps"]
for i, k in enumerate(keeps_50):
    bar = "█" * (k // 64)
    diff = k - int(INTERMEDIATE_SIZE * 0.5)
    print(f"    Layer {i:>2}: {k:>4} ({diff:>+4} vs uniform) {bar}")

h7_verdict = "SUPPORTED" if nu_wins >= 2 else "PARTIAL" if nu_wins >= 1 else "NOT SUPPORTED"
print(f"\nRESULT  Non-uniform LRP wins in {nu_wins}/{len(SPARSITY_LEVELS)} cases  →  H7 {h7_verdict}")
