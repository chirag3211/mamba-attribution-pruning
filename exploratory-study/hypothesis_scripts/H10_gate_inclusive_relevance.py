"""
H10: Gate-Inclusive Relevance Scores
=====================================
Claim: Including the gate half of in_proj's output in the relevance formula
produces meaningfully different channel rankings and lower perplexity than
content-only scoring (the current approach).

Background: in_proj outputs [content | gate] concatenated on the last dim.
Current code uses only content: r = |content * content_grad|.
The gate (z) multiplicatively modulates content output: output = content * silu(z).
A channel where the gate strongly activates (large |z * z_grad|) contributes
to the final output even if its content component has low direct relevance.
Gate-inclusive formula: r = |content * content_grad| + |gate * gate_grad|

This is a one-line change to the hook. This script:
  1. Computes content-only scores (current approach, loaded from cache if available)
  2. Computes gate-inclusive scores (new)
  3. Compares channel rankings (Spearman ρ per layer)
  4. Compares one-shot perplexity at key sparsity levels
  5. Reports which approach gives better compression

Runtime: ~15 minutes on T4.
"""

import sys, os, copy, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/content/MambaLRP")

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME  = "state-spaces/mamba-130m-hf"
DATASET     = "Salesforce/wikitext"
DATASET_CFG = "wikitext-2-raw-v1"
PPL_CHUNK   = 256
PPL_N       = 40
FT_STEPS    = 20
FT_LR       = 2e-4
FT_BATCH    = 4
FT_SEQ      = 128
SPARSITIES  = [0.30, 0.50, 0.70]
CALIB_N     = 256
CALIB_SEQ   = 128
LRP_CACHE   = "/content/lrp_scores_256.pt"
GATE_CACHE  = "/content/gate_inclusive_scores.pt"

def sep(t="", w=68):
    pad = (w-len(t)-2)//2 if t else 0
    print(f"\n{'═'*pad} {t} {'═'*(w-pad-len(t)-2)}" if t else "═"*w)

sep("SETUP")
from transformers import MambaForCausalLM, AutoTokenizer
from mamba_lrp.model.mamba_huggingface import ModifiedMambaForCausalLM
from datasets import load_dataset

tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
base_model = MambaForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE); base_model.eval()
NUM_LAYERS = base_model.config.num_hidden_layers
D          = base_model.config.intermediate_size

wiki_test  = load_dataset(DATASET, DATASET_CFG, split="test")
wiki_train = load_dataset(DATASET, DATASET_CFG, split="train")
calib_raw  = load_dataset(DATASET, DATASET_CFG, split="train")
def build_tokens(ds, mx):
    text = " ".join(t for t in ds["text"] if len(t.strip()) > 20)
    return tokenizer(text, return_tensors="pt")["input_ids"][0][:mx]
eval_tok   = build_tokens(wiki_test,  PPL_CHUNK * PPL_N + 1)
train_tok  = build_tokens(wiki_train, FT_SEQ * 200)
CALIB      = [t for t in calib_raw["text"] if len(t.strip()) > 80][:CALIB_N]
print(f"Model loaded. D={D}, layers={NUM_LAYERS}")

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
    nc = nn.Conv1d(n, n, kernel_size=c.kernel_size, groups=n, padding=c.padding,
                   bias=c.bias is not None, device=c.weight.device, dtype=c.weight.dtype)
    nc.weight.data.copy_(c.weight.data.index_select(0, keep))
    if c.bias is not None: nc.bias.data.copy_(c.bias.data.index_select(0, keep))
    return nc

def prune_model(src, scores, kr):
    m = copy.deepcopy(src).to(DEVICE)
    n_keep = int(round(D * kr))
    for i, block in enumerate(m.backbone.layers):
        mx = block.mixer
        _, idx = torch.topk(scores[i], k=n_keep, largest=True)
        keep = idx.sort().values.to(DEVICE)
        kip  = torch.cat([keep, keep + D])
        mx.in_proj  = slice_linear(mx.in_proj,  ko=kip)
        mx.conv1d   = slice_conv(mx.conv1d, keep)
        mx.x_proj   = slice_linear(mx.x_proj,   ki=keep)
        mx.dt_proj  = slice_linear(mx.dt_proj,  ko=keep)
        mx.out_proj = slice_linear(mx.out_proj, ki=keep)
        mx.A_log    = nn.Parameter(mx.A_log.data.index_select(0, keep))
        mx.D        = nn.Parameter(mx.D.data.index_select(0, keep))
        mx.intermediate_size = n_keep
    m.config.intermediate_size = n_keep
    return m.eval()

def ppl(model, tokens):
    model.eval()
    chunks = [tokens[i:i+PPL_CHUNK] for i in range(0, len(tokens)-PPL_CHUNK, PPL_CHUNK)][:PPL_N]
    nll, n = 0.0, 0
    with torch.no_grad():
        for chunk in chunks:
            ids = chunk.unsqueeze(0).to(DEVICE)
            out = model(input_ids=ids)
            lg  = out.logits if hasattr(out, "logits") else out
            nll += F.cross_entropy(lg[:,:-1].contiguous().transpose(1,2),
                                   ids[:,1:].contiguous(), reduction="sum").item()
            n += PPL_CHUNK - 1
    return float(np.exp(nll / n))

ft_chunks = [train_tok[i:i+FT_SEQ] for i in range(0, len(train_tok)-FT_SEQ, FT_SEQ)
             if len(train_tok[i:i+FT_SEQ]) == FT_SEQ]

def fine_tune(model, steps=FT_STEPS):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=FT_LR)
    for step in range(steps):
        start = (step * FT_BATCH) % max(1, len(ft_chunks) - FT_BATCH)
        ids = torch.stack(ft_chunks[start:start+FT_BATCH]).to(DEVICE)
        opt.zero_grad()
        out = model(input_ids=ids)
        lg  = out.logits if hasattr(out, "logits") else out
        F.cross_entropy(lg[:,:-1].contiguous().transpose(1,2),
                        ids[:,1:].contiguous(), reduction="mean").backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval(); return model

def compute_scores(include_gate):
    """Compute LRP scores with or without gate contribution."""
    wrapped = ModifiedMambaForCausalLM(base_model, is_fast_forward_available=False)
    wrapped.to(DEVICE).eval()
    scores = torch.zeros(NUM_LAYERS, D); n_real, captured = 0, {}
    def make_hook(i):
        def h(m, inp, out): out.retain_grad(); captured[i] = out
        return h
    handles = [wrapped.backbone.layers[i].mixer.in_proj.register_forward_hook(make_hook(i))
               for i in range(NUM_LAYERS)]
    for s in range(0, len(CALIB), 4):
        batch = CALIB[s:s+4]
        enc   = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=CALIB_SEQ).to(DEVICE)
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
                m_ = mask.unsqueeze(-1).float()
                content = (c.detach()[...,:D] * c.grad[...,:D] * m_).abs().sum(dim=(0,1))
                if include_gate:
                    gate = (c.detach()[...,D:] * c.grad[...,D:] * m_).abs().sum(dim=(0,1))
                    scores[i] += content.cpu() + gate.cpu()
                else:
                    scores[i] += content.cpu()
        n_real += int(mask.sum().item()); captured.clear()
    for h in handles: h.remove()
    del wrapped; torch.cuda.empty_cache() if DEVICE=="cuda" else None
    return (scores / max(n_real, 1)).detach()

# ── Load / compute scores ──────────────────────────────────────────────────────
sep("COMPUTING SCORES")
if os.path.exists(LRP_CACHE):
    CONTENT_SCORES = torch.load(LRP_CACHE)
    print(f"Content-only scores loaded from {LRP_CACHE}")
else:
    print("Computing content-only scores...", end=" ", flush=True)
    t0 = time.time(); CONTENT_SCORES = compute_scores(include_gate=False)
    torch.save(CONTENT_SCORES, LRP_CACHE); print(f"{time.time()-t0:.1f}s")

if os.path.exists(GATE_CACHE):
    GATE_SCORES = torch.load(GATE_CACHE)
    print(f"Gate-inclusive scores loaded from {GATE_CACHE}")
else:
    print("Computing gate-inclusive scores...", end=" ", flush=True)
    t0 = time.time(); GATE_SCORES = compute_scores(include_gate=True)
    torch.save(GATE_SCORES, GATE_CACHE); print(f"{time.time()-t0:.1f}s")

# ── Ranking comparison ─────────────────────────────────────────────────────────
sep("H10: RANKING COMPARISON — CONTENT-ONLY vs GATE-INCLUSIVE")
global_rho, _ = spearmanr(CONTENT_SCORES.flatten().numpy(),
                           GATE_SCORES.flatten().numpy())
print(f"\n  Global Spearman ρ (content vs gate-inclusive): {global_rho:.4f}")
print(f"\n  Per-layer ρ and gate contribution fraction:")
print(f"  {'Layer':<8} {'ρ':>8}  {'Gate/Total ratio':>18}  Interpretation")
print("  " + "-"*60)
for i in range(NUM_LAYERS):
    rho_i, _ = spearmanr(CONTENT_SCORES[i].numpy(), GATE_SCORES[i].numpy())
    # Approximate gate contribution as fraction of total combined score
    content_sum = CONTENT_SCORES[i].sum().item()
    gate_approx = GATE_SCORES[i].sum().item() - content_sum
    if GATE_SCORES[i].sum().item() > 0:
        gate_frac = max(0, gate_approx) / (GATE_SCORES[i].sum().item() + 1e-8)
    else:
        gate_frac = 0.0
    flag = " ← gate matters" if rho_i < 0.80 else ""
    print(f"  {i:>5}    {rho_i:>8.4f}  {gate_frac:>17.3f}  {flag}")

# ── Perplexity comparison ──────────────────────────────────────────────────────
sep("PERPLEXITY COMPARISON")
print(f"  Base PPL: ", end=""); base_ppl = ppl(base_model, eval_tok); print(f"{base_ppl:.2f}")
results = {}
for sp in SPARSITIES:
    kr = 1.0 - sp
    print(f"\n  Sparsity {sp*100:.0f}%:")
    for name, scores in [("Content-only", CONTENT_SCORES), ("Gate-inclusive", GATE_SCORES)]:
        print(f"    {name:<16} ...", end=" ", flush=True)
        t0 = time.time()
        m = prune_model(base_model, scores, kr)
        p_pre = ppl(m, eval_tok)
        m = fine_tune(m, FT_STEPS)
        p_post = ppl(m, eval_tok)
        results.setdefault(sp, {})[name] = {"pre": p_pre, "post": p_post}
        print(f"pre={p_pre:.2f}  post={p_post:.2f}  ({time.time()-t0:.0f}s)")
        del m; torch.cuda.empty_cache() if DEVICE=="cuda" else None

sep("RESULTS")
print(f"\n  {'Sparsity':<10}", end="")
for sp in SPARSITIES:
    print(f"  {sp*100:.0f}% pre  {sp*100:.0f}% post", end="")
print()
print("  " + "-"*70)
for name in ["Content-only", "Gate-inclusive"]:
    print(f"  {name:<16}", end="")
    for sp in SPARSITIES:
        print(f"  {results[sp][name]['pre']:>8.2f}  {results[sp][name]['post']:>9.2f}", end="")
    print()

gate_wins = sum(1 for sp in SPARSITIES
                if results[sp]["Gate-inclusive"]["post"] < results[sp]["Content-only"]["post"])
h10_verdict = ("SUPPORTED" if gate_wins == len(SPARSITIES)
               else "PARTIAL" if gate_wins > 0
               else f"NOT SUPPORTED (content-only wins; gate contributes {100*(1-global_rho):.0f}% ranking change)")
print(f"\n  Global ρ (content vs gate-inclusive): {global_rho:.4f}")
print(f"  (ρ < 0.90 means gate changes rankings meaningfully)")
print(f"\nRESULT  Gate-inclusive better in {gate_wins}/{len(SPARSITIES)} cases  →  H10 {h10_verdict}")
print(f"\nNote: if ρ > 0.95, gate contribution is negligible and content-only is justified.")
print(f"If ρ < 0.80, gate-inclusive is capturing meaningful additional signal.")
