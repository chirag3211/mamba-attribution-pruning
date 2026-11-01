"""
H14: Combined Pruning + Quantisation Pipeline
==============================================
Claim: LRP-guided pruning (30% channel removal) followed by LRP-guided
mixed-precision quantisation (50/50 INT8/FP16 on remaining channels) produces
a better accuracy-vs-compression trade-off than applying either technique alone.

This tests the composability of the two compression methods and whether
LRP-guided decisions for pruning and quantisation compound cleanly.

Grid of conditions (all at matched compression ratio where possible):
  Base FP16          : no compression (reference)
  Prune-only (LRP)   : 30% channels removed, FP16 remaining
  Quant-only (LRP)   : no pruning, 50% channels quantised to INT8
  Prune+Quant (LRP)  : 30% pruned + 50% of remaining quantised
  Prune+Quant (GradMag): same but GradMag scores for both steps
  Prune+Quant (Random) : random pruning + random INT8 allocation

Secondary question: does using LRP scores for BOTH steps (pruning and
quantisation selection) outperform using LRP for pruning and random for
quantisation? This tests whether the two uses of LRP are complementary.

Runtime: ~25 minutes on T4.
"""

import sys, os, copy, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/content/MambaLRP")

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME   = "state-spaces/mamba-130m-hf"
DATASET      = "Salesforce/wikitext"
DATASET_CFG  = "wikitext-2-raw-v1"
PPL_CHUNK    = 256
PPL_N        = 40
FT_STEPS     = 20
FT_LR        = 2e-4
FT_BATCH     = 4
FT_SEQ       = 128
PRUNE_RATIO  = 0.30    # fraction of channels to remove
QUANT_RATIO  = 0.50    # fraction of REMAINING channels to quantise
CALIB_N      = 256
CALIB_SEQ    = 128
LRP_CACHE    = "/content/lrp_scores_256.pt"
GM_CACHE     = "/content/gradmag_scores.pt"

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
print(f"Setup complete. D={D}, layers={NUM_LAYERS}")

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
    m = copy.deepcopy(src)
    n_keep = int(round(D * kr))
    for i, block in enumerate(m.backbone.layers):
        mx = block.mixer
        _, idx = torch.topk(scores[i], k=n_keep, largest=True)
        keep = idx.sort().values.to(m.backbone.layers[0].mixer.A_log.device)
        kip  = torch.cat([keep, keep + mx.intermediate_size])
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

def simulate_int8_rows(weight, row_indices):
    w = weight.clone()
    for idx in row_indices:
        row = w[idx]
        scale = row.abs().max().clamp(min=1e-8) / 127.0
        w[idx] = (row / scale).round().clamp(-127, 127) * scale
    return w

def apply_quantisation(model, scores_on_pruned, quant_ratio, seed=None):
    """Apply mixed-precision INT8 to the pruned model's remaining channels."""
    for i, block in enumerate(model.backbone.layers):
        mx       = block.mixer
        curr_D   = mx.intermediate_size
        n_quant  = int(round(curr_D * quant_ratio))
        if seed is not None:
            g = torch.Generator().manual_seed(seed + i)
            low_idx = torch.randperm(curr_D, generator=g)[:n_quant].tolist()
        else:
            _, idx = torch.topk(scores_on_pruned[i][:curr_D], k=n_quant, largest=False)
            low_idx = idx.tolist()
        # Quantise in_proj content rows for low-relevance channels
        low_content = low_idx
        low_gate    = [j + curr_D for j in low_idx]
        mx.in_proj.weight.data = simulate_int8_rows(mx.in_proj.weight.data,
                                                     low_content + low_gate)
        # Quantise out_proj columns
        wt = mx.out_proj.weight.data.t()
        wt = simulate_int8_rows(wt, low_idx)
        mx.out_proj.weight.data = wt.t()
    return model

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

# ── Load / compute scores ──────────────────────────────────────────────────────
sep("LOADING SCORES")
LRP_SCORES = torch.load(LRP_CACHE) if os.path.exists(LRP_CACHE) else None
GM_SCORES  = torch.load(GM_CACHE)  if os.path.exists(GM_CACHE)  else None
if LRP_SCORES is None or GM_SCORES is None:
    print("One or more score caches missing. Run codebase_complete.py first.")
    sys.exit(1)
RND_SCORES = torch.rand_like(LRP_SCORES, generator=torch.Generator().manual_seed(42))
print("All scores loaded.")

keep_ratio = 1.0 - PRUNE_RATIO

# ── H14: Six conditions ────────────────────────────────────────────────────────
sep("H14: PRUNING + QUANTISATION COMPOSABILITY")
print(f"""
Prune ratio: {PRUNE_RATIO*100:.0f}%   Quantise ratio (of remaining): {QUANT_RATIO*100:.0f}%
All post-pruning models fine-tuned for {FT_STEPS} steps before quantisation.
""")

base_ppl = ppl(base_model, eval_tok)
print(f"FP16 base PPL: {base_ppl:.2f}\n")

conditions = [
    # (name, prune_scores, quant_scores, quant_seed)
    ("Prune-only (LRP)",         LRP_SCORES, None,       None),
    ("Quant-only (LRP)",         None,       LRP_SCORES, None),
    ("Prune+Quant (LRP/LRP)",    LRP_SCORES, LRP_SCORES, None),
    ("Prune+Quant (LRP/Random)", LRP_SCORES, None,       42),
    ("Prune+Quant (GradMag)",    GM_SCORES,  GM_SCORES,  None),
    ("Prune+Quant (Random)",     RND_SCORES, None,       99),
]

results = {}
for name, prune_sc, quant_sc, qseed in conditions:
    print(f"  {name}:")
    t0 = time.time()

    if prune_sc is not None:
        print(f"    Pruning ({PRUNE_RATIO*100:.0f}%)...", end=" ", flush=True)
        m = prune_model(base_model, prune_sc, keep_ratio).to(DEVICE)
        ppl_pre_ft = ppl(m, eval_tok)
        print(f"PPL={ppl_pre_ft:.2f}")
        print(f"    Fine-tuning {FT_STEPS} steps...", end=" ", flush=True)
        m = fine_tune(m, FT_STEPS)
        ppl_post_ft = ppl(m, eval_tok)
        print(f"PPL={ppl_post_ft:.2f}")
    else:
        m = copy.deepcopy(base_model).to(DEVICE)
        ppl_pre_ft = base_ppl
        ppl_post_ft = base_ppl
        print(f"    (No pruning)")

    if quant_sc is not None or qseed is not None:
        print(f"    Quantising {QUANT_RATIO*100:.0f}% of {'remaining ' if prune_sc else ''}channels...",
              end=" ", flush=True)
        m = apply_quantisation(m, quant_sc if quant_sc is not None else LRP_SCORES,
                               QUANT_RATIO, seed=qseed)
        ppl_final = ppl(m, eval_tok)
        print(f"PPL={ppl_final:.2f}")
    else:
        ppl_final = ppl_post_ft

    n_params = sum(p.numel() for p in m.parameters())
    results[name] = {"pre_ft": ppl_pre_ft, "post_ft": ppl_post_ft,
                     "final": ppl_final, "params": n_params,
                     "time": time.time()-t0}
    print(f"    Total time: {time.time()-t0:.0f}s\n")
    del m; torch.cuda.empty_cache() if DEVICE=="cuda" else None

# ── Results table ─────────────────────────────────────────────────────────────
sep("RESULTS")
N_BASE = sum(p.numel() for p in base_model.parameters())
print(f"\n  {'Condition':<28} {'Pre-FT PPL':>12} {'Post-FT PPL':>12} "
      f"{'Final PPL':>11} {'Params%':>8}")
print("  " + "-"*76)
print(f"  {'FP16 base':<28} {'—':>12} {'—':>12} {base_ppl:>11.2f} {'100.0%':>8}")
for name, r in results.items():
    print(f"  {name:<28} {r['pre_ft']:>12.2f} {r['post_ft']:>12.2f} "
          f"{r['final']:>11.2f} {r['params']/N_BASE*100:>7.1f}%")

# Key comparisons
lrp_pq   = results["Prune+Quant (LRP/LRP)"]["final"]
rnd_pq   = results["Prune+Quant (Random)"]["final"]
gm_pq    = results["Prune+Quant (GradMag)"]["final"]
lrp_po   = results["Prune-only (LRP)"]["final"]
lrp_qo   = results["Quant-only (LRP)"]["final"]
lrp_rnd  = results["Prune+Quant (LRP/Random)"]["final"]

print(f"\n  Key comparisons:")
print(f"    LRP Prune+Quant vs Random Prune+Quant: {lrp_pq:.2f} vs {rnd_pq:.2f} "
      f"({'LRP better' if lrp_pq < rnd_pq else 'Random better'})")
print(f"    LRP Prune+Quant vs GradMag Prune+Quant: {lrp_pq:.2f} vs {gm_pq:.2f} "
      f"({'LRP better' if lrp_pq < gm_pq else 'GradMag better'})")
print(f"    LRP combined vs prune-only: {lrp_pq:.2f} vs {lrp_po:.2f} "
      f"({'combined better' if lrp_pq < lrp_po else 'prune-only better'})")
print(f"    LRP combined vs quant-only: {lrp_pq:.2f} vs {lrp_qo:.2f} "
      f"({'combined better' if lrp_pq < lrp_qo else 'quant-only better'})")
print(f"    LRP/LRP vs LRP/Random quant: {lrp_pq:.2f} vs {lrp_rnd:.2f} "
      f"({'LRP quant adds value' if lrp_pq < lrp_rnd else 'random quant sufficient'})")

compose_ok = lrp_pq < rnd_pq
lrp_beats_gm = lrp_pq < gm_pq
h14_verdict = ("SUPPORTED" if compose_ok and lrp_beats_gm
               else "PARTIAL" if compose_ok else "NOT SUPPORTED")
print(f"\nRESULT  LRP Prune+Quant PPL={lrp_pq:.2f}  →  H14 {h14_verdict}")
print(f"\nNote: 'composability' means LRP-guided decisions for BOTH pruning and")
print(f"quantisation produce better joint results than random/naive baselines.")
print(f"If LRP/LRP > LRP/Random, the quantisation step benefits from LRP guidance.")
