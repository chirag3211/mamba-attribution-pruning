"""
H9: Hybrid LRP + GradMag Scoring by Layer Depth
================================================
Claim: Combining LRP scores for layers where LRP-GI divergence is high
(layers 12-23, where LRP corrections matter most) and GradMag scores for
early layers (0-11, where LRP and GI largely agree) produces lower perplexity
than either pure method.

Motivation from H2: LRP-GI Spearman ρ decreases monotonically with layer
depth (early ρ=0.87, late ρ=0.65). This means LRP's conservation corrections
provide the most unique information in middle-to-late layers. In early layers,
LRP and vanilla GI rank channels similarly — so GradMag (which incorporates
gradient information like GI but without LRP wiring) is an equally valid and
potentially more stable scoring signal there.

Split strategy tested:
  Hybrid-12: GradMag for layers 0-11, LRP for layers 12-23 (main)
  Hybrid-8 : GradMag for layers 0-7,  LRP for layers 8-23  (ablation)

Runtime: ~15 minutes on T4 (scores loaded from disk if available).
"""

import sys, os, copy, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/content/MambaLRP")

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME   = "state-spaces/mamba-130m-hf"
DATASET      = "Salesforce/wikitext"
DATASET_CFG  = "wikitext-2-raw-v1"
PPL_CHUNK    = 256
PPL_N        = 40
FT_STEPS     = 30
FT_LR        = 2e-4
FT_BATCH     = 4
FT_SEQ       = 128
SPARSITIES   = [0.30, 0.50, 0.70]
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

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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
print(f"Model loaded. Layers={NUM_LAYERS}, D={D}")

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

def prune_with_scores(src, scores, keep_ratio):
    m = copy.deepcopy(src).to(DEVICE)
    n_keep = int(round(D * keep_ratio))
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

def compute_lrp():
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
                scores[i] += (c.detach()[...,:D] * c.grad[...,:D]
                              * mask.unsqueeze(-1).float()).abs().sum(dim=(0,1)).cpu()
        n_real += int(mask.sum().item()); captured.clear()
    for h in handles: h.remove()
    del wrapped; torch.cuda.empty_cache() if DEVICE=="cuda" else None
    return (scores / max(n_real, 1)).detach()

def compute_gradmag():
    model = MambaForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.train(); model.zero_grad(set_to_none=True)
    for s in range(0, len(CALIB), 4):
        batch = CALIB[s:s+4]
        enc   = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=CALIB_SEQ).to(DEVICE)
        mask  = enc["attention_mask"]
        out   = model(**enc)
        logits = out.logits if hasattr(out, "logits") else out
        (F.cross_entropy(logits[:,:-1].contiguous().transpose(1,2),
                         enc["input_ids"][:,1:].contiguous(), reduction="none")
         * mask[:,1:].float()).sum().backward()
    scores = torch.zeros(NUM_LAYERS, D)
    for i, block in enumerate(model.backbone.layers):
        w = block.mixer.in_proj.weight.data
        g = block.mixer.in_proj.weight.grad
        if g is not None:
            scores[i] = (w[:D] * g[:D]).abs().mean(dim=1).cpu()
        else:
            scores[i] = w[:D].abs().mean(dim=1).cpu()
    del model; torch.cuda.empty_cache() if DEVICE=="cuda" else None
    return scores.detach()

# ── Load / compute scores ──────────────────────────────────────────────────────
sep("LOADING SCORES")
LRP_SCORES = torch.load(LRP_CACHE) if os.path.exists(LRP_CACHE) else compute_lrp()
if not os.path.exists(LRP_CACHE): torch.save(LRP_SCORES, LRP_CACHE)
GM_SCORES  = torch.load(GM_CACHE)  if os.path.exists(GM_CACHE)  else compute_gradmag()
if not os.path.exists(GM_CACHE): torch.save(GM_SCORES, GM_CACHE)
print(f"LRP scores:    shape={tuple(LRP_SCORES.shape)}")
print(f"GradMag scores: shape={tuple(GM_SCORES.shape)}")

# ── Build hybrid score tensors ─────────────────────────────────────────────────
def make_hybrid(lrp, gm, lrp_from_layer):
    """Use GradMag for layers 0..lrp_from_layer-1, LRP for the rest."""
    hybrid = torch.zeros_like(lrp)
    for i in range(NUM_LAYERS):
        # Normalize each layer's scores to [0,1] before mixing to avoid scale issues
        lrp_norm = (lrp[i] - lrp[i].min()) / (lrp[i].max() - lrp[i].min() + 1e-8)
        gm_norm  = (gm[i]  - gm[i].min())  / (gm[i].max()  - gm[i].min()  + 1e-8)
        hybrid[i] = lrp_norm if i >= lrp_from_layer else gm_norm
    return hybrid

HYBRID_12 = make_hybrid(LRP_SCORES, GM_SCORES, lrp_from_layer=12)
HYBRID_8  = make_hybrid(LRP_SCORES, GM_SCORES, lrp_from_layer=8)

# Compare rankings
all_scores = {"LRP": LRP_SCORES, "GradMag": GM_SCORES,
              "Hybrid-12": HYBRID_12, "Hybrid-8": HYBRID_8}

sep("SCORE COMPARISON")
print("  Global Spearman ρ between scoring methods (all layers × channels):")
names = list(all_scores.keys())
for i, na in enumerate(names):
    for nb in names[i+1:]:
        rho, _ = spearmanr(all_scores[na].flatten().numpy(),
                           all_scores[nb].flatten().numpy())
        print(f"    {na:<12} vs {nb:<12}: ρ={rho:.4f}")

# ── Perplexity comparison ──────────────────────────────────────────────────────
sep("H9: HYBRID vs PURE METHODS — PPL COMPARISON")
print(f"\n  Base model PPL: ", end=""); base_ppl = ppl(base_model, eval_tok); print(f"{base_ppl:.2f}")

results = {k: {} for k in all_scores}
for sp in SPARSITIES:
    kr = 1.0 - sp
    print(f"\n  ── Sparsity {sp*100:.0f}% ──")
    for name, scores in all_scores.items():
        # Normalize scores for fair comparison (already done for hybrids, do for pure too)
        norm_scores = torch.zeros_like(scores)
        for i in range(NUM_LAYERS):
            s = scores[i]
            norm_scores[i] = (s - s.min()) / (s.max() - s.min() + 1e-8)
        print(f"    {name:<12} ...", end=" ", flush=True)
        t0 = time.time()
        m = prune_with_scores(base_model, norm_scores, kr)
        p_pre = ppl(m, eval_tok)
        m = fine_tune(m, FT_STEPS)
        p_post = ppl(m, eval_tok)
        results[name][sp] = {"pre": p_pre, "post": p_post}
        print(f"pre={p_pre:.2f}  post={p_post:.2f}  ({time.time()-t0:.0f}s)")
        del m; torch.cuda.empty_cache() if DEVICE=="cuda" else None

sep("RESULTS")
print(f"\n  {'Method':<12} ", end="")
for sp in SPARSITIES:
    print(f"{'%d%% pre' % (sp*100):>10} {'%d%% post' % (sp*100):>11}", end="")
print()
print("  " + "-" * (12 + len(SPARSITIES) * 22))
for name in names:
    print(f"  {name:<12} ", end="")
    for sp in SPARSITIES:
        print(f"{results[name][sp]['pre']:>10.2f} {results[name][sp]['post']:>11.2f}", end="")
    print()

print("\n  Post-FT winner per sparsity:")
hybrid_wins = 0
for sp in SPARSITIES:
    best = min(names, key=lambda n: results[n][sp]["post"])
    print(f"    {sp*100:.0f}%: {best}  (PPL={results[best][sp]['post']:.2f})")
    if "Hybrid" in best: hybrid_wins += 1

h9_verdict = ("SUPPORTED" if hybrid_wins == len(SPARSITIES)
              else "PARTIAL" if hybrid_wins > 0 else "NOT SUPPORTED")
print(f"\nRESULT  Hybrid wins in {hybrid_wins}/{len(SPARSITIES)} sparsity levels  →  H9 {h9_verdict}")
