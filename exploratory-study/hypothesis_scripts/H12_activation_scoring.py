"""
H12: Activation Magnitude as a Pruning Criterion
=================================================
Claim: Scoring channels by mean absolute activation (no gradient required)
is insufficient for Mamba — it performs worse than LRP and gradient-aware
magnitude because it cannot distinguish channels that activate strongly but
contribute little to the final output (suppressed by the gate or overridden
by the SSM recurrence).

Method: Score each channel by the mean |activation| at the in_proj output
over a calibration set. This is a forward-pass-only operation — faster than
LRP or GradMag but ignores gradient information entirely.

Hypothesis sub-claims:
  a) Activation magnitude rankings differ from LRP (ρ < 0.80 globally)
  b) Activation magnitude pruning produces higher one-shot perplexity than LRP
  c) Activation magnitude pruning produces higher or similar perplexity to GradMag,
     confirming that gradient information is necessary for useful scoring in Mamba

Runtime: ~15 minutes on T4 (activation scoring is very fast; PPL evaluation
dominates).
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
GM_CACHE    = "/content/gradmag_scores.pt"
ACT_CACHE   = "/content/activation_scores.pt"

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
print(f"Setup complete. D={D}")

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

def compute_activation_scores():
    """
    Forward-pass-only scoring: mean |activation| at in_proj content output.
    No backward pass required — fastest scoring method.
    """
    wrapped = ModifiedMambaForCausalLM(base_model, is_fast_forward_available=False)
    wrapped.to(DEVICE).eval()
    scores = torch.zeros(NUM_LAYERS, D); n_real, captured = 0, {}
    def make_hook(i):
        def h(m, inp, out): captured[i] = out.detach()
        return h
    handles = [wrapped.backbone.layers[i].mixer.in_proj.register_forward_hook(make_hook(i))
               for i in range(NUM_LAYERS)]
    for s in range(0, len(CALIB), 4):
        batch = CALIB[s:s+4]
        enc   = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=CALIB_SEQ).to(DEVICE)
        mask  = enc["attention_mask"]
        with torch.no_grad():
            wrapped(input_ids=enc["input_ids"])
        for i in range(NUM_LAYERS):
            c = captured.get(i)
            if c is not None:
                # Mean absolute content activation across batch and sequence
                scores[i] += (c[...,:D] * mask.unsqueeze(-1).float()).abs().sum(dim=(0,1)).cpu()
        n_real += int(mask.sum().item()); captured.clear()
    for h in handles: h.remove()
    del wrapped; torch.cuda.empty_cache() if DEVICE=="cuda" else None
    return (scores / max(n_real, 1)).detach()

# ── Load/compute all scores ────────────────────────────────────────────────────
sep("COMPUTING SCORES")
LRP_SCORES = torch.load(LRP_CACHE) if os.path.exists(LRP_CACHE) else None
if LRP_SCORES is None:
    print("LRP scores not found — run codebase_complete.py first or H10 script.")
    from mamba_lrp.model.mamba_huggingface import ModifiedMambaForCausalLM as _M
    # Minimal recompute
    LRP_SCORES = compute_activation_scores()  # fallback placeholder
    print("WARNING: using activation scores as LRP placeholder. Run main script first.")
else:
    print(f"LRP scores loaded.")

GM_SCORES = torch.load(GM_CACHE) if os.path.exists(GM_CACHE) else None
if GM_SCORES is None:
    print("GradMag not found — computing...", end=" ", flush=True)
    model = MambaForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.train(); model.zero_grad(set_to_none=True)
    for s in range(0, len(CALIB), 4):
        batch = CALIB[s:s+4]
        enc   = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=CALIB_SEQ).to(DEVICE)
        out   = model(**enc)
        lg    = out.logits if hasattr(out, "logits") else out
        mask  = enc["attention_mask"]
        (F.cross_entropy(lg[:,:-1].contiguous().transpose(1,2),
                         enc["input_ids"][:,1:].contiguous(), reduction="none")
         * mask[:,1:].float()).sum().backward()
    GM_SCORES = torch.zeros(NUM_LAYERS, D)
    for i, block in enumerate(model.backbone.layers):
        w = block.mixer.in_proj.weight.data; g = block.mixer.in_proj.weight.grad
        GM_SCORES[i] = (w[:D] * g[:D]).abs().mean(dim=1).cpu() if g is not None else w[:D].abs().mean(dim=1).cpu()
    torch.save(GM_SCORES, GM_CACHE); del model; torch.cuda.empty_cache() if DEVICE=="cuda" else None
    print("done")
else:
    print(f"GradMag scores loaded.")

if os.path.exists(ACT_CACHE):
    ACT_SCORES = torch.load(ACT_CACHE)
    print(f"Activation scores loaded from {ACT_CACHE}")
else:
    print("Computing activation scores (no backward needed)...", end=" ", flush=True)
    t0 = time.time(); ACT_SCORES = compute_activation_scores()
    torch.save(ACT_SCORES, ACT_CACHE); print(f"{time.time()-t0:.1f}s")

# ── Ranking comparison ─────────────────────────────────────────────────────────
sep("H12: ACTIVATION RANKING vs LRP AND GRADMAG")
from scipy.stats import spearmanr
print()
for name, scores in [("LRP", LRP_SCORES), ("GradMag", GM_SCORES)]:
    rho, _ = spearmanr(ACT_SCORES.flatten().numpy(), scores.flatten().numpy())
    print(f"  Activation vs {name:<8}: global ρ={rho:.4f}")

print(f"\n  Per-layer ρ (Activation vs LRP):")
for i in range(NUM_LAYERS):
    rho_i, _ = spearmanr(ACT_SCORES[i].numpy(), LRP_SCORES[i].numpy())
    bar  = "█" * int(rho_i * 20)
    flag = " ← differs" if rho_i < 0.70 else ""
    print(f"    Layer {i:>2}: {rho_i:.4f} {bar}{flag}")

# ── PPL comparison ─────────────────────────────────────────────────────────────
sep("PERPLEXITY COMPARISON")
base_ppl_val = ppl(base_model, eval_tok)
print(f"  Base PPL: {base_ppl_val:.2f}")
all_scores = {"LRP": LRP_SCORES, "GradMag": GM_SCORES, "Activation": ACT_SCORES}
results = {sp: {} for sp in SPARSITIES}
for sp in SPARSITIES:
    kr = 1.0 - sp
    print(f"\n  Sparsity {sp*100:.0f}%:")
    for name, scores in all_scores.items():
        print(f"    {name:<12} ...", end=" ", flush=True)
        t0 = time.time()
        m = prune_model(base_model, scores, kr)
        p_pre = ppl(m, eval_tok)
        m = fine_tune(m, FT_STEPS)
        p_post = ppl(m, eval_tok)
        results[sp][name] = {"pre": p_pre, "post": p_post}
        print(f"pre={p_pre:.2f}  post={p_post:.2f}  ({time.time()-t0:.0f}s)")
        del m; torch.cuda.empty_cache() if DEVICE=="cuda" else None

sep("RESULTS")
print(f"\n  {'Method':<12}", end="")
for sp in SPARSITIES:
    print(f"  {sp*100:.0f}% pre {sp*100:.0f}% post", end="")
print()
print("  " + "-"*70)
for name in ["LRP", "GradMag", "Activation"]:
    print(f"  {name:<12}", end="")
    for sp in SPARSITIES:
        print(f"  {results[sp][name]['pre']:>8.2f}{results[sp][name]['post']:>9.2f}", end="")
    print()

act_beats_lrp = sum(1 for sp in SPARSITIES
                    if results[sp]["Activation"]["post"] < results[sp]["LRP"]["post"])
act_beats_gm  = sum(1 for sp in SPARSITIES
                    if results[sp]["Activation"]["post"] < results[sp]["GradMag"]["post"])
rho_lrp, _ = spearmanr(ACT_SCORES.flatten().numpy(), LRP_SCORES.flatten().numpy())
rho_gm,  _ = spearmanr(ACT_SCORES.flatten().numpy(), GM_SCORES.flatten().numpy())

h12_verdict = ("NOT SUPPORTED — gradient information is necessary"
               if act_beats_lrp == 0 and act_beats_gm == 0
               else "PARTIAL — activation beats gradient methods in some regimes"
               if act_beats_lrp > 0 or act_beats_gm > 0
               else "SUPPORTED")
print(f"\n  Activation vs LRP   ρ={rho_lrp:.4f}  Activation beats LRP   in {act_beats_lrp}/{len(SPARSITIES)}")
print(f"  Activation vs GradMag ρ={rho_gm:.4f}  Activation beats GradMag in {act_beats_gm}/{len(SPARSITIES)}")
print(f"\nRESULT  →  H12 {h12_verdict}")
print(f"\nNote: if activation magnitude beats neither LRP nor GradMag, this confirms")
print(f"that gradient information (which method is important for outputs, not just")
print(f"which method is active) is necessary for useful Mamba channel scoring.")
