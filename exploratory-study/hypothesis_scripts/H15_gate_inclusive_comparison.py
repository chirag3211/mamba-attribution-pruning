"""
H15: Gate-Inclusive Scoring — Is It LRP or Just Gate Inclusion That Matters?
==============================================================================
H10 showed that gate-inclusive LRP dramatically outperforms content-only LRP.
This raises a critical question: is the improvement from LRP's conservation
corrections, or simply from including the gate path in any scoring method?

If gate-inclusive GradMag ≈ gate-inclusive LRP  →  gate inclusion is the insight;
                                                     LRP adds nothing over GradMag.
If gate-inclusive LRP >> gate-inclusive GradMag →  LRP's corrections matter beyond
                                                     just knowing about the gate.

Scoring methods compared (all gate-inclusive where applicable):
  LRP (content-only)       : current baseline from H4/H6
  LRP (gate-inclusive)     : H10 result — the best seen so far
  GradMag (content-only)   : H4/H6 baseline
  GradMag (gate-inclusive) : NEW — |w[:D]*g[:D]| + |w[D:]*g[D:]| per channel
  Activation (content-only): H12 result
  Activation (gate-incl.)  : NEW — mean(|out[:D]|) + mean(|out[D:]|)
  Random                   : unchanged (no gate concept)

Runtime: ~25 minutes on T4 (scoring + H4/H6 perplexity grid).
Scores are loaded from cache where available.
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

# Cache paths
LRP_CO_CACHE  = "/content/lrp_scores_256.pt"          # content-only LRP
LRP_GI_CACHE  = "/content/gate_inclusive_scores.pt"    # gate-inclusive LRP (H10)
GM_CO_CACHE   = "/content/gradmag_scores.pt"           # content-only GradMag
GM_GI_CACHE   = "/content/gradmag_gate_inclusive.pt"   # gate-inclusive GradMag (NEW)
ACT_CO_CACHE  = "/content/activation_scores.pt"        # content-only activation (H12)
ACT_GI_CACHE  = "/content/activation_gate_inclusive.pt"# gate-inclusive activation (NEW)

def sep(t="", w=70):
    pad = (w-len(t)-2)//2 if t else 0
    print(f"\n{'═'*pad} {t} {'═'*(w-pad-len(t)-2)}" if t else "═"*w)

sep("SETUP")
from transformers import MambaForCausalLM, AutoTokenizer
from mamba_lrp.model.mamba_huggingface import ModifiedMambaForCausalLM
from datasets import load_dataset

tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
base_model = MambaForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
base_model.eval()
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

# ── Pruning helpers ────────────────────────────────────────────────────────────
def slice_linear(l, ko=None, ki=None):
    of = ko.numel() if ko is not None else l.out_features
    inf = ki.numel() if ki is not None else l.in_features
    nl = nn.Linear(inf, of, bias=l.bias is not None,
                   device=l.weight.device, dtype=l.weight.dtype)
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
    nc = nn.Conv1d(n, n, kernel_size=c.kernel_size, groups=n,
                   padding=c.padding, bias=c.bias is not None,
                   device=c.weight.device, dtype=c.weight.dtype)
    nc.weight.data.copy_(c.weight.data.index_select(0, keep))
    if c.bias is not None:
        nc.bias.data.copy_(c.bias.data.index_select(0, keep))
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
    chunks = [tokens[i:i+PPL_CHUNK]
              for i in range(0, len(tokens)-PPL_CHUNK, PPL_CHUNK)][:PPL_N]
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

ft_chunks = [train_tok[i:i+FT_SEQ]
             for i in range(0, len(train_tok)-FT_SEQ, FT_SEQ)
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
    model.eval()
    return model

# ── Scoring functions ──────────────────────────────────────────────────────────
def compute_lrp(gate_inclusive):
    """MambaLRP scores — content-only or gate-inclusive."""
    wrapped = ModifiedMambaForCausalLM(base_model, is_fast_forward_available=False)
    wrapped.to(DEVICE).eval()
    scores = torch.zeros(NUM_LAYERS, D)
    n_real, captured = 0, {}
    def make_hook(i):
        def h(m, inp, out): out.retain_grad(); captured[i] = out
        return h
    handles = [wrapped.backbone.layers[i].mixer.in_proj
               .register_forward_hook(make_hook(i)) for i in range(NUM_LAYERS)]
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
                if gate_inclusive:
                    gate = (c.detach()[...,D:] * c.grad[...,D:] * m_).abs().sum(dim=(0,1))
                    scores[i] += content.cpu() + gate.cpu()
                else:
                    scores[i] += content.cpu()
        n_real += int(mask.sum().item()); captured.clear()
    for h in handles: h.remove()
    del wrapped; torch.cuda.empty_cache() if DEVICE=="cuda" else None
    return (scores / max(n_real, 1)).detach()

def compute_gradmag(gate_inclusive):
    """Taylor importance scores — content-only or gate-inclusive."""
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
        w = block.mixer.in_proj.weight.data   # [2*D, hidden]
        g = block.mixer.in_proj.weight.grad   # [2*D, hidden]
        if g is not None:
            content = (w[:D] * g[:D]).abs().mean(dim=1)
            if gate_inclusive:
                gate = (w[D:] * g[D:]).abs().mean(dim=1)
                scores[i] = (content + gate).cpu()
            else:
                scores[i] = content.cpu()
        else:
            scores[i] = w[:D].abs().mean(dim=1).cpu()
    del model; torch.cuda.empty_cache() if DEVICE=="cuda" else None
    return scores.detach()

def compute_activation(gate_inclusive):
    """Mean absolute activation — content-only or gate-inclusive."""
    wrapped = ModifiedMambaForCausalLM(base_model, is_fast_forward_available=False)
    wrapped.to(DEVICE).eval()
    scores = torch.zeros(NUM_LAYERS, D)
    n_real, captured = 0, {}
    def make_hook(i):
        def h(m, inp, out): captured[i] = out.detach()
        return h
    handles = [wrapped.backbone.layers[i].mixer.in_proj
               .register_forward_hook(make_hook(i)) for i in range(NUM_LAYERS)]
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
                m_ = mask.unsqueeze(-1).float()
                content = (c[...,:D] * m_).abs().sum(dim=(0,1))
                if gate_inclusive:
                    gate = (c[...,D:] * m_).abs().sum(dim=(0,1))
                    scores[i] += content.cpu() + gate.cpu()
                else:
                    scores[i] += content.cpu()
        n_real += int(mask.sum().item()); captured.clear()
    for h in handles: h.remove()
    del wrapped; torch.cuda.empty_cache() if DEVICE=="cuda" else None
    return (scores / max(n_real, 1)).detach()

# ── Load or compute all scores ─────────────────────────────────────────────────
sep("COMPUTING / LOADING SCORES")

def load_or_compute(cache_path, fn, label):
    if os.path.exists(cache_path):
        s = torch.load(cache_path)
        print(f"  {label:<35} loaded from cache")
        return s
    print(f"  {label:<35} computing...", end=" ", flush=True)
    t0 = time.time(); s = fn()
    torch.save(s, cache_path)
    print(f"{time.time()-t0:.1f}s")
    return s

LRP_CO  = load_or_compute(LRP_CO_CACHE,  lambda: compute_lrp(False),        "LRP content-only")
LRP_GI  = load_or_compute(LRP_GI_CACHE,  lambda: compute_lrp(True),         "LRP gate-inclusive")
GM_CO   = load_or_compute(GM_CO_CACHE,   lambda: compute_gradmag(False),     "GradMag content-only")
GM_GI   = load_or_compute(GM_GI_CACHE,   lambda: compute_gradmag(True),      "GradMag gate-inclusive")
ACT_CO  = load_or_compute(ACT_CO_CACHE,  lambda: compute_activation(False),  "Activation content-only")
ACT_GI  = load_or_compute(ACT_GI_CACHE,  lambda: compute_activation(True),   "Activation gate-inclusive")
RND     = torch.rand(NUM_LAYERS, D, generator=torch.Generator().manual_seed(42))

# ── Ranking comparison ─────────────────────────────────────────────────────────
sep("RANKING COMPARISON — GATE EFFECT PER METHOD")
print("""
Key question: how much does adding the gate change channel rankings within
each method? And how do the gate-inclusive methods compare to each other?
""")

print("  Within-method gate effect (content-only vs gate-inclusive):")
pairs_within = [
    ("LRP",        LRP_CO,  LRP_GI),
    ("GradMag",    GM_CO,   GM_GI),
    ("Activation", ACT_CO,  ACT_GI),
]
for name, co, gi in pairs_within:
    rho, _ = spearmanr(co.flatten().numpy(), gi.flatten().numpy())
    layer_rhos = [spearmanr(co[i].numpy(), gi[i].numpy())[0]
                  for i in range(NUM_LAYERS)]
    min_rho = min(layer_rhos)
    low_layers = [i for i, r in enumerate(layer_rhos) if r < 0.70]
    print(f"    {name:<12}: global ρ={rho:.4f}, min per-layer ρ={min_rho:.4f}, "
          f"layers ρ<0.70: {low_layers}")

print("\n  Cross-method gate-inclusive rankings (do all GI methods agree?):")
gi_pairs = [
    ("LRP-GI",    "GradMag-GI",    LRP_GI,  GM_GI),
    ("LRP-GI",    "Activation-GI", LRP_GI,  ACT_GI),
    ("GradMag-GI","Activation-GI", GM_GI,   ACT_GI),
    ("LRP-GI",    "LRP-CO",        LRP_GI,  LRP_CO),
    ("GradMag-GI","GradMag-CO",    GM_GI,   GM_CO),
]
for na, nb, sa, sb in gi_pairs:
    rho, _ = spearmanr(sa.flatten().numpy(), sb.flatten().numpy())
    print(f"    {na:<16} vs {nb:<16}: ρ={rho:.4f}")

# ── Perplexity comparison ──────────────────────────────────────────────────────
sep("ONE-SHOT AND POST-FT PERPLEXITY — ALL METHODS")
print(f"  Base model PPL: ", end=""); base_ppl = ppl(base_model, eval_tok)
print(f"{base_ppl:.2f}")

all_scores = {
    "LRP-CO":     LRP_CO,
    "LRP-GI":     LRP_GI,
    "GradMag-CO": GM_CO,
    "GradMag-GI": GM_GI,
    "Act-CO":     ACT_CO,
    "Act-GI":     ACT_GI,
    "Random":     RND,
}

results = {name: {} for name in all_scores}
for sp in SPARSITIES:
    kr = 1.0 - sp
    print(f"\n  ── Sparsity {sp*100:.0f}% ──")
    for name, scores in all_scores.items():
        print(f"    {name:<12} ...", end=" ", flush=True)
        t0  = time.time()
        m   = prune_model(base_model, scores, kr)
        pre = ppl(m, eval_tok)
        m   = fine_tune(m, FT_STEPS)
        post = ppl(m, eval_tok)
        results[name][sp] = {"pre": pre, "post": post}
        print(f"pre={pre:.2f}  post={post:.2f}  ({time.time()-t0:.0f}s)")
        del m; torch.cuda.empty_cache() if DEVICE=="cuda" else None

# ── Results tables ─────────────────────────────────────────────────────────────
sep("RESULTS")

print(f"\n  {'Method':<14}", end="")
for sp in SPARSITIES:
    print(f"  {sp*100:.0f}% pre  {sp*100:.0f}% post", end="")
print()
print("  " + "-"*80)
for name in all_scores:
    print(f"  {name:<14}", end="")
    for sp in SPARSITIES:
        r = results[name][sp]
        print(f"  {r['pre']:>8.2f}  {r['post']:>9.2f}", end="")
    print()

# Winner per sparsity level
print(f"\n  Post-FT winner per sparsity:")
for sp in SPARSITIES:
    ranked = sorted(all_scores.keys(), key=lambda n: results[n][sp]["post"])
    top3 = [(n, results[n][sp]["post"]) for n in ranked[:3]]
    print(f"    {sp*100:.0f}%: " + "  →  ".join(f"{n}={v:.2f}" for n, v in top3))

# The key diagnostic: gate-inclusive GradMag vs gate-inclusive LRP
sep("KEY DIAGNOSTIC: IS IT LRP OR GATE INCLUSION?")
print()
for sp in SPARSITIES:
    lrp_gi  = results["LRP-GI"][sp]["post"]
    gm_gi   = results["GradMag-GI"][sp]["post"]
    lrp_co  = results["LRP-CO"][sp]["post"]
    gm_co   = results["GradMag-CO"][sp]["post"]
    act_gi  = results["Act-GI"][sp]["post"]

    print(f"  Sparsity {sp*100:.0f}%:")
    print(f"    LRP-GI={lrp_gi:.2f}  GradMag-GI={gm_gi:.2f}  "
          f"Act-GI={act_gi:.2f}")
    print(f"    GI improvement — LRP: {lrp_co-lrp_gi:+.2f}  "
          f"GradMag: {gm_co-gm_gi:+.2f}")
    if lrp_gi < gm_gi:
        margin = (gm_gi - lrp_gi) / gm_gi * 100
        print(f"    LRP-GI beats GradMag-GI by {margin:.1f}%  "
              f"→ LRP corrections add value beyond gate inclusion")
    else:
        margin = (lrp_gi - gm_gi) / lrp_gi * 100
        print(f"    GradMag-GI beats LRP-GI by {margin:.1f}%  "
              f"→ gate inclusion alone explains H10; LRP corrections irrelevant for pruning")
    print()

print("INTERPRETATION GUIDE:")
print("  If LRP-GI >> GradMag-GI: LRP's conservation corrections are the key")
print("    contribution. Gate inclusion helps both, but LRP's causal attribution")
print("    selects fundamentally better channels.")
print("  If LRP-GI ≈ GradMag-GI: The architectural insight (include the gate)")
print("    is the contribution. Any gradient-based method benefits equally.")
print("    The project contribution shifts from 'LRP is better' to 'gate path")
print("    is critical for channel scoring in Mamba — here's why and how'.")
print("  Either result is a clear, publishable finding.")
