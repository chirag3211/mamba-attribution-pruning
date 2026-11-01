"""
H11: Cross-Entropy vs Max-Logit LRP Seeding
=============================================
Claim: Cross-entropy seeding (relevance propagated from the ground-truth
next-token loss) produces better-structured channel rankings and lower
perplexity after pruning than max-logit seeding (relevance propagated from
the model's top predicted token).

Background: MambaLRP's relevance propagation starts from a scalar output.
  - Cross-entropy seeding: target = sum of per-token CE loss weighted by mask.
    Attribution reflects which channels help predict the *correct* next token.
  - Max-logit seeding: target = sum of max logits at the last position.
    Attribution reflects which channels drive the model's *most confident*
    prediction regardless of correctness.

For pruning, we want channels important for correct predictions, not just
confident ones. CE seeding is theoretically preferable and was used throughout
the main experiment. This script verifies that choice empirically.

Additionally: for causal LM, max-logit seeding at only the *last* token
position means channels important for mid-sequence predictions receive no
relevance signal. CE seeding distributes relevance over all positions, which
may produce more robust channel rankings.

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
SPARSITIES  = [0.30, 0.50]
CALIB_N     = 256
CALIB_SEQ   = 128
CE_CACHE    = "/content/lrp_scores_256.pt"         # cross-entropy (main experiment)
ML_CACHE    = "/content/lrp_maxlogit_scores.pt"    # max-logit

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

def compute_lrp_scores(seeding):
    """seeding: 'cross_entropy' or 'max_logit'"""
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
        if seeding == "cross_entropy":
            target = (F.cross_entropy(logits[:,:-1].contiguous().transpose(1,2),
                                      enc["input_ids"][:,1:].contiguous(),
                                      reduction="none") * mask[:,1:].float()).sum()
        else:  # max_logit
            target = logits[:, -1, :].max(dim=-1).values.sum()
        target.backward()
        for i in range(NUM_LAYERS):
            c = captured.get(i)
            if c is not None and c.grad is not None:
                scores[i] += (c.detach()[...,:D] * c.grad[...,:D]
                              * mask.unsqueeze(-1).float()).abs().sum(dim=(0,1)).cpu()
        n_real += int(mask.sum().item()); captured.clear()
    for h in handles: h.remove()
    del wrapped; torch.cuda.empty_cache() if DEVICE=="cuda" else None
    return (scores / max(n_real, 1)).detach()

# ── Load / compute ─────────────────────────────────────────────────────────────
sep("COMPUTING SCORES")
CE_SCORES = torch.load(CE_CACHE) if os.path.exists(CE_CACHE) else None
if CE_SCORES is None:
    print("Computing cross-entropy scores...", end=" ", flush=True)
    t0 = time.time(); CE_SCORES = compute_lrp_scores("cross_entropy")
    torch.save(CE_SCORES, CE_CACHE); print(f"{time.time()-t0:.1f}s")
else:
    print(f"Cross-entropy scores loaded from {CE_CACHE}")

ML_SCORES = torch.load(ML_CACHE) if os.path.exists(ML_CACHE) else None
if ML_SCORES is None:
    print("Computing max-logit scores...", end=" ", flush=True)
    t0 = time.time(); ML_SCORES = compute_lrp_scores("max_logit")
    torch.save(ML_SCORES, ML_CACHE); print(f"{time.time()-t0:.1f}s")
else:
    print(f"Max-logit scores loaded from {ML_CACHE}")

# ── Ranking comparison ─────────────────────────────────────────────────────────
sep("H11: RANKING COMPARISON")
global_rho, _ = spearmanr(CE_SCORES.flatten().numpy(), ML_SCORES.flatten().numpy())
print(f"\n  Global Spearman ρ (CE vs max-logit): {global_rho:.4f}")
print(f"\n  Per-layer ρ:")
low_agreement = []
for i in range(NUM_LAYERS):
    rho_i, _ = spearmanr(CE_SCORES[i].numpy(), ML_SCORES[i].numpy())
    bar  = "█" * int(rho_i * 20)
    flag = " ← disagree" if rho_i < 0.80 else ""
    if rho_i < 0.80: low_agreement.append(i)
    print(f"    Layer {i:>2}: {rho_i:.4f} {bar}{flag}")

print(f"\n  Layers with ρ < 0.80: {low_agreement}")

# ── Perplexity comparison ──────────────────────────────────────────────────────
sep("PERPLEXITY COMPARISON")
base_ppl_val = ppl(base_model, eval_tok)
print(f"  Base PPL: {base_ppl_val:.2f}")
results = {}
for sp in SPARSITIES:
    kr = 1.0 - sp; results[sp] = {}
    print(f"\n  Sparsity {sp*100:.0f}%:")
    for name, scores in [("CE seeding", CE_SCORES), ("Max-logit", ML_SCORES)]:
        print(f"    {name:<14} ...", end=" ", flush=True)
        t0 = time.time()
        m = prune_model(base_model, scores, kr)
        p_pre = ppl(m, eval_tok)
        m = fine_tune(m, FT_STEPS)
        p_post = ppl(m, eval_tok)
        results[sp][name] = {"pre": p_pre, "post": p_post}
        print(f"pre={p_pre:.2f}  post={p_post:.2f}  ({time.time()-t0:.0f}s)")
        del m; torch.cuda.empty_cache() if DEVICE=="cuda" else None

sep("RESULTS")
print(f"\n  {'Sparsity':<10} {'CE pre':>10} {'CE post':>10} {'ML pre':>10} {'ML post':>10} {'Better':>10}")
print("  " + "-"*64)
ce_wins = 0
for sp in SPARSITIES:
    r = results[sp]
    winner = "CE" if r["CE seeding"]["post"] < r["Max-logit"]["post"] else "Max-logit"
    if winner == "CE": ce_wins += 1
    print(f"  {sp*100:.0f}%{'':<7} "
          f"{r['CE seeding']['pre']:>10.2f} {r['CE seeding']['post']:>10.2f} "
          f"{r['Max-logit']['pre']:>10.2f} {r['Max-logit']['post']:>10.2f} {winner:>10}")

h11_verdict = ("SUPPORTED" if ce_wins == len(SPARSITIES)
               else "PARTIAL" if ce_wins > 0
               else "NOT SUPPORTED — max-logit wins")
print(f"\n  Global ρ between seedings: {global_rho:.4f}")
if global_rho > 0.95:
    print(f"  High ρ: seeding choice has minimal impact on channel selection.")
elif global_rho > 0.80:
    print(f"  Moderate ρ: seedings agree mostly but differ in {len(low_agreement)} layers.")
else:
    print(f"  Low ρ: seedings produce substantially different channel rankings.")
print(f"\nRESULT  CE seeding wins in {ce_wins}/{len(SPARSITIES)} cases  →  H11 {h11_verdict}")
