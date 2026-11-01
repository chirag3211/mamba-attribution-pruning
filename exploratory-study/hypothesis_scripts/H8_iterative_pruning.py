"""
H8: Iterative Pruning vs One-Shot Pruning
==========================================
Claim: Pruning in two rounds — prune to intermediate sparsity, fine-tune,
re-score on the pruned model, prune to final sparsity, fine-tune again —
produces lower final perplexity than one-shot pruning at the same target
sparsity with the same total fine-tuning budget.

Key principle: iterative pruning allows each round's fine-tuning to adjust
weights before the next round's relevance scores are computed. The scores
therefore reflect the current model's behaviour, not the original dense
model's behaviour. This is especially important for LRP, where relevance
depends on the actual forward-pass computation.

Schedule (geometric spacing):
  Round 1: prune to keep_ratio^(1/2) of channels (e.g., 70% → keep 0.548)
  Fine-tune: budget/2 steps
  Re-score: run LRP on the pruned-and-fine-tuned model
  Round 2: prune to keep_ratio of channels from the round-1 model
  Fine-tune: budget/2 steps

Tested at 50% and 70% sparsity. Each round uses half the total FT budget,
so the comparison against one-shot is at matched total fine-tuning steps.

Runtime: ~30 minutes on T4 (expensive due to double fine-tuning + re-scoring).
"""

import sys, os, copy, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/content/MambaLRP")

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME    = "state-spaces/mamba-130m-hf"
DATASET       = "Salesforce/wikitext"
DATASET_CFG   = "wikitext-2-raw-v1"
PPL_CHUNK     = 256
PPL_N         = 40
FT_BUDGET     = 30      # total steps — split equally between rounds
FT_LR         = 2e-4
FT_BATCH      = 4
FT_SEQ        = 128
TARGET_SPS    = [0.50, 0.70]
CALIB_SEQ     = 128
CALIB_N       = 128     # smaller calib for re-scoring speed
SCORE_CACHE   = "/content/lrp_scores_256.pt"

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
print(f"Model: {NUM_LAYERS} layers, D={D}")

wiki_test  = load_dataset(DATASET, DATASET_CFG, split="test")
wiki_train = load_dataset(DATASET, DATASET_CFG, split="train")
calib_raw  = load_dataset(DATASET, DATASET_CFG, split="train")
def build_tokens(ds, max_t):
    text = " ".join(t for t in ds["text"] if len(t.strip()) > 20)
    return tokenizer(text, return_tensors="pt")["input_ids"][0][:max_t]
eval_tok   = build_tokens(wiki_test,  PPL_CHUNK * PPL_N + 1)
train_tok  = build_tokens(wiki_train, FT_SEQ * 200)
CALIB      = [t for t in calib_raw["text"] if len(t.strip()) > 80][:CALIB_N]
print(f"Data ready.")

# ── Helpers ────────────────────────────────────────────────────────────────────
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

def prune(src, scores, keep_ratio):
    m = copy.deepcopy(src).to(DEVICE); m.eval()
    n_keep = int(round(scores.shape[1] * keep_ratio))  # FIX: use actual channel count from scores, not global D
    for i, block in enumerate(m.backbone.layers):
        mx = block.mixer
        _, idx = torch.topk(scores[i], k=n_keep, largest=True)
        keep = idx.sort().values.to(DEVICE)
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
    return m

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

def fine_tune(model, steps):
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

def compute_lrp_scores_on(model, calib_texts):
    """Compute LRP scores on an arbitrary model (possibly pruned)."""
    current_D = model.backbone.layers[0].mixer.intermediate_size
    wrapped = ModifiedMambaForCausalLM(model, is_fast_forward_available=False)
    wrapped.to(DEVICE).eval()
    scores = torch.zeros(NUM_LAYERS, current_D)
    n_real, captured = 0, {}
    def make_hook(i):
        def h(m, inp, out): out.retain_grad(); captured[i] = out
        return h
    handles = [wrapped.backbone.layers[i].mixer.in_proj.register_forward_hook(make_hook(i))
               for i in range(NUM_LAYERS)]
    for start in range(0, len(calib_texts), 4):
        batch = calib_texts[start:start+4]
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
                scores[i] += (c.detach()[...,:current_D] * c.grad[...,:current_D]
                              * mask.unsqueeze(-1).float()).abs().sum(dim=(0,1)).cpu()
        n_real += int(mask.sum().item()); captured.clear()
    for h in handles: h.remove()
    del wrapped; torch.cuda.empty_cache() if DEVICE=="cuda" else None
    return (scores / max(n_real, 1)).detach()

# ── Load baseline LRP scores ───────────────────────────────────────────────────
sep("LOADING SCORES")
if os.path.exists(SCORE_CACHE):
    LRP_SCORES = torch.load(SCORE_CACHE)
    print(f"Loaded from {SCORE_CACHE}")
else:
    print("Computing LRP scores...", end=" ", flush=True)
    t0 = time.time()
    LRP_SCORES = compute_lrp_scores_on(base_model, CALIB)
    torch.save(LRP_SCORES, SCORE_CACHE)
    print(f"{time.time()-t0:.1f}s")

base_ppl_val = ppl(base_model, eval_tok)
print(f"Base model PPL: {base_ppl_val:.2f}")

# ── H8: Iterative vs one-shot ─────────────────────────────────────────────────
sep("H8: ITERATIVE vs ONE-SHOT PRUNING")
print(f"""
For each target sparsity:
  One-shot  : prune once to target → fine-tune {FT_BUDGET} steps
  Iterative : prune to sqrt(keep_ratio) → fine-tune {FT_BUDGET//2} steps
              → re-score on pruned model → prune to target → fine-tune {FT_BUDGET//2} steps
Total FT budget is identical ({FT_BUDGET} steps). Comparison on post-FT PPL.
""")

results = {}
for sp in TARGET_SPS:
    kr = 1.0 - sp
    mid_kr = kr ** 0.5   # geometric midpoint
    half   = FT_BUDGET // 2
    print(f"\n{'─'*60}")
    print(f"  Target sparsity {sp*100:.0f}%  "
          f"(final keep_ratio={kr:.3f}, midpoint={mid_kr:.3f})")
    print(f"{'─'*60}")

    # One-shot
    print(f"\n  ONE-SHOT:")
    print(f"    Pruning to {sp*100:.0f}%...", end=" ", flush=True)
    t0 = time.time()
    m1 = prune(base_model, LRP_SCORES, kr)
    ppl_pre = ppl(m1, eval_tok)
    print(f"PPL(pre-FT)={ppl_pre:.2f}  ({time.time()-t0:.1f}s)")
    print(f"    Fine-tuning {FT_BUDGET} steps...", end=" ", flush=True)
    t0 = time.time()
    m1 = fine_tune(m1, FT_BUDGET)
    ppl_oneshot = ppl(m1, eval_tok)
    print(f"PPL(post-FT)={ppl_oneshot:.2f}  ({time.time()-t0:.0f}s)")

    # Iterative
    print(f"\n  ITERATIVE (Round 1 — prune to midpoint {mid_kr:.3f}):")
    print(f"    Pruning...", end=" ", flush=True)
    t0 = time.time()
    m2 = prune(base_model, LRP_SCORES, mid_kr)
    ppl_r1 = ppl(m2, eval_tok)
    print(f"PPL={ppl_r1:.2f}  ({time.time()-t0:.1f}s)")
    print(f"    Fine-tuning {half} steps...", end=" ", flush=True)
    t0 = time.time()
    m2 = fine_tune(m2, half)
    ppl_r1_ft = ppl(m2, eval_tok)
    print(f"PPL={ppl_r1_ft:.2f}  ({time.time()-t0:.0f}s)")

    print(f"\n  ITERATIVE (Round 2 — re-score + prune to final {kr:.3f}):")
    print(f"    Re-scoring on pruned model ({CALIB_N} sequences)...", end=" ", flush=True)
    t0 = time.time()
    scores_r2 = compute_lrp_scores_on(m2, CALIB)
    print(f"{time.time()-t0:.1f}s")

    final_keep   = int(round(D * kr))                              # absolute target channel count
    actual_mid_d = m2.backbone.layers[0].mixer.intermediate_size   # FIX: actual post-round-1 size
    round2_kr    = final_keep / actual_mid_d                       # fraction of current channels
    round2_kr    = max(0.05, min(0.99, round2_kr))
    print(f"    Pruning round-2 model to {round2_kr:.3f} of remaining channels...",
          end=" ", flush=True)
    t0 = time.time()
    m2 = prune(m2, scores_r2, round2_kr)
    ppl_r2 = ppl(m2, eval_tok)
    print(f"PPL={ppl_r2:.2f}  ({time.time()-t0:.1f}s)")
    print(f"    Fine-tuning {half} steps...", end=" ", flush=True)
    t0 = time.time()
    m2 = fine_tune(m2, half)
    ppl_iterative = ppl(m2, eval_tok)
    print(f"PPL={ppl_iterative:.2f}  ({time.time()-t0:.0f}s)")

    results[sp] = {"oneshot": ppl_oneshot, "iterative": ppl_iterative,
                   "oneshot_pre": ppl_pre, "r1_post": ppl_r1_ft, "r2_pre": ppl_r2}
    del m1, m2; torch.cuda.empty_cache() if DEVICE=="cuda" else None

# ── Summary ────────────────────────────────────────────────────────────────────
sep("RESULTS")
print(f"\n  {'Sparsity':<10} {'OneShot pre':>12} {'OneShot post':>13} "
      f"{'Iter R1 post':>13} {'Iter R2 pre':>12} {'Iterative post':>15} {'Better':>8}")
print("  " + "-"*86)
iter_wins = 0
for sp in TARGET_SPS:
    r = results[sp]
    better = "Iterative" if r["iterative"] < r["oneshot"] else "One-shot"
    if better == "Iterative": iter_wins += 1
    print(f"  {sp*100:.0f}%{'':<7} {r['oneshot_pre']:>12.2f} {r['oneshot']:>13.2f} "
          f"{r['r1_post']:>13.2f} {r['r2_pre']:>12.2f} {r['iterative']:>15.2f} {better:>8}")

h8_verdict = "SUPPORTED" if iter_wins == len(TARGET_SPS) else "PARTIAL" if iter_wins > 0 else "NOT SUPPORTED"
print(f"\nRESULT  Iterative better in {iter_wins}/{len(TARGET_SPS)} cases  →  H8 {h8_verdict}")
print(f"\nNote: Iterative pruning re-scores on the already-pruned+fine-tuned model,")
print(f"allowing LRP to reflect the current model's activation patterns rather")
print(f"than the original dense model's. This is the key advantage — relevance")
print(f"scores shift after the first round of fine-tuning.")
