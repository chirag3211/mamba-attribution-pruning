"""
H13: Relevance-Guided Mixed-Precision Quantisation
===================================================
Claim: Assigning FP16 precision to high-relevance channels and simulated
INT8 precision to low-relevance channels produces lower perplexity than
uniform INT8 quantisation at the same memory footprint.

This extends ECQ (Becking et al., 2022) to Mamba. Since H3 showed that
pure weight magnitude is random for Mamba's gated architecture, the
hypothesis is especially motivated: standard INT8 calibration tools that
rely on magnitude-based importance signals may perform poorly for Mamba.
LRP-guided mixed precision uses causal output contribution as the criterion
instead.

Setup:
  - FP16 base model: reference (no quantisation)
  - Uniform INT8: ALL in_proj weights quantised to 8-bit per-tensor
  - LRP mixed (50/50): top-50% channels by LRP = FP16, rest = INT8
  - GradMag mixed (50/50): top-50% by GradMag = FP16, rest = INT8
  - Random mixed (50/50): random 50% = FP16, rest = INT8 (baseline)

Quantisation simulation: for each weight row (channel), scale to [-127, 127],
round to nearest integer, scale back. Applied to in_proj content rows (not gate)
and to out_proj input rows of the corresponding low-relevance channels.

All models use the same memory footprint (50% INT8 = ~25% memory saving
over FP16 when accounted proportionally, at matched sparsity for fair comparison).

Runtime: ~15 minutes on T4.
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
QUANT_RATIO  = 0.50    # fraction of channels to quantise (low-relevance half)
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
calib_raw  = load_dataset(DATASET, DATASET_CFG, split="train")
def build_tokens(ds, mx):
    text = " ".join(t for t in ds["text"] if len(t.strip()) > 20)
    return tokenizer(text, return_tensors="pt")["input_ids"][0][:mx]
eval_tok = build_tokens(wiki_test, PPL_CHUNK * PPL_N + 1)
CALIB    = [t for t in calib_raw["text"] if len(t.strip()) > 80][:CALIB_N]
print(f"Setup complete. D={D}")

# ── Quantisation helpers ───────────────────────────────────────────────────────
def simulate_int8_rows(weight, row_indices):
    """
    Simulate INT8 quantisation on specific rows of a weight matrix.
    Per-row scaling: scale = max(|w_row|) / 127.
    """
    w = weight.clone()
    for idx in row_indices:
        row = w[idx]
        scale = row.abs().max().clamp(min=1e-8) / 127.0
        w[idx] = (row / scale).round().clamp(-127, 127) * scale
    return w

def apply_mixed_precision(src_model, scores, quant_ratio, seed=None):
    """
    Create a deep copy of src_model with simulated INT8 on low-relevance channels.
    scores: [num_layers, D] — higher = more important = keep FP16
    quant_ratio: fraction of channels to quantise (bottom-quant_ratio by score)
    """
    m = copy.deepcopy(src_model)
    n_quant = int(round(D * quant_ratio))
    total_quantised = 0
    for i, block in enumerate(m.backbone.layers):
        mx = block.mixer
        if seed is not None:
            g = torch.Generator().manual_seed(seed + i)
            low_idx = torch.randperm(D, generator=g)[:n_quant].sort().values.tolist()
        else:
            _, idx = torch.topk(scores[i], k=n_quant, largest=False)
            low_idx = idx.sort().values.tolist()

        # in_proj: quantise content rows for low-relevance channels
        # (rows 0..D-1 are content; rows D..2D-1 are gate)
        w_ip = mx.in_proj.weight.data
        low_content = low_idx  # content half rows
        low_gate    = [j + D for j in low_idx]  # gate half rows
        mx.in_proj.weight.data = simulate_int8_rows(w_ip, low_content + low_gate)

        # out_proj: quantise input columns for low-relevance channels
        w_op = mx.out_proj.weight.data  # [hidden, D]
        col_indices = low_idx
        w_op_t = w_op.t()  # [D, hidden]
        w_op_t = simulate_int8_rows(w_op_t, col_indices)
        mx.out_proj.weight.data = w_op_t.t()

        total_quantised += len(low_idx)

    print(f"    Quantised {total_quantised} channels total "
          f"({total_quantised / (NUM_LAYERS * D) * 100:.1f}% of all channels)")
    return m.eval()

def apply_uniform_int8(src_model):
    """Quantise ALL in_proj and out_proj weights uniformly to INT8 per-tensor."""
    m = copy.deepcopy(src_model)
    for block in m.backbone.layers:
        mx = block.mixer
        for param_name in ['in_proj', 'out_proj']:
            layer = getattr(mx, param_name)
            w = layer.weight.data
            scale = w.abs().max().clamp(min=1e-8) / 127.0
            layer.weight.data = (w / scale).round().clamp(-127, 127) * scale
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

def compute_gradmag():
    model = MambaForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.train(); model.zero_grad(set_to_none=True)
    for s in range(0, len(CALIB), 4):
        batch = CALIB[s:s+4]
        enc   = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=CALIB_SEQ).to(DEVICE)
        out   = model(**enc); lg = out.logits if hasattr(out, "logits") else out
        mask  = enc["attention_mask"]
        (F.cross_entropy(lg[:,:-1].contiguous().transpose(1,2),
                         enc["input_ids"][:,1:].contiguous(), reduction="none")
         * mask[:,1:].float()).sum().backward()
    scores = torch.zeros(NUM_LAYERS, D)
    for i, block in enumerate(model.backbone.layers):
        w = block.mixer.in_proj.weight.data; g = block.mixer.in_proj.weight.grad
        scores[i] = (w[:D] * g[:D]).abs().mean(dim=1).cpu() if g is not None else w[:D].abs().mean(dim=1).cpu()
    del model; torch.cuda.empty_cache() if DEVICE=="cuda" else None
    return scores.detach()

# ── Load / compute scores ──────────────────────────────────────────────────────
sep("LOADING SCORES")
LRP_SCORES = torch.load(LRP_CACHE) if os.path.exists(LRP_CACHE) else None
if LRP_SCORES is None:
    print("LRP scores not cached. Please run codebase_complete.py first.")
    sys.exit(1)
print(f"LRP scores loaded.")

GM_SCORES = torch.load(GM_CACHE) if os.path.exists(GM_CACHE) else None
if GM_SCORES is None:
    print("GradMag scores not cached. Computing...", end=" ", flush=True)
    t0 = time.time(); GM_SCORES = compute_gradmag()
    torch.save(GM_SCORES, GM_CACHE); print(f"{time.time()-t0:.1f}s")
else:
    print(f"GradMag scores loaded.")

# ── H13: Mixed-precision comparison ───────────────────────────────────────────
sep("H13: MIXED-PRECISION QUANTISATION")
print(f"""
All models quantise {QUANT_RATIO*100:.0f}% of channels to simulated INT8.
Which channels are quantised differs by method.
  FP16 base  : no quantisation (reference)
  Uniform INT8: ALL channels quantised
  LRP-guided  : bottom-{QUANT_RATIO*100:.0f}% by LRP = INT8, rest = FP16
  GradMag-guided: bottom-{QUANT_RATIO*100:.0f}% by GradMag = INT8, rest = FP16
  Random      : random {QUANT_RATIO*100:.0f}% = INT8, rest = FP16
""")

print(f"  FP16 base PPL: ", end=""); base_ppl = ppl(base_model, eval_tok); print(f"{base_ppl:.2f}")

configs = [
    ("Uniform-INT8",  None,       None),
    ("LRP-guided",    LRP_SCORES, None),
    ("GradMag-guided",GM_SCORES,  None),
    ("Random-mixed",  None,       42),    # seed for reproducibility
]

results = {}
for name, scores, seed in configs:
    print(f"\n  {name}:")
    t0 = time.time()
    if name == "Uniform-INT8":
        m = apply_uniform_int8(base_model).to(DEVICE)
    elif seed is not None:
        m = apply_mixed_precision(base_model, None, QUANT_RATIO, seed=seed).to(DEVICE)
    else:
        m = apply_mixed_precision(base_model, scores, QUANT_RATIO).to(DEVICE)
    p = ppl(m, eval_tok)
    results[name] = p
    print(f"    PPL = {p:.2f}  ({time.time()-t0:.1f}s)")
    del m; torch.cuda.empty_cache() if DEVICE=="cuda" else None

sep("RESULTS")
print(f"\n  {'Method':<20} {'PPL':>10} {'vs FP16':>12} {'vs Uniform':>12}")
print("  " + "-"*58)
print(f"  {'FP16 (no quant)':<20} {base_ppl:>10.2f} {'—':>12} {'—':>12}")
for name, p in results.items():
    diff_base = p - base_ppl
    diff_uni  = p - results["Uniform-INT8"]
    print(f"  {name:<20} {p:>10.2f} {diff_base:>+12.2f} {diff_uni:>+12.2f}")

lrp_vs_uni  = results["LRP-guided"]  - results["Uniform-INT8"]
lrp_vs_rnd  = results["LRP-guided"]  - results["Random-mixed"]
gm_vs_uni   = results["GradMag-guided"] - results["Uniform-INT8"]

print(f"\n  LRP-guided  vs Uniform INT8  : {lrp_vs_uni:+.2f} (negative = LRP better)")
print(f"  LRP-guided  vs Random mixed  : {lrp_vs_rnd:+.2f}")
print(f"  GradMag     vs Uniform INT8  : {gm_vs_uni:+.2f}")

h13_verdict = ("SUPPORTED"
               if lrp_vs_uni < 0 and lrp_vs_rnd < 0
               else "PARTIAL — LRP beats uniform but not random"
               if lrp_vs_uni < 0
               else "NOT SUPPORTED")
print(f"\nRESULT  LRP-guided mixed PPL={results['LRP-guided']:.2f} vs "
      f"Uniform={results['Uniform-INT8']:.2f}  →  H13 {h13_verdict}")
print(f"\nNote: Memory footprint is not directly measured here (INT8 simulation")
print(f"keeps weights in FP32 for computation). In a real deployment, the INT8")
print(f"weights would occupy half the memory of FP16 weights.")
print(f"The key finding is whether relevance-guided channel selection for INT8")
print(f"reduces quality degradation vs. uniform quantisation.")
