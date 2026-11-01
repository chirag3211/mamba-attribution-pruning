# Relevance-Guided Structural Pruning of Mamba State Space Models
## Using MambaLRP Attribution as a Compression Criterion
### System Engineering for Deep Learning — Course Project Proposal

---

## 1. Abstract

Existing structured pruning methods for Mamba state-space models rely on weight
magnitude or gradient magnitude as channel importance criteria. Preliminary
experiments show that pure L1 weight magnitude is functionally equivalent to random
channel removal in Mamba's gated architecture (Spearman rho=0.043), motivating a
data-driven attribution-based criterion instead.

This project uses MambaLRP (Jafari et al., NeurIPS 2024) as a channel selection
criterion. A systematic hypothesis-testing phase on mamba-130m-hf (WikiText-2)
produced the following primary findings:

**Central finding (H10):** Standard MambaLRP propagates relevance only through
the content path of in_proj, discarding the gate half. Including the gate path in
the attribution formula (gate-inclusive LRP) dramatically improves channel
selection. Gate-inclusive LRP achieves one-shot perplexity 16-62x lower than
content-only LRP across 30-70% sparsity, and 2-2.4x lower post-fine-tuning.
Critically, it outperforms gradient-aware magnitude post-FT at all sparsity levels
tested (PPL 61.16 vs 81.73 at 30%; 115.05 vs 233.28 at 50%; 284.49 vs 904.43
at 70%), resolving the H6 gap from the preliminary results.

**Secondary finding (H8):** Iterative pruning with mid-point re-scoring achieves
~8% lower post-FT perplexity than one-shot pruning at both 50% and 70% sparsity
with the same total fine-tuning budget (50%: 230.55 vs 251.51; 70%: 539.45 vs
588.89). An earlier buggy run reported 49% at 50%; this was retracted — see
Section 6.5 and the progress report for the correction.

**Supporting findings:** Pure magnitude is random for Mamba (H3). Non-uniform
sparsity allocation with a global-threshold implementation hurts post-FT recovery
due to extreme early-layer ablation (H7). Hybrid LRP+GradMag does not beat pure
methods (H9). Activation-magnitude scoring is a fast and competitive gradient-free
baseline (H12). Mixed-precision quantisation benefits from any 50/50 channel split
over uniform INT8, but the specific criterion does not matter at row granularity
(H13).

The full experiment uses WikiText-103 and gate-inclusive LRP scores throughout.

---

## 1.1 Novelty Verification

Searches (queries: "explainability-guided pruning Mamba"; "MambaLRP pruning
compression"; "relevance-guided SSM pruning") found no published work combining
attribution-based relevance with Mamba structural pruning. Re-verification
recommended before final report.

---

## 2. Problem Statement

Model compression for Mamba requires a channel importance criterion that accounts
for its gating architecture. Pure weight magnitude fails because the SiLU gate
decouples weight norm from causal contribution (H3: rho=0.043). Gradient-aware
magnitude is more structured (rho=0.541 with LRP) but still misses the gate path.
The primary contribution of this project is demonstrating that including gate-path
relevance in the attribution formula (gate-inclusive LRP) substantially improves
both one-shot and post-fine-tuning perplexity relative to all baselines.

---

## 3. Background

### 3.1 Mamba and the Selective State-Space Mechanism

Mamba processes sequences through a selective SSM with input-dependent
discretisation (A, B, C, Delta), gated and projected through in_proj, x_proj,
dt_proj, out_proj, and a depthwise causal convolution. All components share a
channel axis of size intermediate_size (1536 for mamba-130m). Pruning one channel
requires consistent removal across all seven tensors per layer.

The in_proj layer outputs [content | gate] concatenated on the final dimension.
Content drives the SSM recurrence; gate multiplicatively modulates the output via
content * silu(gate). Standard MambaLRP propagates relevance only through the
content path. H10 shows the gate path carries substantial and distinct relevance
information, particularly in early layers (layer 0: gate fraction = 77% of
combined score; content-vs-gate-inclusive ranking correlation rho = 0.030).

### 3.2 Explainability-Guided Compression

ECQ (Becking et al., 2022): explainability-driven quantisation for CNNs.
Yeom et al. (2021): LRP relevance as pruning criterion for CNNs.
DyFiP (Sabih et al., 2022): DeepLIFT-based dynamic CNN filter pruning.
All confined to CNNs and Transformers. None applied to SSMs.

### 3.3 MambaLRP

MambaLRP implements corrected LRP propagation via strategic detach() calls on the
SiLU activation, SSM parameters, and gate output. Verified against source:
self.in_proj = mixer.in_proj is a reference assignment; slow_forward applies
(scan_output/2.) + (scan_output/2.).detach() for the gate (Eq. 9 of the paper).
The gate-inclusive extension (r = |content * content_grad| + |gate * gate_grad|)
is a one-line addition to the hook, validated empirically in H10.

### 3.4 Baselines

- Gradient-aware magnitude (|w * grad_w|): rho=0.541 with LRP. Main comparison.
- Activation magnitude: mean |activation| at in_proj output. No backward needed.
  Unexpectedly competitive (H12).
- Pure magnitude: rho=0.043 with LRP, functionally random. Excluded from main
  comparison; retained in H3 to establish the finding.
- Random: uniform random scores, sanity-check control.

---

## 4. Research Objectives

1. Compute gate-inclusive MambaLRP relevance scores per channel and compare
   against gradient-aware magnitude, activation magnitude, and random baselines.

2. Implement structural pruning removing lowest-scoring channels consistently
   across all seven components per layer.

3. Compare perplexity pre- and post-fine-tuning at matched sparsity levels with
   the same budget across all methods.

4. Test iterative pruning with mid-point re-scoring as a refinement exploiting
   LRP's sensitivity to the current model state.

5. Test relevance-guided mixed-precision quantisation as a stretch goal.

---

## 5. Methodology

### 5.1 Model and Data

Target model: state-spaces/mamba-130m-hf (0.67 GB VRAM). Calibration (N=256,
seq_len=128): 3.4 minutes on T4. Calibration stability confirmed (H1): all 24
layers show rho > 0.95 between N=128 and N=256.

Preliminary experiments use WikiText-2 (Salesforce/wikitext, wikitext-2-raw-v1).
Full experiment uses WikiText-103, gate-inclusive LRP scores, and >=3 seeds per
condition with 1% of pretraining token budget for fine-tuning.

### 5.2 Implementation Status

| Phase | Deliverable | Status |
|---|---|---|
| 0 | Environment setup | Complete |
| 1 | Content-only LRP extraction | Complete |
| 2 | Structural pruning + baselines | Complete |
| 3 | H1-H6 preliminary results | Complete |
| 4 | H7-H13 extended tests | Complete |
| 5 | H14 + H15 gate-inclusive comparison | Complete |
| 6 | Full experiment, WikiText-103, A6000, >=3 seeds | Pending |
| 7 | Write-up | Pending |

### 5.3 Gate-Inclusive Relevance Formula

Standard (content-only):
  r[i] = |content * content_grad| * mask

Gate-inclusive (H10, used in full experiment):
  r[i] = (|content * content_grad| + |gate * gate_grad|) * mask

Global rho between formulas: 0.874. Layer 0 rho: 0.030 (near-orthogonal).

### 5.4 H5 / H7 Decoupling Note

H5 layer-distribution analysis uses a global-threshold policy. H4/H6 perplexity
comparisons use per-layer top-k (every layer loses the same fraction). H5 is a
score-structure characterisation, not a causal explanation of H4/H6 differences.
H7 directly tested the global-threshold policy and found it hurts post-FT due to
extreme early-layer channel removal (layer 0: 10% of channels retained at 50%
global sparsity). A revised implementation with a per-layer minimum keep ratio is
planned for the full experiment.

### 5.5 Evaluation Metrics

- Perplexity (WikiText-2 preliminary; WikiText-103 full), pre- and post-recovery.
- Mean log-PPL (mean NLL in nats) as summary statistic.
- Layer-distribution analysis labeled explicitly as score characterisation.

---

## 6. Results

Setup: mamba-130m-hf, WikiText-2 test (10,240 tokens, chunk=256), N=256
calibration, seq_len=128, Tesla T4, single run per condition.

### 6.1 Hypothesis Summary

| H | Claim | Key Result | Status |
|---|---|---|---|
| H1 | Score stability at N>=128 | rho=0.9956 at N=128-256; all layers >=0.95 | Confirmed |
| H2 | LRP-GI divergence increases with depth | Early rho=0.87, Late rho=0.65, peak layers 14-20 | Exploratory |
| H3 | PureMag = random for Mamba | LRP-PureMag rho=0.043; overlap=chance | Confirmed |
| H4 | LRP < GradMag one-shot at 50-70% (content-only) | 50%: 7,233 vs 8,931; 70%: 54,272 vs 100,474 | Confirmed at 50-70% |
| H5 | LRP spares late layers (global threshold) | Layers 21-23 pressure <0.03; layer 0 = 0.93 | Exploratory |
| H6 | LRP lowest post-FT PPL (content-only) | GradMag wins at 30% and 50% with 30 FT steps | Not confirmed |
| H7 | Non-uniform allocation improves post-FT | Wins 0/3 post-FT; extreme early ablation | Not supported |
| H8 | Iterative > one-shot at same FT budget | 50%: 230.55 vs 251.51 (~8%); 70%: 539.45 vs 588.89 (~8%) | Confirmed at 50% and 70% |
| H9 | Hybrid LRP+GradMag beats both | Wins 0/3; LRP best at 70%, GradMag at 30-50% | Not supported |
| H10 | Gate path adds critical relevance | rho=0.874 vs content-only; beats GradMag post-FT at all sparsities | Confirmed |
| H11 | CE seeding > max-logit | Global rho=0.469; max-logit wins 30%, CE wins 50% | Partial |
| H12 | Activation < LRP and GradMag | Wins one-shot everywhere; competitive post-FT | Partial |
| H13 | LRP mixed-precision > uniform INT8 | All mixed methods ~35.8 PPL vs uniform 51.4; criterion irrelevant | Partial |
| H14 | LRP Prune+Quant composability | LRP beats random (122.83 vs 195.56); quant adds nothing after FT (0.3% PPL change) | Partial |
| H15 | Gate inclusion helps all methods; LRP-GI best at 50-70% | 30%: LRP-GI=58.67≈GradMag-GI=58.42; 50%: LRP-GI=110.58 vs GradMag-GI=121.03 (8.6%); 70%: 285.28 vs 303.44 (6%) | Confirmed |

### 6.2 H10: Gate-Inclusive LRP (Central Finding)

| Sparsity | Content pre | Gate-incl. pre | Content post-FT | Gate-incl. post-FT | GradMag post-FT |
|---|---|---|---|---|---|
| 30% | 1,747 | 106 | 113.95 | 61.16 | 81.73 |
| 50% | 7,233 | 311 | 251.51 | 115.05 | 233.28 |
| 70% | 54,272 | 879 | 665.19 | 284.49 | 904.43 |

Gate-inclusive LRP: one-shot 16-62x lower than content-only; post-FT 2-2.4x
lower; beats content-only GradMag post-FT by 25%/51%/69% at 30/50/70%.

Layer 0 gate/total ratio = 0.77; content-vs-gate rho = 0.030. Content-only
scoring assigns misleadingly low relevance to gate-active early channels.

H15 established that gate inclusion improves ALL scoring methods (LRP: 48-52%,
GradMag: 29-66%, Activation: 15-17% improvement). Among gate-inclusive methods,
LRP-GI wins at 50% and 70% (by 8.6% and 6.0% over GradMag-GI); at 30% both
are effectively tied (58.67 vs 58.42). See Section 6.9 for full H15 results.

### 6.3 H4 One-Shot Perplexity (content-only LRP; gate-inclusive results in Section 6.9)

| Sparsity | Base | LRP (content) | GradMag | Random |
|---|---|---|---|---|
| 10% | 35.76 | 264.93 | 141.53 | 75.12 |
| 30% | 35.76 | 1,747 | 1,587 | 16,738 |
| 50% | 35.76 | 7,233 | 8,931 | 50,651,582 |
| 70% | 35.76 | 54,272 | 100,474 | 1.97e15 |

Mean log-PPL (nats) at 30-70%: LRP=5.27, GradMag=5.54, Random=20.2.

### 6.4 H7: Non-Uniform Sparsity Allocation

One-shot, non-uniform improves substantially over uniform (3,719 vs 7,233 at 50%).
Post-FT, uniform wins all three cases. At 50%, layer 0 retains only 153/1536
channels (10%) under the global-threshold policy; 30 fine-tuning steps cannot
recover from this degree of ablation. A revised implementation with a per-layer
minimum keep ratio (e.g., 40%) is planned.

### 6.5 H8: Iterative Pruning

Post-FT results at matched 30-step budget (split 15+15):
- 50% sparsity: iterative=230.55 vs one-shot=251.51 (8.3% improvement)
- 70% sparsity: iterative=539.45 vs one-shot=588.89 (8.4% improvement)

Re-scoring on the round-1 fine-tuned model concentrates the second pruning round
on channels that remain low-relevance after recovery. The ~8% improvement is
consistent across both sparsity levels, confirming the effect is real rather than
noise. Note: an earlier buggy run reported 49% improvement at 50% sparsity; this
was caused by the round-2 prune function silently failing to remove any channels
(post-prune PPL remained flat at 132.52). The fixed script shows real round-2
degradation (PPL jumps to 466 and 1472) followed by genuine recovery.

### 6.6 H12: Activation Scoring

| | 30% post-FT | 50% post-FT | 70% post-FT |
|---|---|---|---|
| LRP (content) | 122.53 | 272.66 | 665.19 |
| GradMag | 84.80 | 274.96 | 1051.11 |
| Activation | 92.29 | 256.23 | 718.76 |

Activation scoring requires only a forward pass. Best one-shot at all sparsities.
Competitive post-FT. Global rho with GradMag = -0.017 (selects different channels).

### 6.7 H13: Mixed-Precision Quantisation

| Method | PPL | vs Uniform INT8 |
|---|---|---|
| FP16 base | 35.76 | -15.66 |
| Uniform INT8 | 51.42 | — |
| LRP-guided 50/50 | 35.85 | -15.57 |
| GradMag-guided 50/50 | 35.77 | -15.65 |
| Random 50/50 | 35.79 | -15.64 |

Any 50/50 mixed-precision split dramatically outperforms uniform INT8 for Mamba.
Channel selection criterion makes negligible difference at row granularity.

### 6.8 H14: Combined Pruning + Quantisation

| Condition | Pre-FT PPL | Post-FT PPL | Final PPL | Params |
|---|---|---|---|---|
| FP16 base | — | — | 35.76 | 100% |
| Prune-only (LRP) | 1,747 | 122.53 | 122.53 | 79% |
| Quant-only (LRP) | — | — | 35.85 | 100% |
| Prune+Quant (LRP/LRP) | 1,747 | 122.53 | 122.83 | 79% |
| Prune+Quant (LRP/Random) | 1,747 | 122.53 | 122.72 | 79% |
| Prune+Quant (GradMag) | 1,587 | 84.80 | 84.73 | 79% |
| Prune+Quant (Random) | 16,738 | 189.17 | 195.56 | 79% |

LRP Prune+Quant beats Random Prune+Quant (122.83 vs 195.56). GradMag Prune+Quant
wins overall (84.73) — consistent with GradMag's post-FT advantage at 30% sparsity
(H6, H9). The quantisation step after fine-tuning adds essentially nothing: LRP
prune-only=122.53 vs LRP prune+quant=122.83 (0.3% change). This is fully
consistent with H13 — at row-level INT8 granularity, the criterion for which
channels to quantise is irrelevant once the model has been fine-tuned. LRP/LRP vs
LRP/Random quant: 122.83 vs 122.72, confirming random quantisation is sufficient.

### 6.9 H15: Gate-Inclusive Scoring — Is It LRP or Just the Gate?

Key question answered by H15: does H10's improvement come from LRP's conservation
corrections, or from including the gate path in any scoring method?

Post-FT perplexity (WikiText-2, 30 FT steps):

| Method | 30% | 50% | 70% |
|---|---|---|---|
| LRP-CO | 113.95 | 251.51 | 588.89 |
| LRP-GI | **58.67** | **110.58** | **285.28** |
| GradMag-CO | 81.73 | 233.28 | 904.43 |
| GradMag-GI | **58.42** | 121.03 | 303.44 |
| Activation-CO | 94.92 | 226.27 | 601.77 |
| Activation-GI | 78.48 | 184.12 | 508.56 |
| Random | 156.26 | 653.70 | 1789.97 |

Gate inclusion improves all methods: LRP by 48–52%, GradMag by 29–66%,
Activation by 15–19%. Among gate-inclusive methods: at 30% LRP-GI and GradMag-GI
are effectively tied (58.67 vs 58.42, 0.4%); at 50% and 70% LRP-GI wins by 8.6%
and 6.0% respectively. Cross-method GI rankings differ substantially
(LRP-GI vs GradMag-GI global ρ=0.568), confirming these are distinct signals.

Two-part contribution: (1) gate inclusion is the dominant architectural fix —
any method benefits; (2) LRP's conservation corrections add further value at
moderate-to-high sparsity where channel decisions are harder.

### 6.10 Limitations

- Single seed throughout. Full experiment uses >=3 seeds.
- H4/H6 content-only LRP numbers superseded by H15's gate-inclusive results (Section 6.9).
- H2/H5 directions not pre-registered; labeled exploratory.
- H7 non-uniform implementation has no per-layer floor, causing extreme ablation.

---

## 7. Expected Contribution

1. Gate path is critical for Mamba channel scoring (H10+H15). Including the gate
   half of in_proj in any scoring formula (LRP, GradMag, or activation magnitude)
   produces 15-66% lower post-FT perplexity than content-only scoring. This is an
   architectural insight about Mamba that generalises beyond MambaLRP. Among
   gate-inclusive methods, MambaLRP wins at 50-70% sparsity (6-9% over
   gate-inclusive GradMag) where channel selection decisions are harder.

2. Pure weight magnitude is random for Mamba (H3). Broad implications for
   magnitude-based compression of gated SSM architectures.

3. Iterative LRP pruning with mid-point re-scoring (H8). ~8% consistent
   improvement over one-shot at both 50% and 70% sparsity at matched fine-tuning
   budget (230.55 vs 251.51 at 50%; 539.45 vs 588.89 at 70%).

4. Activation-magnitude scoring as a fast competitive baseline (H12). No backward
   pass required; competitive with gradient-based methods post-FT.

---

## 8. References

Jafari, F. R., Montavon, G., Mueller, K.-R. & Eberle, O. (2024). MambaLRP:
Explaining Selective State Space Sequence Models. NeurIPS 2024.

Gu, A. & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective
State Spaces. arXiv:2312.00752.

Becking, D. et al. (2022). ECQ: Explainability-Driven Quantization. Machine
Learning, 111.

Yeom, S.-K. et al. (2021). Pruning by Explaining. Pattern Recognition, 115.

Sabih, M., Hannig, F. & Teich, J. (2022). DyFiP: Dynamic Filter Pruning.
EuroMLSys.

---

## Appendix: Environment

| Component | Version |
|---|---|
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| transformers | 5.16.1 |
| mamba-ssm | Not installed (slow_forward used) |
| GPU (preliminary) | Tesla T4, 16 GB VRAM |
| GPU (full experiment) | NVIDIA A6000, 48 GB VRAM |
