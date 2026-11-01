# Method and protocol

## What is pruned
Mamba block: `in_proj` output = [content | gate] (each of width D); output = content ⊙ SiLU(gate). Removing inner channel *c* deletes
rows *c* and *c+D* of `in_proj`, conv1d channel *c*, input column *c* of `x_proj`, row *c* of `dt_proj`, input column *c* of `out_proj`,
and entry *c* of `A_log` and `D`. The same number of channels is kept in every layer (per-layer top-*k*), rounded to a multiple of 8 because
the fused kernels require it (`--keep-multiple 8`).

| model | layers | D | kept at 30 / 50 / 70 % removed |
|---|---|---|---|
| mamba-130m | 24 | 1536 | 1072 / 768 / 464 |
| mamba-370m | 48 | 2048 | 1432 / 1024 / 616 |

## Channel-selection methods (16)
| name | measured at | what |
|---|---|---|
| LRP-CO / LRP-GI | `in_proj` output | MambaLRP relevance, content half / content + gate half |
| LRP-Out | `out_proj` input (after the gate) | MambaLRP relevance |
| GxI-CO / GxI-GI | `in_proj` output | plain gradient × input (stock model, no LRP rules) |
| GxA-Out | `out_proj` input | plain gradient × activation |
| GradMag-CO / GradMag-GI | `in_proj` weights | Taylor \|w · ∂L/∂w\| |
| Act-CO / Act-GI / Act-Out | `in_proj` output / `out_proj` input | mean \|activation\| (forward pass only) |
| Mag-CO / Mag-GI | `in_proj` weights | mean \|w\| of the content rows / content + gate rows (no data) |
| Random | – | seed-dependent random ranking |
| Hyb-L0GI | – | LRP-CO everywhere, LRP-GI ranking in layer 0 |
| Hyb-L0CO | – | LRP-GI everywhere, LRP-CO ranking in layer 0 |
| Dense | – | control: unpruned model given the same fine-tuning |

Scoring runs on the pure-PyTorch path (MambaLRP hooks), so `mamba_ssm` must be **absent** during `--stage score`; fine-tuning and evaluation use the fused kernels.

## Protocol (`--protocol main`)
- **Calibration:** 256 sequences × 128 tokens from WikiText-103 train (seed-dependent), summed next-token cross-entropy.
- **Fine-tuning:** all weights, AdamW, lr 2e-4, 20 warm-up steps, cosine to 10 %, clip 1.0, 500 steps, batch 8 × 128; identical training windows for all methods within a seed.
- **Evaluation:** full WikiText-103 test set in 256-token non-overlapping chunks (286,208 tokens). Unpruned perplexity: 32.44 (130M), 22.42 (370M). Dense after the same fine-tuning: 22.17 (130M), 17.71 (370M).
- **Seeds:** 0, 1, 2 change calibration data, fine-tuning order and the random score.
- `--protocol legacy` reproduces the earlier single-seed WikiText-2 setting (see `exploratory-study`).

## Statistics
Differences are paired: for methods A and B the per-seed Δ log-perplexity is formed, reported as a percentage change with a 95 % t-interval over seeds
(n = 3, t = 4.303) and the number of seeds in which A wins. "Equivalent" means the whole interval lies within ±2 % (margin chosen after seeing seed 0).

| contrasts | status |
|---|---|
| C1–C8 (gate path, LRP rules, Taylor, activation, post- vs pre-gate, random) | fixed before the grid was run |
| C9–C12 (gate path for other score families, plain grad × input vs LRP) | added after seed 0 |
| C13–C15 (layer-0 hybrids) | added after the kept-channel overlap analysis |
| C16–C18 (weight-magnitude baselines) | added after the grid |

## Run format
One run = (seed, method, sparsity) → `runs/main/seed{S}_{method}_sp{NN}.json` with pre- / mid- / post-fine-tuning perplexity, training loss curve,
parameter count and timings. Re-running skips finished runs. Cached scores per seed are in `scores/main_seed{S}.pt`.

## Flags (`src/mamba_grid.py`)
`--stage score|run|all`, `--methods`, `--sparsities`, `--seeds`, `--model`, `--out`, `--score-micro` (use 1 for 370m),
`--ft-steps`, `--ft-batch`, `--ft-micro`, `--keep-multiple`, `--eval-max-tokens`, `--protocol main|legacy`.
