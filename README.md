# Beyond Magnitude: Attribution-Guided Structural Pruning of Mamba SSMs

Course project, CS6886 *System Engineering for Deep Learning* (IIT Madras), Chirag (DA25M008).

I remove whole channels from every block of a pretrained Mamba language model and ask **which importance score picks the best channels**:
weight magnitude, gradient and activation scores, random, or relevance from [MambaLRP](https://github.com/FarnoushRJ/MambaLRP) (layer-wise relevance
propagation for Mamba). I compare 16 selection methods on **mamba-130m** (147 runs) and **mamba-370m** (75 runs), 3 seeds each, fine-tune for
500 steps and measure WikiText-103 perplexity. Every difference is a paired percentage change with a 95 % confidence interval.

## Main results

| Question | Result |
|---|---|
| Does the classic baseline, weight magnitude, work? | **No, it is worse than random** (130M, after fine-tuning: +39 / +55 / +40 % content-only, +51 / +89 / +117 % gate-inclusive); LRP-GI beats it by 51 / 68 / 82 % |
| Does attribution beat random and gradients? | LRP-GI is 26 / 40 / 60 % better than random (22 / 33 / 45 % on 370M) and 4 / 12 / 13 % better than the Taylor score |
| Which ingredient matters? | **The gate path.** Adding the gate half of `in_proj` lowers perplexity by 19–25 % for LRP (7–12 % on 370M), and by 8–26 % for gradient, activation and Taylor scores |
| Do LRP's special rules help? | **Little or nothing** once the gate is included: plain gradient × input and plain post-gate activation (forward pass only) are within about 4 % |
| Why do the models differ? | The gate-aware scores keep 87–96 % of the same channels; content-only LRP differs mainly in **layer 0** (overlap at chance level) |
| Is the gate effect all in layer 0? | **No**: the gate in layer 0 alone recovers about 31 / 37 / 58 % of it (21–32 % on 370M); the other layers carry the rest |

Full tables and all 18 contrasts: [`results/mamba-130m/summary.md`](results/mamba-130m/summary.md) and [`results/mamba-370m/summary.md`](results/mamba-370m/summary.md).
Unpruned perplexity is 32.44 (130M) and 22.42 (370M); the dense model after the same fine-tuning reaches 22.17 and 17.71.
Caveats: two checkpoints of one model family, three seeds, in-domain fine-tuning, one definition of weight magnitude, and several contrasts were added after seeing data (see [`docs/method.md`](docs/method.md)).

## Repository structure

```
.
├── src/                       experiment code
│   ├── mamba_grid.py            score → prune → fine-tune → evaluate; one JSON per run, resumable
│   ├── aggregate.py             run JSONs → tables, paired contrasts, figures
│   ├── compare_pruned_models.py which channels each method keeps (offline, no GPU)
│   ├── install_env.py           Colab kernel environment (torch 2.10, mamba_ssm, causal_conv1d)
│   └── kernel_check.py          fused kernels vs. slow path on pruned widths
├── tests/                     offline correctness tests (tiny random Mamba, CPU)
├── colab/                     ready-to-run Colab notebooks (one account per seed)
│   ├── 1_main_grid_130m.ipynb
│   ├── 2_extras_130m.ipynb      layer-0 hybrids, other gate/content pairs, magnitude baselines
│   └── 3_replication_370m.ipynb
├── scripts/make_colab_zip.sh  packages src/ + tests/ as mamba_grid.zip for Colab
├── results/
│   ├── mamba-130m/              runs/ (147 JSONs), scores/, summary.md, results.csv, contrasts.csv, figures/, kept_channels/
│   └── mamba-370m/              runs/ (75 JSONs), scores/, summary.md, results.csv, contrasts.csv, figures/
├── docs/method.md             pruning operation, the 16 methods, protocol, statistics
└── exploratory-study/         earlier single-seed study (hypotheses H1–H15) that motivated the grid
```

## Quick start: reproduce the tables and figures (no GPU)

```bash
git clone https://github.com/<your-username>/mamba-attribution-pruning.git
cd mamba-attribution-pruning
pip install -r requirements.txt

python src/aggregate.py --runs results/mamba-130m --out my_results/130m      # tables, contrasts, figures
python src/aggregate.py --runs results/mamba-370m --out my_results/370m
python src/compare_pruned_models.py --results results/mamba-130m --out my_results/kept   # kept-channel analysis
```
The output matches the stored `summary.md`, `results.csv` and `contrasts.csv`.

## Re-run the experiments (Colab T4)

One Colab account can run one GPU session at a time, so use one account per seed (0, 1, 2).

1. `bash scripts/make_colab_zip.sh` creates `mamba_grid.zip`.
2. In Colab, upload `mamba_grid.zip` to `/content` and open a notebook from `colab/`. Set `SEED` (0 / 1 / 2) in the first cell.
3. *Runtime → Run all*. The notebook computes the scores (slow PyTorch path), installs the fused kernels, restarts the runtime by itself, and runs the
   grid. Press *Run all* a second time after the restart. Finished runs are skipped, so restarting is always safe. Results go to Google Drive
   (`MyDrive/mamba_grid_out[_370m]`) and are also downloaded as a zip.
4. Unzip each seed's output into one folder per seed, then merge: `python src/aggregate.py --runs <seed0> <seed1> <seed2> --out my_results/new`.

Run order: notebook 1 first (it creates the cached scores); notebook 2 reuses those scores; notebook 3 is independent (370M).
Typical cost: 5–7 min per run on 130M and 8–14 min on 370M. Keep `KEEP_MULTIPLE = 8`.

## Tests

```bash
git clone https://github.com/FarnoushRJ/MambaLRP /tmp/MambaLRP
python tests/test_grid.py /tmp/MambaLRP     # scores match the original formulas; pruned model == zeroed model; resumable
python tests/test_aggregate.py              # synthetic effects give the right verdicts
```

## Earlier study
[`exploratory-study/`](exploratory-study): proposal, progress report and the hypothesis scripts (single seed, WikiText-2, 30 fine-tuning steps). Several of its conclusions were revised by the multi-seed grid (see `docs/method.md` and the summaries).

## References
- A. Gu, T. Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752, 2023.
- F. Rezaei Jafari et al. *MambaLRP: Explaining Selective State Space Sequence Models.* NeurIPS 2024. https://github.com/FarnoushRJ/MambaLRP
- S.-K. Yeom et al. *Pruning by explaining.* Pattern Recognition 115, 2021.
- S. Han et al. *Learning both Weights and Connections for Efficient Neural Networks.* NeurIPS 2015.
- P. Molchanov et al. *Importance Estimation for Neural Network Pruning.* CVPR 2019.