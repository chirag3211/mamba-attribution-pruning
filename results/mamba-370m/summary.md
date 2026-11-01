# Grid results  (protocol `main`, seeds [0, 1, 2], 75 runs)

Unpruned model perplexity on the evaluation set: **22.42**.

Dense control after the same fine-tuning: 17.71 (n=3).

Fine-tuning: {'steps': 500, 'batch': 8, 'seq': 128, 'lr': 0.0002}.  Evaluation tokens: 286,208.  Calibration texts: 256.

## 1. Perplexity by method and sparsity

Geometric mean over seeds; post-FT is after fine-tuning. `vs Random` = paired change in post-FT perplexity (negative = better than random). `vs dense` = change relative to the UNPRUNED model given the same fine-tuning (the right reference: fine-tuning on the training corpus also improves the dense model).

### Sparsity 30%

| method | seeds | pre-FT PPL | post-FT PPL | spread (sd of log) | vs Random | vs dense-FT |
|---|---|---|---|---|---|---|
| GxI-GI | 3 | 45.7 | **22.7** | 0.003 | -24.1% [-25.3, -22.9] | +28.4% [+27.6, +29.1] |
| LRP-GI | 3 | 69.6 | **23.3** | 0.005 | -22.4% [-23.9, -20.8] | +31.4% [+30.2, +32.6] |
| Act-Out | 3 | 61.1 | **23.5** | 0.005 | -21.6% [-23.2, -20.0] | +32.6% [+31.4, +33.8] |
| Hyb-L0CO | 3 | 76.3 | **23.6** | 0.005 | -21.3% [-23.0, -19.6] | +33.1% [+31.6, +34.6] |
| GradMag-GI | 3 | 56.5 | **23.6** | 0.008 | -21.2% [-23.1, -19.3] | +33.2% [+31.0, +35.6] |
| Hyb-L0GI | 3 | 312.7 | **24.7** | 0.002 | -17.7% [-18.9, -16.5] | +39.3% [+38.5, +40.1] |
| LRP-CO | 3 | 364.6 | **25.0** | 0.004 | -16.4% [-17.9, -15.0] | +41.4% [+40.2, +42.6] |
| Random | 3 | 481.4 | **30.0** | 0.004 | - | +69.2% [+67.3, +71.1] |

### Sparsity 50%

| method | seeds | pre-FT PPL | post-FT PPL | spread (sd of log) | vs Random | vs dense-FT |
|---|---|---|---|---|---|---|
| GxI-GI | 3 | 151.8 | **32.7** | 0.004 | -35.3% [-38.0, -32.5] | +84.8% [+83.2, +86.4] |
| LRP-GI | 3 | 254.1 | **33.7** | 0.004 | -33.4% [-36.3, -30.4] | +90.3% [+88.8, +91.8] |
| Act-Out | 3 | 222.9 | **34.1** | 0.006 | -32.5% [-35.5, -29.4] | +92.8% [+90.1, +95.5] |
| Hyb-L0CO | 3 | 292.2 | **34.7** | 0.005 | -31.5% [-34.5, -28.3] | +95.8% [+93.8, +97.9] |
| Hyb-L0GI | 3 | 1,468.7 | **36.4** | 0.005 | -28.1% [-31.5, -24.5] | +105.5% [+103.0, +108.0] |
| GradMag-GI | 3 | 238.3 | **36.7** | 0.008 | -27.4% [-31.1, -23.5] | +107.5% [+103.9, +111.1] |
| LRP-CO | 3 | 1,964.2 | **37.6** | 0.006 | -25.7% [-29.5, -21.7] | +112.3% [+108.9, +115.6] |
| Random | 3 | 93,574.3 | **50.6** | 0.015 | - | +185.7% [+175.2, +196.6] |

### Sparsity 70%

| method | seeds | pre-FT PPL | post-FT PPL | spread (sd of log) | vs Random | vs dense-FT |
|---|---|---|---|---|---|---|
| GxI-GI | 3 | 522.9 | **61.6** | 0.009 | -47.5% [-51.4, -43.2] | +247.7% [+240.2, +255.3] |
| Act-Out | 3 | 663.4 | **62.8** | 0.005 | -46.4% [-50.8, -41.6] | +254.6% [+250.0, +259.4] |
| LRP-GI | 3 | 775.2 | **64.4** | 0.009 | -45.0% [-49.2, -40.6] | +263.7% [+256.3, +271.2] |
| Hyb-L0CO | 3 | 844.5 | **67.1** | 0.008 | -42.7% [-47.1, -37.9] | +279.2% [+272.4, +286.1] |
| Hyb-L0GI | 3 | 37,982.1 | **70.4** | 0.013 | -39.9% [-43.3, -36.4] | +297.4% [+285.9, +309.4] |
| LRP-CO | 3 | 53,344.5 | **73.4** | 0.010 | -37.4% [-41.4, -33.0] | +314.6% [+305.3, +324.1] |
| GradMag-GI | 3 | 875.5 | **74.1** | 0.018 | -36.8% [-41.2, -32.0] | +318.4% [+301.2, +336.3] |
| Random | 3 | 2,984,784,039,395.5 | **117.2** | 0.032 | - | +561.8% [+512.2, +615.3] |

## 2. Pre-specified paired contrasts

Percent change in perplexity of A relative to B (negative = A better), 95% t-interval over seeds, seeds in which A < B. Post-FT is the primary endpoint; pre-FT (one-shot) is secondary. Verdict: *equivalent* if the whole interval lies within +-2%; otherwise A/B better if the interval excludes 0; otherwise inconclusive.

| id | A vs B | sparsity | post-FT change [95% CI] | wins | verdict | pre-FT change [95% CI] |
|---|---|---|---|---|---|---|
| C1 | LRP-GI vs LRP-CO | 30% | -7.1% [-8.1, -6.0] | 3/3 | LRP-GI better | -80.9% [-83.0, -78.5] |
| C1 | LRP-GI vs LRP-CO | 50% | -10.3% [-11.4, -9.3] | 3/3 | LRP-GI better | -87.1% [-88.8, -85.0] |
| C1 | LRP-GI vs LRP-CO | 70% | -12.3% [-13.3, -11.2] | 3/3 | LRP-GI better | -98.5% [-99.0, -97.8] |
| C2 | LRP-GI vs GxI-GI | 30% | +2.3% [+1.5, +3.2] | 0/3 | GxI-GI better | +52.3% [+49.1, +55.7] |
| C2 | LRP-GI vs GxI-GI | 50% | +3.0% [+2.7, +3.3] | 0/3 | GxI-GI better | +67.3% [+57.9, +77.4] |
| C2 | LRP-GI vs GxI-GI | 70% | +4.6% [+4.5, +4.7] | 0/3 | GxI-GI better | +48.2% [+32.3, +66.0] |
| C4 | LRP-GI vs GradMag-GI | 30% | -1.4% [-2.6, -0.2] | 3/3 | LRP-GI better | +23.3% [+17.5, +29.3] |
| C4 | LRP-GI vs GradMag-GI | 50% | -8.3% [-9.2, -7.4] | 3/3 | LRP-GI better | +6.6% [-0.2, +14.0] |
| C4 | LRP-GI vs GradMag-GI | 70% | -13.1% [-15.1, -11.0] | 3/3 | LRP-GI better | -11.5% [-22.9, +1.7] |
| C5 | LRP-GI vs Act-Out | 30% | -0.9% [-1.0, -0.8] | 3/3 | equivalent (within +-2%) | +14.0% [+10.3, +17.8] |
| C5 | LRP-GI vs Act-Out | 50% | -1.3% [-2.1, -0.5] | 3/3 | LRP-GI better | +14.0% [+11.5, +16.5] |
| C5 | LRP-GI vs Act-Out | 70% | +2.5% [+1.1, +4.0] | 0/3 | Act-Out better | +16.8% [+12.4, +21.5] |
| C8 | LRP-GI vs Random | 30% | -22.4% [-23.9, -20.8] | 3/3 | LRP-GI better | -85.5% [-95.6, -52.8] |
| C8 | LRP-GI vs Random | 50% | -33.4% [-36.3, -30.4] | 3/3 | LRP-GI better | -99.7% [-100.0, -89.7] |
| C8 | LRP-GI vs Random | 70% | -45.0% [-49.2, -40.6] | 3/3 | LRP-GI better | -100.0% [-100.0, -100.0] |
| C13 | Hyb-L0GI vs LRP-CO | 30% | -1.5% [-1.8, -1.2] | 3/3 | equivalent (within +-2%) | -14.2% [-16.9, -11.5] |
| C13 | Hyb-L0GI vs LRP-CO | 50% | -3.2% [-3.6, -2.8] | 3/3 | Hyb-L0GI better | -25.2% [-28.0, -22.3] |
| C13 | Hyb-L0GI vs LRP-CO | 70% | -4.1% [-5.1, -3.2] | 3/3 | Hyb-L0GI better | -28.8% [-34.3, -22.9] |
| C14 | Hyb-L0GI vs LRP-GI | 30% | +6.0% [+4.8, +7.2] | 0/3 | LRP-GI better | +349.1% [+309.9, +392.1] |
| C14 | Hyb-L0GI vs LRP-GI | 50% | +8.0% [+7.0, +9.0] | 0/3 | LRP-GI better | +478.1% [+414.2, +550.0] |
| C14 | Hyb-L0GI vs LRP-GI | 70% | +9.3% [+6.9, +11.7] | 0/3 | LRP-GI better | +4799.9% [+3071.5, +7470.2] |
| C15 | Hyb-L0CO vs LRP-GI | 30% | +1.3% [+1.1, +1.5] | 0/3 | equivalent (within +-2%) | +9.6% [+7.1, +12.2] |
| C15 | Hyb-L0CO vs LRP-GI | 50% | +2.9% [+2.6, +3.2] | 0/3 | LRP-GI better | +15.0% [+9.9, +20.3] |
| C15 | Hyb-L0CO vs LRP-GI | 70% | +4.3% [+3.9, +4.6] | 0/3 | LRP-GI better | +8.9% [+6.9, +11.0] |
| C12 | GxI-GI vs LRP-GI | 30% | -2.3% [-3.1, -1.4] | 3/3 | GxI-GI better | -34.4% [-35.8, -32.9] |
| C12 | GxI-GI vs LRP-GI | 50% | -2.9% [-3.2, -2.6] | 3/3 | GxI-GI better | -40.2% [-43.6, -36.7] |
| C12 | GxI-GI vs LRP-GI | 70% | -4.4% [-4.5, -4.3] | 3/3 | GxI-GI better | -32.5% [-39.8, -24.4] |

Contrast questions: **C1** Does adding the gate path help LRP scoring?; **C2** Do the LRP propagation rules help beyond plain gradient x input (same sites, in_proj)?; **C3** Do the LRP rules help beyond plain gradient x activation at the post-gate site?; **C4** LRP-GI vs gate-inclusive Taylor/gradient-magnitude (the H15 comparison); **C5** LRP-GI vs the cheapest post-gate baseline (mean |activation| at out_proj input); **C6** Does LRP beat plain activation at the SAME post-gate site?; **C7** Is post-gate activation better than pre-gate (in_proj) activation?; **C8** Is LRP-GI better than random channel removal?; **C9** [added after seed 0] gate path for plain gradient x input; **C10** [added after seed 0] gate path for pre-gate activation; **C11** [added after seed 0] gate path for Taylor/gradient-magnitude; **C13** [after the overlap analysis] does the gate in LAYER 0 ALONE recover the gate effect? (content-only LRP everywhere else); **C14** [after the overlap analysis] how much of the gate effect is NOT in layer 0?; **C15** [after the overlap analysis] does removing the gate in LAYER 0 ALONE hurt gate-aware LRP?; **C12** [added after seed 0] plain gradient x input vs LRP-GI (reverse of C2: is plain >= LRP?)

## 3. Cost of the scores

Wall-clock cost is not recorded per score in the grid; from earlier runs on a T4: LRP/activation ~3.4 min (one backward sweep over 256 sequences), activation alone ~4x cheaper (forward only).

Figures: `figs/ppl_vs_sparsity.png`, `figs/contrasts_forest.png`, `figs/recovery_curves.png`.
