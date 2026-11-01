# Grid results  (protocol `main`, seeds [0, 1, 2], 147 runs)

Unpruned model perplexity on the evaluation set: **32.44**.

Dense control after the same fine-tuning: 22.17 (n=3).

Fine-tuning: {'steps': 500, 'batch': 8, 'seq': 128, 'lr': 0.0002}.  Evaluation tokens: 286,208.  Calibration texts: 256.

## 1. Perplexity by method and sparsity

Geometric mean over seeds; post-FT is after fine-tuning. `vs Random` = paired change in post-FT perplexity (negative = better than random). `vs dense` = change relative to the UNPRUNED model given the same fine-tuning (the right reference: fine-tuning on the training corpus also improves the dense model).

### Sparsity 30%

| method | seeds | pre-FT PPL | post-FT PPL | spread (sd of log) | vs Random | vs dense-FT |
|---|---|---|---|---|---|---|
| GxI-GI | 3 | 72.0 | **29.9** | 0.004 | -27.5% [-29.3, -25.6] | +35.0% [+33.6, +36.4] |
| LRP-GI | 3 | 105.8 | **30.4** | 0.004 | -26.4% [-28.6, -24.1] | +37.0% [+35.6, +38.3] |
| LRP-Out | 3 | 95.8 | **30.4** | 0.005 | -26.3% [-27.9, -24.6] | +37.2% [+35.9, +38.6] |
| GxA-Out | 3 | 98.3 | **30.6** | 0.007 | -25.9% [-27.6, -24.2] | +37.9% [+35.6, +40.2] |
| Act-Out | 3 | 97.6 | **31.0** | 0.003 | -24.9% [-26.9, -22.8] | +39.8% [+38.8, +40.8] |
| GradMag-GI | 3 | 107.1 | **31.6** | 0.007 | -23.5% [-25.1, -21.8] | +42.5% [+40.3, +44.7] |
| Hyb-L0CO | 3 | 243.5 | **32.1** | 0.003 | -22.2% [-24.4, -19.9] | +44.8% [+43.7, +45.8] |
| GxI-CO | 3 | 1,017.6 | **34.2** | 0.004 | -17.1% [-18.9, -15.2] | +54.3% [+53.0, +55.7] |
| Hyb-L0GI | 3 | 965.6 | **35.1** | 0.006 | -14.9% [-17.5, -12.3] | +58.4% [+55.8, +61.1] |
| GradMag-CO | 3 | 2,513.4 | **35.2** | 0.003 | -14.7% [-17.6, -11.7] | +58.8% [+57.2, +60.5] |
| Act-GI | 3 | 239.0 | **37.4** | 0.002 | -9.3% [-11.7, -6.9] | +68.8% [+68.1, +69.4] |
| LRP-CO | 3 | 1,950.5 | **37.5** | 0.007 | -9.1% [-12.3, -5.7] | +69.2% [+66.0, +72.6] |
| Act-CO | 3 | 632.2 | **40.9** | 0.004 | -0.9% [-5.0, +3.4] | +84.5% [+81.8, +87.1] |
| Random | 3 | 1,492.4 | **41.3** | 0.013 | - | +86.1% [+81.0, +91.4] |
| Mag-CO | 3 | 321,792,703,779.0 | **57.3** | 0.006 | +38.9% [+36.4, +41.4] | +158.5% [+155.5, +161.6] |
| Mag-GI | 3 | 31,498,730,425.0 | **62.3** | 0.010 | +51.0% [+48.4, +53.7] | +181.1% [+175.0, +187.4] |

### Sparsity 50%

| method | seeds | pre-FT PPL | post-FT PPL | spread (sd of log) | vs Random | vs dense-FT |
|---|---|---|---|---|---|---|
| LRP-GI | 3 | 326.5 | **45.5** | 0.009 | -40.0% [-46.3, -33.0] | +105.4% [+101.4, +109.4] |
| Act-Out | 3 | 279.0 | **45.6** | 0.003 | -40.0% [-46.3, -32.9] | +105.6% [+104.2, +107.1] |
| LRP-Out | 3 | 249.3 | **45.9** | 0.009 | -39.6% [-45.8, -32.7] | +106.8% [+102.3, +111.4] |
| GxI-GI | 3 | 253.7 | **46.7** | 0.011 | -38.6% [-44.2, -32.4] | +110.4% [+104.5, +116.4] |
| GxA-Out | 3 | 246.4 | **47.0** | 0.011 | -38.2% [-43.4, -32.5] | +111.7% [+105.5, +118.1] |
| Hyb-L0CO | 3 | 747.8 | **50.3** | 0.011 | -33.7% [-40.6, -26.0] | +127.0% [+121.3, +132.8] |
| GradMag-GI | 3 | 409.3 | **51.7** | 0.015 | -32.0% [-38.2, -25.1] | +133.0% [+124.4, +141.8] |
| Hyb-L0GI | 3 | 4,045.6 | **54.4** | 0.011 | -28.3% [-35.9, -19.9] | +145.5% [+139.2, +152.0] |
| GxI-CO | 3 | 2,684.4 | **55.6** | 0.010 | -26.8% [-33.1, -20.0] | +150.6% [+144.2, +157.3] |
| GradMag-CO | 3 | 11,222.7 | **59.2** | 0.013 | -22.1% [-28.2, -15.4] | +166.9% [+158.1, +176.0] |
| Act-GI | 3 | 1,068.2 | **59.7** | 0.004 | -21.5% [-30.6, -11.2] | +169.0% [+166.8, +171.3] |
| LRP-CO | 3 | 6,889.5 | **60.5** | 0.009 | -20.3% [-27.8, -12.0] | +173.0% [+167.0, +179.2] |
| Act-CO | 3 | 1,992.6 | **72.2** | 0.004 | -5.0% [-15.0, +6.3] | +225.5% [+223.0, +228.0] |
| Random | 3 | 797,455.7 | **76.0** | 0.045 | - | +242.5% [+205.3, +284.3] |
| Mag-CO | 3 | 59,786,472,005,156.9 | **117.7** | 0.051 | +55.0% [+25.0, +92.2] | +430.8% [+369.1, +500.7] |
| Mag-GI | 3 | 14,236,172,227,139.0 | **143.4** | 0.005 | +88.7% [+66.6, +113.8] | +546.5% [+539.5, +553.6] |

### Sparsity 70%

| method | seeds | pre-FT PPL | post-FT PPL | spread (sd of log) | vs Random | vs dense-FT |
|---|---|---|---|---|---|---|
| GxI-GI | 3 | 913.8 | **82.7** | 0.007 | -61.8% [-70.0, -51.2] | +273.0% [+266.8, +279.4] |
| Act-Out | 3 | 849.7 | **83.7** | 0.006 | -61.3% [-69.6, -50.8] | +277.5% [+271.8, +283.3] |
| LRP-Out | 3 | 715.1 | **85.6** | 0.006 | -60.4% [-68.9, -49.6] | +286.2% [+280.4, +292.1] |
| LRP-GI | 3 | 965.5 | **86.0** | 0.008 | -60.3% [-68.9, -49.2] | +287.7% [+280.3, +295.3] |
| GxA-Out | 3 | 770.4 | **86.1** | 0.005 | -60.2% [-68.6, -49.6] | +288.3% [+283.1, +293.5] |
| Hyb-L0GI | 3 | 14,065.6 | **94.4** | 0.007 | -56.4% [-65.7, -44.5] | +325.8% [+317.7, +334.2] |
| Hyb-L0CO | 3 | 1,740.5 | **97.3** | 0.009 | -55.1% [-64.6, -43.0] | +338.7% [+328.2, +349.3] |
| GradMag-GI | 3 | 1,469.8 | **98.5** | 0.008 | -54.5% [-64.4, -41.9] | +344.3% [+336.3, +352.4] |
| LRP-CO | 3 | 32,642.4 | **107.7** | 0.015 | -50.2% [-61.0, -36.6] | +385.8% [+366.9, +405.4] |
| GxI-CO | 3 | 22,759.0 | **111.2** | 0.021 | -48.6% [-59.2, -35.3] | +401.6% [+374.5, +430.3] |
| Act-GI | 3 | 6,764.0 | **114.9** | 0.004 | -46.9% [-57.9, -33.1] | +418.3% [+412.1, +424.7] |
| GradMag-CO | 3 | 118,929.4 | **124.8** | 0.011 | -42.3% [-53.8, -28.0] | +462.8% [+445.6, +480.6] |
| Act-CO | 3 | 5,662.7 | **136.7** | 0.010 | -36.9% [-50.9, -18.9] | +516.3% [+501.9, +531.1] |
| Random | 3 | 31,744,057,023,774,172.0 | **216.5** | 0.091 | - | +876.2% [+677.9, +1125.0] |
| Mag-CO | 3 | 397,466,154,545,962,240.0 | **302.0** | 0.032 | +39.5% [+2.9, +89.1] | +1262.0% [+1159.0, +1373.3] |
| Mag-GI | 3 | 3,183,819,359,762,370.5 | **469.8** | 0.037 | +117.0% [+78.4, +164.1] | +2018.5% [+1826.1, +2230.2] |

## 2. Pre-specified paired contrasts

Percent change in perplexity of A relative to B (negative = A better), 95% t-interval over seeds, seeds in which A < B. Post-FT is the primary endpoint; pre-FT (one-shot) is secondary. Verdict: *equivalent* if the whole interval lies within +-2%; otherwise A/B better if the interval excludes 0; otherwise inconclusive.

| id | A vs B | sparsity | post-FT change [95% CI] | wins | verdict | pre-FT change [95% CI] |
|---|---|---|---|---|---|---|
| C1 | LRP-GI vs LRP-CO | 30% | -19.1% [-19.8, -18.3] | 3/3 | LRP-GI better | -94.6% [-94.9, -94.2] |
| C1 | LRP-GI vs LRP-CO | 50% | -24.8% [-25.6, -23.9] | 3/3 | LRP-GI better | -95.3% [-96.1, -94.3] |
| C1 | LRP-GI vs LRP-CO | 70% | -20.2% [-22.8, -17.4] | 3/3 | LRP-GI better | -97.0% [-97.9, -95.9] |
| C2 | LRP-GI vs GxI-GI | 30% | +1.4% [+0.9, +2.0] | 0/3 | equivalent (within +-2%) | +46.9% [+44.6, +49.2] |
| C2 | LRP-GI vs GxI-GI | 50% | -2.4% [-3.8, -0.9] | 3/3 | LRP-GI better | +28.7% [+18.7, +39.5] |
| C2 | LRP-GI vs GxI-GI | 70% | +3.9% [+3.7, +4.2] | 0/3 | GxI-GI better | +5.7% [-10.3, +24.4] |
| C3 | LRP-Out vs GxA-Out | 30% | -0.5% [-1.2, +0.2] | 3/3 | equivalent (within +-2%) | -2.6% [-5.0, -0.1] |
| C3 | LRP-Out vs GxA-Out | 50% | -2.3% [-4.4, -0.2] | 3/3 | LRP-Out better | +1.2% [-1.5, +3.9] |
| C3 | LRP-Out vs GxA-Out | 70% | -0.5% [-1.1, -0.0] | 3/3 | equivalent (within +-2%) | -7.2% [-16.5, +3.1] |
| C4 | LRP-GI vs GradMag-GI | 30% | -3.9% [-4.9, -2.9] | 3/3 | LRP-GI better | -1.2% [-14.9, +14.6] |
| C4 | LRP-GI vs GradMag-GI | 50% | -11.8% [-13.6, -10.0] | 3/3 | LRP-GI better | -20.2% [-33.5, -4.2] |
| C4 | LRP-GI vs GradMag-GI | 70% | -12.7% [-13.5, -12.0] | 3/3 | LRP-GI better | -34.3% [-43.5, -23.7] |
| C5 | LRP-GI vs Act-Out | 30% | -2.0% [-2.4, -1.6] | 3/3 | LRP-GI better | +8.4% [+6.6, +10.1] |
| C5 | LRP-GI vs Act-Out | 50% | -0.1% [-1.4, +1.2] | 2/3 | equivalent (within +-2%) | +17.0% [+9.0, +25.7] |
| C5 | LRP-GI vs Act-Out | 70% | +2.7% [+2.1, +3.3] | 0/3 | Act-Out better | +13.6% [+10.7, +16.7] |
| C6 | LRP-Out vs Act-Out | 30% | -1.8% [-2.3, -1.3] | 3/3 | LRP-Out better | -1.8% [-13.7, +11.6] |
| C6 | LRP-Out vs Act-Out | 50% | +0.6% [-0.9, +2.1] | 1/3 | inconclusive | -10.6% [-15.9, -5.1] |
| C6 | LRP-Out vs Act-Out | 70% | +2.3% [+2.1, +2.5] | 0/3 | Act-Out better | -15.8% [-22.5, -8.6] |
| C7 | Act-Out vs Act-GI | 30% | -17.2% [-17.5, -16.9] | 3/3 | Act-Out better | -59.2% [-61.0, -57.2] |
| C7 | Act-Out vs Act-GI | 50% | -23.6% [-24.7, -22.5] | 3/3 | Act-Out better | -73.9% [-74.9, -72.9] |
| C7 | Act-Out vs Act-GI | 70% | -27.2% [-27.9, -26.4] | 3/3 | Act-Out better | -87.4% [-91.2, -82.1] |
| C8 | LRP-GI vs Random | 30% | -26.4% [-28.6, -24.1] | 3/3 | LRP-GI better | -92.9% [-97.9, -76.3] |
| C8 | LRP-GI vs Random | 50% | -40.0% [-46.3, -33.0] | 3/3 | LRP-GI better | -100.0% [-100.0, -99.2] |
| C8 | LRP-GI vs Random | 70% | -60.3% [-68.9, -49.2] | 3/3 | LRP-GI better | -100.0% [-100.0, -100.0] |
| C9 | GxI-GI vs GxI-CO | 30% | -12.5% [-12.8, -12.2] | 3/3 | GxI-GI better | -92.9% [-93.9, -91.7] |
| C9 | GxI-GI vs GxI-CO | 50% | -16.1% [-17.6, -14.5] | 3/3 | GxI-GI better | -90.6% [-92.6, -88.0] |
| C9 | GxI-GI vs GxI-CO | 70% | -25.6% [-29.7, -21.4] | 3/3 | GxI-GI better | -96.0% [-98.1, -91.6] |
| C10 | Act-GI vs Act-CO | 30% | -8.5% [-10.0, -7.0] | 3/3 | Act-GI better | -62.2% [-63.3, -61.0] |
| C10 | Act-GI vs Act-CO | 50% | -17.3% [-18.6, -16.1] | 3/3 | Act-GI better | -46.4% [-49.8, -42.7] |
| C10 | Act-GI vs Act-CO | 70% | -15.9% [-17.8, -13.9] | 3/3 | Act-GI better | +19.4% [-49.3, +181.4] |
| C11 | GradMag-GI vs GradMag-CO | 30% | -10.3% [-11.7, -8.9] | 3/3 | GradMag-GI better | -95.7% [-97.8, -91.6] |
| C11 | GradMag-GI vs GradMag-CO | 50% | -12.7% [-14.9, -10.5] | 3/3 | GradMag-GI better | -96.4% [-98.6, -90.2] |
| C11 | GradMag-GI vs GradMag-CO | 70% | -21.1% [-24.3, -17.6] | 3/3 | GradMag-GI better | -98.8% [-99.9, -87.8] |
| C13 | Hyb-L0GI vs LRP-CO | 30% | -6.4% [-6.9, -5.9] | 3/3 | Hyb-L0GI better | -50.5% [-52.3, -48.7] |
| C13 | Hyb-L0GI vs LRP-CO | 50% | -10.1% [-11.4, -8.7] | 3/3 | Hyb-L0GI better | -41.3% [-42.9, -39.6] |
| C13 | Hyb-L0GI vs LRP-CO | 70% | -12.3% [-14.4, -10.2] | 3/3 | Hyb-L0GI better | -56.9% [-63.6, -49.0] |
| C14 | Hyb-L0GI vs LRP-GI | 30% | +15.7% [+14.8, +16.5] | 0/3 | LRP-GI better | +813.0% [+772.7, +855.1] |
| C14 | Hyb-L0GI vs LRP-GI | 50% | +19.5% [+18.7, +20.4] | 0/3 | LRP-GI better | +1139.0% [+954.4, +1355.9] |
| C14 | Hyb-L0GI vs LRP-GI | 70% | +9.8% [+8.7, +10.9] | 0/3 | LRP-GI better | +1356.8% [+837.4, +2164.1] |
| C15 | Hyb-L0CO vs LRP-GI | 30% | +5.7% [+5.4, +6.0] | 0/3 | LRP-GI better | +130.2% [+121.8, +138.9] |
| C15 | Hyb-L0CO vs LRP-GI | 50% | +10.5% [+9.9, +11.2] | 0/3 | LRP-GI better | +129.0% [+122.2, +136.0] |
| C15 | Hyb-L0CO vs LRP-GI | 70% | +13.1% [+10.9, +15.4] | 0/3 | LRP-GI better | +80.3% [+68.0, +93.4] |
| C16 | LRP-GI vs Mag-GI | 30% | -51.3% [-52.7, -49.8] | 3/3 | LRP-GI better | -100.0% [-100.0, -100.0] |
| C16 | LRP-GI vs Mag-GI | 50% | -68.2% [-69.1, -67.3] | 3/3 | LRP-GI better | -100.0% [-100.0, -100.0] |
| C16 | LRP-GI vs Mag-GI | 70% | -81.7% [-83.5, -79.7] | 3/3 | LRP-GI better | -100.0% [-100.0, -100.0] |
| C17 | LRP-GI vs Mag-CO | 30% | -47.0% [-48.0, -46.0] | 3/3 | LRP-GI better | -100.0% [-100.0, -100.0] |
| C17 | LRP-GI vs Mag-CO | 50% | -61.3% [-65.5, -56.6] | 3/3 | LRP-GI better | -100.0% [-100.0, -100.0] |
| C17 | LRP-GI vs Mag-CO | 70% | -71.5% [-73.2, -69.7] | 3/3 | LRP-GI better | -100.0% [-100.0, -100.0] |
| C18 | Mag-GI vs Random | 30% | +51.0% [+48.4, +53.7] | 0/3 | Random better | +2110624566.3% [+642899550.1, +6929131812.2] |
| C18 | Mag-GI vs Random | 50% | +88.7% [+66.6, +113.8] | 0/3 | Random better | +1785199119.5% [+94112492.1, +33863016343.9] |
| C18 | Mag-GI vs Random | 70% | +117.0% [+78.4, +164.1] | 0/3 | Random better | -90.0% [-100.0, +3057013621.9] |
| C12 | GxI-GI vs LRP-GI | 30% | -1.4% [-1.9, -0.9] | 3/3 | equivalent (within +-2%) | -31.9% [-33.0, -30.9] |
| C12 | GxI-GI vs LRP-GI | 50% | +2.4% [+0.9, +4.0] | 0/3 | LRP-GI better | -22.3% [-28.3, -15.8] |
| C12 | GxI-GI vs LRP-GI | 70% | -3.8% [-4.0, -3.5] | 3/3 | GxI-GI better | -5.4% [-19.6, +11.4] |

Contrast questions: **C1** Does adding the gate path help LRP scoring?; **C2** Do the LRP propagation rules help beyond plain gradient x input (same sites, in_proj)?; **C3** Do the LRP rules help beyond plain gradient x activation at the post-gate site?; **C4** LRP-GI vs gate-inclusive Taylor/gradient-magnitude (the H15 comparison); **C5** LRP-GI vs the cheapest post-gate baseline (mean |activation| at out_proj input); **C6** Does LRP beat plain activation at the SAME post-gate site?; **C7** Is post-gate activation better than pre-gate (in_proj) activation?; **C8** Is LRP-GI better than random channel removal?; **C9** [added after seed 0] gate path for plain gradient x input; **C10** [added after seed 0] gate path for pre-gate activation; **C11** [added after seed 0] gate path for Taylor/gradient-magnitude; **C13** [after the overlap analysis] does the gate in LAYER 0 ALONE recover the gate effect? (content-only LRP everywhere else); **C14** [after the overlap analysis] how much of the gate effect is NOT in layer 0?; **C15** [after the overlap analysis] does removing the gate in LAYER 0 ALONE hurt gate-aware LRP?; **C16** [added after the grid] LRP-GI vs gate-inclusive weight magnitude (the classic baseline); **C17** [added after the grid] LRP-GI vs content-only weight magnitude (the H3 definition); **C18** [added after the grid] is weight magnitude better than random channel removal?; **C12** [added after seed 0] plain gradient x input vs LRP-GI (reverse of C2: is plain >= LRP?)

## 3. Cost of the scores

Wall-clock cost is not recorded per score in the grid; from earlier runs on a T4: LRP/activation ~3.4 min (one backward sweep over 256 sequences), activation alone ~4x cheaper (forward only).

Figures: `figs/ppl_vs_sparsity.png`, `figs/contrasts_forest.png`, `figs/recovery_curves.png`.
