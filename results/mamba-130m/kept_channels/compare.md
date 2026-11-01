# How do the pruned models differ?

Offline analysis of the kept channel sets (24 layers x 1,536 channels), mean over seeds [0, 1, 2]. Kept sets are rebuilt exactly as the pruning code builds them (per-layer top-k).


## 30% of channels removed (keep 1072 of 1536 per layer; chance overlap = 0.70)

**Kept-set overlap** (1.00 = identical selection):

| | LRP-GI | LRP-Out | GxI-GI | GxA-Out | Act-Out | GradMag-GI | LRP-CO | Act-GI | Random |
|---|---|---|---|---|---|---|---|---|---|
| **LRP-GI** | 1.00 | 0.96 | 0.92 | 0.94 | 0.93 | 0.84 | 0.91 | 0.74 | 0.70 |
| **LRP-Out** | 0.96 | 1.00 | 0.92 | 0.96 | 0.94 | 0.85 | 0.88 | 0.74 | 0.70 |
| **GxI-GI** | 0.92 | 0.92 | 1.00 | 0.94 | 0.91 | 0.86 | 0.87 | 0.74 | 0.70 |
| **GxA-Out** | 0.94 | 0.96 | 0.94 | 1.00 | 0.95 | 0.85 | 0.86 | 0.73 | 0.70 |
| **Act-Out** | 0.93 | 0.94 | 0.91 | 0.95 | 1.00 | 0.83 | 0.86 | 0.74 | 0.70 |
| **GradMag-GI** | 0.84 | 0.85 | 0.86 | 0.85 | 0.83 | 1.00 | 0.80 | 0.72 | 0.70 |
| **LRP-CO** | 0.91 | 0.88 | 0.87 | 0.86 | 0.86 | 0.80 | 1.00 | 0.74 | 0.70 |
| **Act-GI** | 0.74 | 0.74 | 0.74 | 0.73 | 0.74 | 0.72 | 0.74 | 1.00 | 0.70 |
| **Random** | 0.70 | 0.70 | 0.70 | 0.70 | 0.70 | 0.70 | 0.70 | 0.70 | 1.00 |

**What each pruned model preserves** — share of each score's mass kept (per-layer normalised; a random selection keeps 0.70):

| kept set of | seen by LRP-GI | seen by GxI-GI | seen by Act-Out | seen by Act-GI | seen by GradMag-GI | seen by LRP-CO | post-FT PPL |
|---|---|---|---|---|---|---|---|
| **LRP-GI** | 0.869 | 0.844 | 0.861 | 0.714 | 0.829 | 0.862 | 30.4 |
| **LRP-Out** | 0.866 | 0.843 | 0.865 | 0.712 | 0.831 | 0.853 | 30.4 |
| **GxI-GI** | 0.859 | 0.857 | 0.858 | 0.719 | 0.839 | 0.852 | 29.9 |
| **GxA-Out** | 0.864 | 0.847 | 0.866 | 0.712 | 0.832 | 0.850 | 30.6 |
| **Act-Out** | 0.860 | 0.843 | 0.870 | 0.714 | 0.825 | 0.845 | 31.0 |
| **GradMag-GI** | 0.836 | 0.831 | 0.832 | 0.711 | 0.871 | 0.826 | 31.6 |
| **LRP-CO** | 0.835 | 0.803 | 0.789 | 0.708 | 0.783 | 0.879 | 37.5 |
| **Act-GI** | 0.761 | 0.765 | 0.764 | 0.762 | 0.751 | 0.764 | 37.4 |
| **Random** | 0.696 | 0.696 | 0.697 | 0.698 | 0.695 | 0.696 | 41.3 |

Closeness to the gate-aware consensus (mean votes of the five gate-aware scores for a method's kept channels) vs post-FT perplexity across the 9 methods: Spearman rho = -0.73 (p = 0.025).


## 50% of channels removed (keep 768 of 1536 per layer; chance overlap = 0.50)

**Kept-set overlap** (1.00 = identical selection):

| | LRP-GI | LRP-Out | GxI-GI | GxA-Out | Act-Out | GradMag-GI | LRP-CO | Act-GI | Random |
|---|---|---|---|---|---|---|---|---|---|
| **LRP-GI** | 1.00 | 0.94 | 0.88 | 0.90 | 0.88 | 0.76 | 0.87 | 0.58 | 0.50 |
| **LRP-Out** | 0.94 | 1.00 | 0.88 | 0.94 | 0.91 | 0.77 | 0.81 | 0.57 | 0.50 |
| **GxI-GI** | 0.88 | 0.88 | 1.00 | 0.91 | 0.87 | 0.78 | 0.80 | 0.60 | 0.50 |
| **GxA-Out** | 0.90 | 0.94 | 0.91 | 1.00 | 0.92 | 0.77 | 0.79 | 0.57 | 0.50 |
| **Act-Out** | 0.88 | 0.91 | 0.87 | 0.92 | 1.00 | 0.75 | 0.77 | 0.58 | 0.50 |
| **GradMag-GI** | 0.76 | 0.77 | 0.78 | 0.77 | 0.75 | 1.00 | 0.69 | 0.56 | 0.50 |
| **LRP-CO** | 0.87 | 0.81 | 0.80 | 0.79 | 0.77 | 0.69 | 1.00 | 0.57 | 0.50 |
| **Act-GI** | 0.58 | 0.57 | 0.60 | 0.57 | 0.58 | 0.56 | 0.57 | 1.00 | 0.50 |
| **Random** | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 1.00 |

**What each pruned model preserves** — share of each score's mass kept (per-layer normalised; a random selection keeps 0.50):

| kept set of | seen by LRP-GI | seen by GxI-GI | seen by Act-Out | seen by Act-GI | seen by GradMag-GI | seen by LRP-CO | post-FT PPL |
|---|---|---|---|---|---|---|---|
| **LRP-GI** | 0.732 | 0.699 | 0.719 | 0.526 | 0.684 | 0.724 | 45.5 |
| **LRP-Out** | 0.728 | 0.700 | 0.727 | 0.525 | 0.688 | 0.707 | 45.9 |
| **GxI-GI** | 0.718 | 0.717 | 0.718 | 0.534 | 0.696 | 0.705 | 46.7 |
| **GxA-Out** | 0.723 | 0.705 | 0.728 | 0.526 | 0.691 | 0.701 | 47.0 |
| **Act-Out** | 0.717 | 0.698 | 0.735 | 0.530 | 0.678 | 0.692 | 45.6 |
| **GradMag-GI** | 0.684 | 0.677 | 0.680 | 0.523 | 0.746 | 0.669 | 51.7 |
| **LRP-CO** | 0.699 | 0.658 | 0.642 | 0.522 | 0.637 | 0.747 | 60.5 |
| **Act-GI** | 0.585 | 0.595 | 0.595 | 0.586 | 0.572 | 0.586 | 59.7 |
| **Random** | 0.497 | 0.497 | 0.499 | 0.500 | 0.499 | 0.496 | 76.0 |

Closeness to the gate-aware consensus (mean votes of the five gate-aware scores for a method's kept channels) vs post-FT perplexity across the 9 methods: Spearman rho = -0.73 (p = 0.025).


## 70% of channels removed (keep 464 of 1536 per layer; chance overlap = 0.30)

**Kept-set overlap** (1.00 = identical selection):

| | LRP-GI | LRP-Out | GxI-GI | GxA-Out | Act-Out | GradMag-GI | LRP-CO | Act-GI | Random |
|---|---|---|---|---|---|---|---|---|---|
| **LRP-GI** | 1.00 | 0.89 | 0.82 | 0.84 | 0.79 | 0.66 | 0.83 | 0.44 | 0.30 |
| **LRP-Out** | 0.89 | 1.00 | 0.82 | 0.91 | 0.86 | 0.67 | 0.72 | 0.45 | 0.30 |
| **GxI-GI** | 0.82 | 0.82 | 1.00 | 0.87 | 0.81 | 0.67 | 0.71 | 0.51 | 0.30 |
| **GxA-Out** | 0.84 | 0.91 | 0.87 | 1.00 | 0.89 | 0.67 | 0.68 | 0.47 | 0.30 |
| **Act-Out** | 0.79 | 0.86 | 0.81 | 0.89 | 1.00 | 0.64 | 0.64 | 0.49 | 0.30 |
| **GradMag-GI** | 0.66 | 0.67 | 0.67 | 0.67 | 0.64 | 1.00 | 0.58 | 0.42 | 0.30 |
| **LRP-CO** | 0.83 | 0.72 | 0.71 | 0.68 | 0.64 | 0.58 | 1.00 | 0.43 | 0.30 |
| **Act-GI** | 0.44 | 0.45 | 0.51 | 0.47 | 0.49 | 0.42 | 0.43 | 1.00 | 0.30 |
| **Random** | 0.30 | 0.30 | 0.30 | 0.30 | 0.30 | 0.30 | 0.30 | 0.30 | 1.00 |

**What each pruned model preserves** — share of each score's mass kept (per-layer normalised; a random selection keeps 0.30):

| kept set of | seen by LRP-GI | seen by GxI-GI | seen by Act-Out | seen by Act-GI | seen by GradMag-GI | seen by LRP-CO | post-FT PPL |
|---|---|---|---|---|---|---|---|
| **LRP-GI** | 0.549 | 0.509 | 0.527 | 0.332 | 0.499 | 0.544 | 86.0 |
| **LRP-Out** | 0.541 | 0.509 | 0.544 | 0.334 | 0.505 | 0.512 | 85.6 |
| **GxI-GI** | 0.529 | 0.532 | 0.531 | 0.344 | 0.511 | 0.514 | 82.7 |
| **GxA-Out** | 0.534 | 0.516 | 0.548 | 0.336 | 0.506 | 0.503 | 86.1 |
| **Act-Out** | 0.524 | 0.508 | 0.555 | 0.340 | 0.491 | 0.488 | 83.7 |
| **GradMag-GI** | 0.487 | 0.480 | 0.485 | 0.329 | 0.576 | 0.473 | 98.5 |
| **LRP-CO** | 0.522 | 0.481 | 0.461 | 0.329 | 0.465 | 0.571 | 107.7 |
| **Act-GI** | 0.402 | 0.415 | 0.416 | 0.394 | 0.385 | 0.403 | 114.9 |
| **Random** | 0.298 | 0.300 | 0.301 | 0.301 | 0.302 | 0.297 | 216.5 |

Closeness to the gate-aware consensus (mean votes of the five gate-aware scores for a method's kept channels) vs post-FT perplexity across the 9 methods: Spearman rho = -0.70 (p = 0.036).


## Seed stability (same method, different calibration sample): kept-set overlap between seeds

| method | 30% | 50% | 70% |
|---|---|---|---|
| LRP-GI | 0.98 | 0.97 | 0.96 |
| LRP-Out | 0.97 | 0.97 | 0.96 |
| GxI-GI | 0.97 | 0.97 | 0.96 |
| GxA-Out | 0.98 | 0.97 | 0.96 |
| Act-Out | 0.98 | 0.97 | 0.97 |
| GradMag-GI | 0.94 | 0.92 | 0.90 |
| LRP-CO | 0.98 | 0.96 | 0.95 |
| Act-GI | 0.99 | 0.98 | 0.98 |
| Random | 0.70 | 0.50 | 0.30 |
