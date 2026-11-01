"""How do the pruned models differ?  Offline analysis of WHICH channels each method keeps (no GPU, no model).

Inputs : <results>/scores/main_seed{S}.pt   (cached scores, 24 x 1536 per score)  and  <results>/results.csv
Outputs: <out>/compare.md, <out>/figures/{kept_overlap,retention,layer_overlap,overlap_vs_ppl}.png, <out>/compare.json

For every sparsity and seed it rebuilds the kept channel set of each method exactly as prune_model() does
(per-layer top-k with k = 1072 / 768 / 464 of 1536) and reports
  1. kept-set overlap (fraction of the kept channels two methods share, 1.0 = identical, chance = keep fraction);
  2. score retention: share of a score's total mass that a method's kept set preserves, seen through each score ("lens");
  3. per-layer overlap between content-only and gate-inclusive (and pre- vs post-gate activation) selections;
  4. whether closeness to the best methods' selection predicts post-fine-tuning perplexity (Spearman, across methods).
"""
import argparse, csv, json, math, os
import numpy as np
import torch
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ["LRP-GI", "LRP-Out", "GxI-GI", "GxA-Out", "Act-Out", "GradMag-GI", "LRP-CO", "Act-GI", "Random"]
GATE_AWARE = ["LRP-GI", "LRP-Out", "GxI-GI", "GxA-Out", "Act-Out"]
KEEP = {0.3: 1072, 0.5: 768, 0.7: 464}          # --keep-multiple 8 widths used in the grid
D, L = 1536, 24


def load_scores(res, seed):
    d = torch.load(os.path.join(res, "scores", f"main_seed{seed}.pt"), map_location="cpu")
    g = torch.Generator().manual_seed(10_000 + seed)
    d["Random"] = torch.rand(L, D, generator=g)
    return d


def kept_mask(score, k):
    m = torch.zeros(L, D, dtype=torch.bool)
    idx = torch.topk(score, k=k, dim=1, largest=True).indices
    m.scatter_(1, idx, True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    scores = {s: load_scores(a.results, s) for s in seeds}
    ppl = {}
    for r in csv.DictReader(open(os.path.join(a.results, "results.csv"))):
        ppl[(r["method"], round(float(r["sparsity"]), 2))] = float(r["ppl_post_geomean"])
    out = {}
    md = ["# How do the pruned models differ?\n",
          "Offline analysis of the kept channel sets (24 layers x 1,536 channels), mean over seeds "
          f"{seeds}. Kept sets are rebuilt exactly as the pruning code builds them (per-layer top-k).\n"]

    for sp, k in KEEP.items():
        masks = {s: {m: kept_mask(scores[s][m], k) for m in METHODS} for s in seeds}
        n_tot = L * k
        # 1. overlap matrix
        ov = np.zeros((len(METHODS), len(METHODS)))
        for s in seeds:
            for i, mi in enumerate(METHODS):
                for j, mj in enumerate(METHODS):
                    ov[i, j] += (masks[s][mi] & masks[s][mj]).sum().item() / n_tot
        ov /= len(seeds)
        # 2. retention: share of lens-score mass kept by each method's kept set
        lenses = ["LRP-GI", "GxI-GI", "Act-Out", "Act-GI", "GradMag-GI", "LRP-CO"]
        ret = np.zeros((len(METHODS), len(lenses)))
        for s in seeds:
            for i, mi in enumerate(METHODS):
                for j, lj in enumerate(lenses):
                    sc = scores[s][lj]
                    sc = sc / sc.sum(1, keepdim=True)            # per-layer normalised so every layer counts equally
                    ret[i, j] += ((sc * masks[s][mi]).sum(1)).mean().item()
        ret /= len(seeds)
        # 3. per-layer overlap
        def layer_ov(a_, b_):
            v = np.zeros(L)
            for s in seeds:
                v += (masks[s][a_] & masks[s][b_]).sum(1).numpy() / k
            return v / len(seeds)
        lay = {"LRP-CO vs LRP-GI": layer_ov("LRP-CO", "LRP-GI"), "Act-GI vs Act-Out": layer_ov("Act-GI", "Act-Out"),
               "GradMag-GI vs LRP-GI": layer_ov("GradMag-GI", "LRP-GI"), "Act-Out vs LRP-GI": layer_ov("Act-Out", "LRP-GI")}
        # 4. does closeness to the gate-aware consensus predict quality?
        cons = {s: sum(masks[s][m].int() for m in GATE_AWARE) for s in seeds}          # 0..5 votes
        close = {}
        for m in METHODS:
            v = 0
            for s in seeds:
                v += ((masks[s][m].int() * cons[s]).sum().item() / (n_tot * len(GATE_AWARE)))
            close[m] = v / len(seeds)
        ms = [m for m in METHODS]
        rho, p = spearmanr([close[m] for m in ms], [ppl[(m, sp)] for m in ms])
        # seed stability (same method, different seeds) -- only meaningful for data-dependent scores
        stab = {}
        for m in METHODS:
            vs = []
            for i in range(len(seeds)):
                for j in range(i + 1, len(seeds)):
                    vs.append((masks[seeds[i]][m] & masks[seeds[j]][m]).sum().item() / n_tot)
            stab[m] = float(np.mean(vs)) if vs else float("nan")
        out[str(sp)] = dict(overlap=ov.tolist(), retention=ret.tolist(), lenses=lenses, layer=lay_to_list(lay),
                            consensus_closeness=close, spearman_closeness_vs_ppl=[rho, p], seed_stability=stab, k=k)

        # ----- markdown
        md.append(f"\n## {int(sp * 100)}% of channels removed (keep {k} of {D} per layer; chance overlap = {k / D:.2f})\n")
        md.append("**Kept-set overlap** (1.00 = identical selection):\n")
        md.append("| | " + " | ".join(METHODS) + " |\n|---|" + "---|" * len(METHODS))
        for i, mi in enumerate(METHODS):
            md.append(f"| **{mi}** | " + " | ".join(f"{ov[i, j]:.2f}" for j in range(len(METHODS))) + " |")
        md.append("\n**What each pruned model preserves** — share of each score's mass kept (per-layer normalised; "
                  f"a random selection keeps {k / D:.2f}):\n")
        md.append("| kept set of | " + " | ".join(f"seen by {l}" for l in lenses) + " | post-FT PPL |\n|---|" + "---|" * (len(lenses) + 1))
        for i, mi in enumerate(METHODS):
            md.append(f"| **{mi}** | " + " | ".join(f"{ret[i, j]:.3f}" for j in range(len(lenses))) + f" | {ppl[(mi, sp)]:.1f} |")
        md.append(f"\nCloseness to the gate-aware consensus (mean votes of the five gate-aware scores for a method's kept channels) "
                  f"vs post-FT perplexity across the 9 methods: Spearman rho = {rho:.2f} (p = {p:.3f}).\n")

    json.dump(out, open(os.path.join(a.out, "compare.json"), "w"))

    # ----- figures
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.6))
    for ax, sp in zip(axs, KEEP):
        ov = np.array(out[str(sp)]["overlap"])
        im = ax.imshow(ov, vmin=KEEP[sp] / D, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(METHODS, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(METHODS))); ax.set_yticklabels(METHODS, fontsize=8)
        for i in range(len(METHODS)):
            for j in range(len(METHODS)):
                ax.text(j, i, f"{ov[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if ov[i, j] < 0.8 else "black")
        ax.set_title(f"{int(sp * 100)}% removed: share of kept channels in common")
    fig.colorbar(im, ax=axs, fraction=0.012)
    fig.savefig(os.path.join(a.out, "figures", "kept_overlap.png"), dpi=140, bbox_inches="tight"); plt.close(fig)


    # single large panel (50%) for the report / slides
    sp = 0.5
    ov = np.array(out[str(sp)]["overlap"])
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(ov, vmin=KEEP[sp] / D, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(METHODS, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(METHODS))); ax.set_yticklabels(METHODS, fontsize=10)
    for i in range(len(METHODS)):
        for j in range(len(METHODS)):
            ax.text(j, i, f"{ov[i, j]:.2f}", ha="center", va="center", fontsize=9, color="white" if ov[i, j] < 0.8 else "black")
    ax.set_title("50% removed: share of kept channels in common (chance 0.50)", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(os.path.join(a.out, "figures", "kept_overlap_50.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    for ax, sp in zip(axs, KEEP):
        ret = np.array(out[str(sp)]["retention"]); lens = out[str(sp)]["lenses"]
        im = ax.imshow(ret, cmap="magma_r", vmin=KEEP[sp] / D, vmax=1)
        ax.set_xticks(range(len(lens))); ax.set_xticklabels(lens, rotation=45, ha="right", fontsize=8); ax.set_xlabel("score used to measure what is preserved")
        ax.set_yticks(range(len(METHODS))); ax.set_yticklabels([f"{m}  ({ppl[(m, sp)]:.0f})" for m in METHODS], fontsize=8)
        for i in range(len(METHODS)):
            for j in range(len(lens)):
                ax.text(j, i, f"{ret[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="white" if ret[i, j] > 0.75 else "black")
        ax.set_title(f"{int(sp * 100)}% removed: score mass preserved\n(row label: kept-set method, post-FT PPL)", fontsize=10)
    fig.savefig(os.path.join(a.out, "figures", "retention.png"), dpi=140, bbox_inches="tight"); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for ax, sp in zip(axs, KEEP):
        for name, v in out[str(sp)]["layer"].items():
            ax.plot(range(L), v, marker="o", ms=3, label=name)
        ax.axhline(KEEP[sp] / D, color="gray", ls=":", label="chance")
        ax.set_title(f"{int(sp * 100)}% removed"); ax.set_xlabel("layer"); ax.set_ylim(0, 1.02)
        if sp == 0.3: ax.set_ylabel("share of kept channels in common")
    axs[0].legend(fontsize=7)
    fig.savefig(os.path.join(a.out, "figures", "layer_overlap.png"), dpi=140, bbox_inches="tight"); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, sp in zip(axs, KEEP):
        c = out[str(sp)]["consensus_closeness"]
        xs = [c[m] for m in METHODS]; ys = [ppl[(m, sp)] for m in METHODS]
        ax.scatter(xs, ys, color="#1E2761")
        for m, x, y in zip(METHODS, xs, ys): ax.annotate(m, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_yscale("log"); ax.set_title(f"{int(sp * 100)}% removed  (rho = {out[str(sp)]['spearman_closeness_vs_ppl'][0]:.2f})")
        ax.set_xlabel("closeness to gate-aware consensus selection"); ax.set_ylabel("post-FT perplexity")
    fig.savefig(os.path.join(a.out, "figures", "overlap_vs_ppl.png"), dpi=140, bbox_inches="tight"); plt.close(fig)

    md.append("\n## Seed stability (same method, different calibration sample): kept-set overlap between seeds\n")
    md.append("| method | " + " | ".join(f"{int(sp * 100)}%" for sp in KEEP) + " |\n|---|---|---|---|")
    for m in METHODS:
        md.append(f"| {m} | " + " | ".join(f"{out[str(sp)]['seed_stability'][m]:.2f}" for sp in KEEP) + " |")
    open(os.path.join(a.out, "compare.md"), "w").write("\n".join(md) + "\n")
    print("wrote", os.path.join(a.out, "compare.md"))


def lay_to_list(d):
    return {k: [float(x) for x in v] for k, v in d.items()}


if __name__ == "__main__":
    main()
