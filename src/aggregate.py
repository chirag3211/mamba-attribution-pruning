#!/usr/bin/env python3
"""
Aggregate grid runs (any number of directories, e.g. one per Colab account) into tables, paired statistics and figures.

  python aggregate.py --runs acct1/grid_out acct2/grid_out acct3/grid_out --out report [--protocol main]

Statistics.  Seeds are the unit of replication.  All methods in a seed share the calibration sample, the fine-tuning
data order and the evaluation set, so comparisons are PAIRED:  for contrast A-vs-B the per-seed difference
d_s = log PPL_A(s) - log PPL_B(s) is formed; we report exp(mean d)-1 as a percent change in perplexity (negative = A
is better), a 95% t-interval over seeds, and the number of seeds in which A beats B.  With few seeds the t-interval is
wide (n=3: t=4.30) -- that is the honest price of 3 seeds, and a verdict is only given when the interval excludes 0.

C1-C8 below were fixed before the grid was run; C9-C12 were added after seeing seed 0 and are exploratory, as is everything else (e.g. the full pairwise matrix).
"""
import argparse, glob, json, math, os, sys

import numpy as np

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}

# (id, A, B, question)   -- A "better" means lower perplexity than B
CONTRASTS = [
    ("C1", "LRP-GI",   "LRP-CO",     "Does adding the gate path help LRP scoring?"),
    ("C2", "LRP-GI",   "GxI-GI",     "Do the LRP propagation rules help beyond plain gradient x input (same sites, in_proj)?"),
    ("C3", "LRP-Out",  "GxA-Out",    "Do the LRP rules help beyond plain gradient x activation at the post-gate site?"),
    ("C4", "LRP-GI",   "GradMag-GI", "LRP-GI vs gate-inclusive Taylor/gradient-magnitude (the H15 comparison)"),
    ("C5", "LRP-GI",   "Act-Out",    "LRP-GI vs the cheapest post-gate baseline (mean |activation| at out_proj input)"),
    ("C6", "LRP-Out",  "Act-Out",    "Does LRP beat plain activation at the SAME post-gate site?"),
    ("C7", "Act-Out",  "Act-GI",     "Is post-gate activation better than pre-gate (in_proj) activation?"),
    ("C8", "LRP-GI",   "Random",     "Is LRP-GI better than random channel removal?"),
    # C9-C11 were added AFTER looking at seed 0 (same gate question as C1, for the other score families): treat as exploratory
    ("C9",  "GxI-GI",     "GxI-CO",     "[added after seed 0] gate path for plain gradient x input"),
    ("C10", "Act-GI",     "Act-CO",     "[added after seed 0] gate path for pre-gate activation"),
    ("C11", "GradMag-GI", "GradMag-CO", "[added after seed 0] gate path for Taylor/gradient-magnitude"),
    ("C13", "Hyb-L0GI",   "LRP-CO",     "[after the overlap analysis] does the gate in LAYER 0 ALONE recover the gate effect? (content-only LRP everywhere else)"),
    ("C14", "Hyb-L0GI",   "LRP-GI",     "[after the overlap analysis] how much of the gate effect is NOT in layer 0?"),
    ("C15", "Hyb-L0CO",   "LRP-GI",     "[after the overlap analysis] does removing the gate in LAYER 0 ALONE hurt gate-aware LRP?"),
    ("C16", "LRP-GI",     "Mag-GI",     "[added after the grid] LRP-GI vs gate-inclusive weight magnitude (the classic baseline)"),
    ("C17", "LRP-GI",     "Mag-CO",     "[added after the grid] LRP-GI vs content-only weight magnitude (the H3 definition)"),
    ("C18", "Mag-GI",     "Random",     "[added after the grid] is weight magnitude better than random channel removal?"),
    ("C12", "GxI-GI",     "LRP-GI",     "[added after seed 0] plain gradient x input vs LRP-GI (reverse of C2: is plain >= LRP?)"),
]
ORDER = ["Dense", "Random", "Mag-CO", "Mag-GI", "LRP-CO", "Hyb-L0GI", "Hyb-L0CO", "LRP-GI", "LRP-Out", "GxI-CO", "GxI-GI", "GxA-Out", "GradMag-CO", "GradMag-GI",
         "Act-CO", "Act-GI", "Act-Out"]


def load_runs(dirs, protocol):
    runs = {}
    for d in dirs:
        for p in glob.glob(os.path.join(d, "**", "runs", protocol, "*.json"), recursive=True) + \
                 glob.glob(os.path.join(d, "runs", protocol, "*.json")):
            r = json.load(open(p))
            key = (r["seed"], r["method"], round(r["sparsity"], 3))
            if key not in runs or r["time"] > runs[key]["time"]:
                runs[key] = r
    return runs


def paired(runs, A, B, sp, field):
    seeds = sorted({s for (s, m, q) in runs if m == A and q == sp} & {s for (s, m, q) in runs if m == B and q == sp})
    d = [math.log(runs[(s, A, sp)][field]) - math.log(runs[(s, B, sp)][field]) for s in seeds]
    return seeds, d


def summarize(d):
    n = len(d)
    if n == 0:
        return None
    mean = float(np.mean(d))
    if n < 2:
        return dict(n=n, pct=(math.exp(mean) - 1) * 100, lo=float("nan"), hi=float("nan"), wins=int(sum(x < 0 for x in d)))
    sd = float(np.std(d, ddof=1))
    half = T975.get(n - 1, 1.96) * sd / math.sqrt(n)
    return dict(n=n, pct=(math.exp(mean) - 1) * 100, lo=(math.exp(mean - half) - 1) * 100,
                hi=(math.exp(mean + half) - 1) * 100, wins=int(sum(x < 0 for x in d)))


MARGIN = 2.0      # percent of perplexity treated as practically negligible (set with --margin)


def verdict(s, A, B):
    if s is None or s["n"] < 2:
        return "n/a (need >=2 seeds)"
    if s["lo"] > -MARGIN and s["hi"] < MARGIN:
        return f"equivalent (within +-{MARGIN:g}%)"
    if s["hi"] < 0:
        return f"{A} better"
    if s["lo"] > 0:
        return f"{B} better"
    return "inconclusive"


def fmt_pct(s):
    if s is None:
        return "-"
    ci = "" if math.isnan(s["lo"]) else f" [{s['lo']:+.1f}, {s['hi']:+.1f}]"
    return f"{s['pct']:+.1f}%{ci}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="one or more grid output directories")
    ap.add_argument("--out", default="report")
    ap.add_argument("--protocol", default="main")
    ap.add_argument("--margin", type=float, default=2.0, help="percent perplexity difference treated as negligible")
    a = ap.parse_args()
    global MARGIN
    MARGIN = a.margin
    runs = load_runs(a.runs, a.protocol)
    if not runs:
        sys.exit("no run files found")
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    methods = [m for m in ORDER if any(k[1] == m for k in runs)] + sorted({k[1] for k in runs} - set(ORDER))
    sps = sorted({k[2] for k in runs if k[2] > 0})
    seeds = sorted({k[0] for k in runs})
    dense_ppl = None
    for d in a.runs:
        p = os.path.join(d, f"dense_baseline_{a.protocol}.json")
        if os.path.exists(p):
            dense_ppl = json.load(open(p))["ppl"]; break

    L = [f"# Grid results  (protocol `{a.protocol}`, seeds {seeds}, {len(runs)} runs)\n"]
    if dense_ppl:
        L.append(f"Unpruned model perplexity on the evaluation set: **{dense_ppl:.2f}**.\n")
    dk = [k for k in runs if k[1] == "Dense"]
    if dk:
        v = [runs[k]["ppl_post"] for k in dk]
        L.append(f"Dense control after the same fine-tuning: {np.exp(np.mean(np.log(v))):.2f} (n={len(v)}).\n")
    r0 = next(iter(runs.values()))
    L.append(f"Fine-tuning: {r0['ft']}.  Evaluation tokens: {r0['eval_tokens']:,}.  Calibration texts: {r0['calib_n']}.\n")

    # ---- main table -------------------------------------------------------------------------------------------
    csv = ["sparsity,method,n_seeds,ppl_pre_geomean,ppl_post_geomean,ppl_post_sd_log,vs_random_pct"]
    L.append("## 1. Perplexity by method and sparsity\n")
    L.append("Geometric mean over seeds; post-FT is after fine-tuning. `vs Random` = paired change in post-FT perplexity "
             "(negative = better than random). `vs dense` = change relative to the UNPRUNED model given the same fine-tuning "
             "(the right reference: fine-tuning on the training corpus also improves the dense model).\n")
    for sp in sps:
        L.append(f"### Sparsity {sp:.0%}\n")
        L.append("| method | seeds | pre-FT PPL | post-FT PPL | spread (sd of log) | vs Random | vs dense-FT |")
        L.append("|---|---|---|---|---|---|---|")
        rows = []
        for m in methods:
            if m == "Dense":
                continue
            pre = [runs[k]["ppl_pre"] for k in runs if k[1] == m and k[2] == sp]
            post = [runs[k]["ppl_post"] for k in runs if k[1] == m and k[2] == sp]
            if not post:
                continue
            gp, gq = float(np.exp(np.mean(np.log(pre)))), float(np.exp(np.mean(np.log(post))))
            sd = float(np.std(np.log(post), ddof=1)) if len(post) > 1 else float("nan")
            vr = summarize(paired(runs, m, "Random", sp, "ppl_post")[1]) if m != "Random" else None
            dsd = [math.log(runs[(k[0], m, sp)]["ppl_post"]) - math.log(runs[(k[0], "Dense", 0.0)]["ppl_post"])
                   for k in runs if k[1] == m and k[2] == sp and (k[0], "Dense", 0.0) in runs]
            vd = summarize(dsd)
            rows.append((gq, m, len(post), gp, gq, sd, vr, vd))
            csv.append(f"{sp},{m},{len(post)},{gp:.4f},{gq:.4f},{sd:.5f},{'' if vr is None else round(vr['pct'], 3)}")
        for _, m, n, gp, gq, sd, vr, vd in sorted(rows, key=lambda r: r[0]):
            L.append(f"| {m} | {n} | {gp:,.1f} | **{gq:,.1f}** | {sd:.3f} | {fmt_pct(vr) if vr else '-'} | {fmt_pct(vd) if vd else '-'} |")
        L.append("")

    # ---- pre-specified contrasts ---------------------------------------------------------------------------
    L.append("## 2. Pre-specified paired contrasts\n")
    L.append("Percent change in perplexity of A relative to B (negative = A better), 95% t-interval over seeds, "
             "seeds in which A < B. Post-FT is the primary endpoint; pre-FT (one-shot) is secondary. Verdict: *equivalent* if the whole "
             f"interval lies within +-{MARGIN:g}%; otherwise A/B better if the interval excludes 0; otherwise inconclusive.\n")
    L.append("| id | A vs B | sparsity | post-FT change [95% CI] | wins | verdict | pre-FT change [95% CI] |")
    L.append("|---|---|---|---|---|---|---|")
    ccsv = ["id,A,B,sparsity,n,post_pct,post_lo,post_hi,wins,verdict,pre_pct,pre_lo,pre_hi"]
    forest = []
    for cid, A, B, q in CONTRASTS:
        if not any(k[1] == A for k in runs) or not any(k[1] == B for k in runs):
            continue
        for sp in sps:
            sp_post = summarize(paired(runs, A, B, sp, "ppl_post")[1])
            sp_pre = summarize(paired(runs, A, B, sp, "ppl_pre")[1])
            if sp_post is None:
                continue
            v = verdict(sp_post, A, B)
            L.append(f"| {cid} | {A} vs {B} | {sp:.0%} | {fmt_pct(sp_post)} | {sp_post['wins']}/{sp_post['n']} | {v} | {fmt_pct(sp_pre)} |")
            ccsv.append(f"{cid},{A},{B},{sp},{sp_post['n']},{sp_post['pct']:.3f},{sp_post['lo']:.3f},{sp_post['hi']:.3f},"
                        f"{sp_post['wins']},{v},{sp_pre['pct']:.3f},{sp_pre['lo']:.3f},{sp_pre['hi']:.3f}")
            forest.append((f"{cid} {A} vs {B} @{sp:.0%}", sp_post))
    L.append("")
    L.append("Contrast questions: " + "; ".join(f"**{c}** {q}" for c, _, _, q in CONTRASTS) + "\n")

    # ---- cost -----------------------------------------------------------------------------------------------
    L.append("## 3. Cost of the scores\n")
    L.append("Wall-clock cost is not recorded per score in the grid; from earlier runs on a T4: LRP/activation ~3.4 min "
             "(one backward sweep over 256 sequences), activation alone ~4x cheaper (forward only).\n")

    # ---- figures ---------------------------------------------------------------------------------------------
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        cmap = plt.get_cmap("tab10")
        for ci, m in enumerate([m for m in methods if m != "Dense"]):
            xs, ys, lo, hi = [], [], [], []
            for sp in sps:
                v = [runs[k]["ppl_post"] for k in runs if k[1] == m and k[2] == sp]
                if v:
                    xs.append(sp * 100); ys.append(np.exp(np.mean(np.log(v)))); lo.append(min(v)); hi.append(max(v))
            if xs:
                ax.plot(xs, ys, "-o", color=cmap(ci % 10), label=m, ms=4)
                ax.fill_between(xs, lo, hi, color=cmap(ci % 10), alpha=0.12, lw=0)
        ax.set_yscale("log"); ax.set_xlabel("channels removed (%)"); ax.set_ylabel("post-fine-tuning perplexity (log)")
        ax.set_xticks([s * 100 for s in sps]); ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
        ax.set_title("Post-FT perplexity vs sparsity (geometric mean, band = min..max over seeds)")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "figures", "ppl_vs_sparsity.png"), dpi=160); plt.close(fig)

        if forest:
            fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(forest) + 1.2))
            for i, (lab, s) in enumerate(reversed(forest)):
                lo = s["lo"] if not math.isnan(s["lo"]) else s["pct"]; hi = s["hi"] if not math.isnan(s["hi"]) else s["pct"]
                col = "tab:green" if hi < 0 else ("tab:red" if lo > 0 else "tab:gray")
                ax.plot([lo, hi], [i, i], color=col, lw=2); ax.plot(s["pct"], i, "o", color=col, ms=4)
            ax.axvline(0, color="k", lw=0.8); ax.set_yticks(range(len(forest))); ax.set_yticklabels([f[0] for f in reversed(forest)], fontsize=7)
            ax.set_xlabel("change in post-FT perplexity of A vs B (%), 95% CI   (left of 0 = A better)")
            fig.tight_layout(); fig.savefig(os.path.join(a.out, "figures", "contrasts_forest.png"), dpi=160); plt.close(fig)

        mids = sorted({int(s) for r in runs.values() for s in r.get("ppl_mid", {})})
        if mids:
            fig, axs = plt.subplots(1, len(sps), figsize=(4.2 * len(sps), 3.8), squeeze=False)
            steps_all = [0] + mids + [r0["ft"]["steps"]]
            for j, sp in enumerate(sps):
                ax = axs[0][j]
                for ci, m in enumerate([m for m in methods if m != "Dense"]):
                    ys = []
                    for st in steps_all:
                        key = "ppl_pre" if st == 0 else ("ppl_post" if st == r0["ft"]["steps"] else None)
                        v = [(runs[k][key] if key else runs[k]["ppl_mid"].get(str(st))) for k in runs if k[1] == m and k[2] == sp]
                        v = [x for x in v if x]
                        ys.append(np.exp(np.mean(np.log(v))) if v else np.nan)
                    ax.plot(range(len(steps_all)), ys, "-o", ms=3, color=cmap(ci % 10), label=m)
                ax.set_yscale("log"); ax.set_xticks(range(len(steps_all))); ax.set_xticklabels(steps_all)
                ax.set_title(f"{sp:.0%} removed"); ax.set_xlabel("fine-tuning steps"); ax.grid(alpha=0.3)
            axs[0][0].set_ylabel("perplexity"); axs[0][-1].legend(fontsize=6)
            fig.tight_layout(); fig.savefig(os.path.join(a.out, "figures", "recovery_curves.png"), dpi=160); plt.close(fig)
        L.append("Figures: `figures/ppl_vs_sparsity.png`, `figures/contrasts_forest.png`" + (", `figures/recovery_curves.png`" if mids else "") + ".\n")
    except Exception as e:                                         # plotting must never block the tables
        L.append(f"(figures skipped: {type(e).__name__}: {e})\n")

    # ---- score analysis (needs the cached scores) -------------------------------------------------------------
    try:
        import torch
        from scipy.stats import spearmanr
        sc = {}
        for d in a.runs:
            for p in glob.glob(os.path.join(d, "scores", f"{a.protocol}_seed*.pt")):
                sc[(d, p)] = torch.load(p)
        if sc:
            names = [m for m in ORDER if m not in ("Dense",) and all(m in v for v in sc.values())]
            M = np.zeros((len(names), len(names)))
            for v in sc.values():
                for i, x in enumerate(names):
                    for j, y in enumerate(names):
                        M[i, j] += spearmanr(v[x].flatten().numpy(), v[y].flatten().numpy())[0] / len(sc)
            L.append("## 4. How similar are the rankings? (Spearman rho over all 24x1536 channels, mean over seeds)\n")
            L.append("| | " + " | ".join(names) + " |"); L.append("|---|" + "---|" * len(names))
            for i, x in enumerate(names):
                L.append(f"| **{x}** | " + " | ".join(f"{M[i, j]:.2f}" for j in range(len(names))) + " |")
            L.append("")
            try:
                fig, ax = plt.subplots(figsize=(7, 5.5))
                im = ax.imshow(M, vmin=-0.2, vmax=1, cmap="viridis")
                ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
                ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
                for i in range(len(names)):
                    for j in range(len(names)):
                        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5, color="w")
                fig.colorbar(im); fig.tight_layout(); fig.savefig(os.path.join(a.out, "figures", "score_correlation.png"), dpi=160); plt.close(fig)
                # per-layer share of total score mass: where does each method put its importance?
                fig, ax = plt.subplots(figsize=(7.5, 4))
                for ci, m in enumerate([n for n in ("LRP-GI", "LRP-Out", "GradMag-GI", "Act-Out", "Act-GI") if n in names]):
                    share = np.mean([(v[m].sum(1) / v[m].sum()).numpy() for v in sc.values()], axis=0)
                    ax.plot(range(len(share)), share, "-o", ms=3, label=m)
                ax.set_xlabel("layer"); ax.set_ylabel("share of total score mass"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
                ax.set_title("Where each score puts its importance across depth")
                fig.tight_layout(); fig.savefig(os.path.join(a.out, "figures", "layer_score_share.png"), dpi=160); plt.close(fig)
                L.append("Figures: `figures/score_correlation.png`, `figures/layer_score_share.png`.\n")
            except Exception as e:
                L.append(f"(score figures skipped: {e})\n")
    except Exception as e:
        L.append(f"(score analysis skipped: {type(e).__name__}: {e})\n")

    open(os.path.join(a.out, "summary.md"), "w").write("\n".join(L))
    open(os.path.join(a.out, "results.csv"), "w").write("\n".join(csv))
    open(os.path.join(a.out, "contrasts.csv"), "w").write("\n".join(ccsv))
    print("\n".join(L[:60]))
    print(f"\nwrote {a.out}/summary.md, results.csv, contrasts.csv, figures/")


if __name__ == "__main__":
    main()
