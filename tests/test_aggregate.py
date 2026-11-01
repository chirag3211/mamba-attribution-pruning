"""Synthetic check of aggregate.py: known effects must give the right verdicts."""
import json, os, subprocess, sys, tempfile, math, random
rng = random.Random(0)
d = tempfile.mkdtemp(); os.makedirs(f"{d}/runs/main")
base = {"Random": 600, "LRP-CO": 250, "LRP-GI": 115, "LRP-Out": 117, "GxI-GI": 135, "GxA-Out": 119, "GradMag-GI": 121,
        "Act-Out": 117, "Act-GI": 184}       # true post-FT PPL at one sparsity
for seed in range(3):
    shared = rng.gauss(0, 0.03)              # seed-level shift shared by all methods (pairing must remove it)
    for m, v in base.items():
        for sp in (0.5,):
            post = v * math.exp(shared + rng.gauss(0, 0.002 if m in ('LRP-Out', 'Act-Out') else 0.01))
            json.dump(dict(protocol="main", seed=seed, method=m, sparsity=sp, ppl_pre=post * 3, ppl_post=post, ppl_mid={},
                           train_loss=[1], n_params=1, seconds=1, ft=dict(steps=500), eval_tokens=1, calib_n=256,
                           time=f"2026-01-01 00:00:0{seed}"), open(f"{d}/runs/main/seed{seed}_{m}_sp50.json", "w"))
_H = os.path.dirname(os.path.abspath(__file__))
AGG = next(p for p in (os.path.join(_H, "aggregate.py"), os.path.join(_H, "..", "src", "aggregate.py")) if os.path.exists(p))
out = subprocess.run([sys.executable, AGG, "--runs", d, "--out", f"{d}/rep"], capture_output=True, text=True)
assert out.returncode == 0, out.stderr
txt = open(f"{d}/rep/summary.md").read()
rows = {l.split("|")[1].strip(): l for l in txt.splitlines() if l.startswith("| C")}
def row(cid): return [l for l in txt.splitlines() if l.startswith(f"| {cid} ")][0]
print(row("C1")); print(row("C2")); print(row("C4")); print(row("C6")); print(row("C8"))
assert "LRP-GI better" in row("C1")          # 115 vs 250 : clearly better
assert "LRP-GI better" in row("C2")          # 115 vs 135 : ~15% better
assert "equivalent" in row("C6")             # 117 vs 117 with tiny noise: certified tie
assert "inconclusive" in row("C4") or "better" in row("C4")
assert "LRP-GI better" in row("C8")
# pairing: the shared seed shift (sd 3%) must not widen the interval of a near-tie much
print("AGGREGATOR TESTS PASSED")
