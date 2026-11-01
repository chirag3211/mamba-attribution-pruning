#!/usr/bin/env python3
"""
Is the fast-kernel runtime usable for fine-tuning + evaluating PRUNED mamba-130m?   (run in the torch-2.10 + mamba_ssm runtime)

Checks, for 30 / 50 / 70 % pruning (random scores; kernel behaviour does not depend on which channels are kept):
  1. the fused-kernel forward of the pruned model equals the slow reference forward (MambaLRP wrapper, pure PyTorch);
  2. the fused-kernel BACKWARD runs (fine-tuning needs it) and gives finite gradients;
  3. speed: seconds per fine-tuning step (batch 8x128) and per evaluation batch (8x256), and the implied time per grid run;
both with the exact pruned width (e.g. 1075, 461) and with widths rounded to a multiple of 8 (--keep-multiple 8).
"""
import argparse, os, sys, time, types
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--mambalrp-path", default="/content/MambaLRP")
ap.add_argument("--model", default="state-spaces/mamba-130m-hf")
ap.add_argument("--tiny", action="store_true", help="offline plumbing test")
a = ap.parse_args()
sys.path.insert(0, a.mambalrp_path)
import mamba_grid as G
from mamba_lrp.model.mamba_huggingface import ModifiedMambaForCausalLM

dev = "cuda" if torch.cuda.is_available() else "cpu"
try:
    import mamba_ssm, causal_conv1d
    print("mamba_ssm", mamba_ssm.__version__, "| causal_conv1d", causal_conv1d.__version__, "| torch", torch.__version__)
except ImportError:
    print("!! mamba_ssm / causal_conv1d NOT importable: this is the slow runtime; the check only tests plumbing here.")
if a.tiny:
    base, _, _ = G.make_tiny(types.SimpleNamespace(protocol="main")); vocab = 97
else:
    from transformers import MambaForCausalLM
    base = MambaForCausalLM.from_pretrained(a.model); vocab = base.config.vocab_size
base = base.to(dev).eval()
L, D = base.config.num_hidden_layers, base.config.intermediate_size
g = torch.Generator().manual_seed(0)
ids_eval = torch.randint(0, vocab, (8, 256 if not a.tiny else 32), generator=g).to(dev)
ids_ft = torch.randint(0, vocab, (8, 128 if not a.tiny else 16), generator=g).to(dev)


def sync():
    if dev == "cuda": torch.cuda.synchronize()


results = {}
for mult in (1, 8):
    for sp in (0.3, 0.5, 0.7):
        tag = f"sparsity {sp:.0%}, multiple={mult}"
        rec = {}
        try:
            pm = G.prune_model(base, torch.rand(L, D, generator=g), 1 - sp, dev, mult)
            n_keep = pm.backbone.layers[0].mixer.intermediate_size
            with torch.no_grad():
                f = pm(input_ids=ids_eval, use_cache=False).logits.float()
                slow = ModifiedMambaForCausalLM(pm, is_fast_forward_available=False).to(dev).eval()
                s = slow(input_ids=ids_eval); s = (s.logits if hasattr(s, "logits") else s).float()
            rec["rel_err"] = ((f - s).abs().max() / s.abs().max()).item()
            # backward through the training path
            pm.train(); opt = torch.optim.AdamW(pm.parameters(), lr=1e-5)
            def step():
                opt.zero_grad(set_to_none=True)
                lg = pm(input_ids=ids_ft, use_cache=False).logits
                loss = F.cross_entropy(lg[:, :-1].transpose(1, 2), ids_ft[:, 1:]); loss.backward(); opt.step()
                return loss.item()
            for _ in range(2): step()
            sync(); t = time.time(); [step() for _ in range(5)]; sync(); rec["ft_step_s"] = (time.time() - t) / 5
            rec["finite_grads"] = all(torch.isfinite(p.grad).all().item() for p in pm.parameters() if p.grad is not None)
            pm.eval()
            with torch.no_grad():
                pm(input_ids=ids_eval, use_cache=False); sync(); t = time.time()
                for _ in range(3): pm(input_ids=ids_eval, use_cache=False)
                sync(); rec["eval_batch_s"] = (time.time() - t) / 3
            rec["n_keep"] = n_keep; rec["ok"] = rec["rel_err"] < 2e-3 and rec["finite_grads"]
            del pm, slow
        except Exception as e:
            rec = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:160]}"}
        results[(sp, mult)] = rec
        if dev == "cuda": torch.cuda.empty_cache()
        print(f"{tag:28s}", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)

print("\n================ VERDICT ================")
for mult in (1, 8):
    oks = [results[(sp, mult)].get("ok") for sp in (0.3, 0.5, 0.7)]
    print(f"keep-multiple={mult}: {'ALL OK' if all(oks) else 'FAILED for some sparsity'}")
good = 1 if all(results[(sp, 1)].get("ok") for sp in (0.3, 0.5, 0.7)) else (8 if all(results[(sp, 8)].get("ok") for sp in (0.3, 0.5, 0.7)) else None)
if good is None:
    print("-> kernels cannot be used for the pruned models: use PLAN B (slow path, reduced budget).")
else:
    r = results[(0.5, good)]
    for steps, batch, ev_tok, n_ev in [(500, 8, 290_000, 4), (1000, 8, 290_000, 4)]:
        mins = (steps * r["ft_step_s"] * (batch / 8) + n_ev * (ev_tok / 256 / 8) * r["eval_batch_s"]) / 60
        print(f"-> PLAN A usable (use --keep-multiple {good}): {steps} steps, batch {batch}, full test set: "
              f"~{mins:.1f} min per run, ~{mins * 28 / 60:.1f} h for 28 runs per seed")
