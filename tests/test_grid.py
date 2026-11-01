"""
Offline correctness tests for mamba_grid.py on a random tiny Mamba (CPU, no downloads).
  python test_grid.py [path/to/MambaLRP]
The reference implementations below are the ORIGINAL scoring functions from H15_gate_inclusive_comparison.py /
codebase_complete.py, transcribed unchanged except that tokenisation is replaced by pre-made (ids, mask) batches.
"""
import copy, os, sys, tempfile, types
import torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (HERE, os.path.join(HERE, '..', 'src')):          # flat Colab folder, or tests/ next to src/
    sys.path.insert(0, _d)
LRP_PATH = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MAMBALRP_PATH", "/content/MambaLRP")
sys.path.insert(0, LRP_PATH)
import mamba_grid as G
from mamba_lrp.model.mamba_huggingface import ModifiedMambaForCausalLM

args = types.SimpleNamespace(mambalrp_path=LRP_PATH, protocol="main")
base, _, D_ = G.make_tiny(args)
L, Dm = base.config.num_hidden_layers, base.config.intermediate_size
batches = D_.tiny_calib
# make one batch use padding so the mask matters
ids0, m0 = batches[0]; m0 = m0.clone(); m0[:, -5:] = 0; batches = [(ids0, m0)] + batches[1:]


def close(a, b, tol=1e-4, what=""):
    err = ((a - b).abs().max() / (b.abs().max() + 1e-12)).item()
    assert err < tol, f"{what}: relative max error {err:.2e}"
    return err


# ---------------- reference (original) scorers ----------------
def ref_lrp(gate_inclusive):
    wrapped = ModifiedMambaForCausalLM(base, is_fast_forward_available=False).eval()
    scores = torch.zeros(L, Dm); cap = {}
    def mk(i):
        def h(m, inp, out): out.retain_grad(); cap[i] = out
        return h
    hs = [wrapped.backbone.layers[i].mixer.in_proj.register_forward_hook(mk(i)) for i in range(L)]
    for ids, mask in batches:
        wrapped.zero_grad(set_to_none=True)
        lg = wrapped(input_ids=ids); lg = lg.logits if hasattr(lg, "logits") else lg
        (F.cross_entropy(lg[:, :-1].contiguous().transpose(1, 2), ids[:, 1:].contiguous(), reduction="none")
         * mask[:, 1:].float()).sum().backward()
        for i in range(L):
            c = cap[i]; m_ = mask.unsqueeze(-1).float()
            content = (c.detach()[..., :Dm] * c.grad[..., :Dm] * m_).abs().sum(dim=(0, 1))
            if gate_inclusive:
                gate = (c.detach()[..., Dm:] * c.grad[..., Dm:] * m_).abs().sum(dim=(0, 1))
                scores[i] += content + gate
            else:
                scores[i] += content
        cap.clear()
    for h in hs: h.remove()
    return scores


def ref_act(gate_inclusive):
    wrapped = ModifiedMambaForCausalLM(base, is_fast_forward_available=False).eval()
    scores = torch.zeros(L, Dm); cap = {}
    def mk(i):
        def h(m, inp, out): cap[i] = out.detach()
        return h
    hs = [wrapped.backbone.layers[i].mixer.in_proj.register_forward_hook(mk(i)) for i in range(L)]
    for ids, mask in batches:
        with torch.no_grad(): wrapped(input_ids=ids)
        for i in range(L):
            c = cap[i]; m_ = mask.unsqueeze(-1).float()
            content = (c[..., :Dm] * m_).abs().sum(dim=(0, 1))
            scores[i] += content + ((c[..., Dm:] * m_).abs().sum(dim=(0, 1)) if gate_inclusive else 0)
    for h in hs: h.remove()
    return scores


def ref_gradmag(gate_inclusive):
    model = copy.deepcopy(base); model.train(); model.zero_grad(set_to_none=True)
    for ids, mask in batches:
        out = model(input_ids=ids); lg = out.logits
        (F.cross_entropy(lg[:, :-1].contiguous().transpose(1, 2), ids[:, 1:].contiguous(), reduction="none")
         * mask[:, 1:].float()).sum().backward()
    scores = torch.zeros(L, Dm)
    for i, blk in enumerate(model.backbone.layers):
        w, g = blk.mixer.in_proj.weight.data, blk.mixer.in_proj.weight.grad
        c = (w[:Dm] * g[:Dm]).abs().mean(dim=1)
        scores[i] = c + (w[Dm:] * g[Dm:]).abs().mean(dim=1) if gate_inclusive else c
    return scores


def ref_gxi_content():
    model = copy.deepcopy(base); model.train(); cap = {}
    scores = torch.zeros(L, Dm)
    def mk(i):
        def h(m, inp, out): out.retain_grad(); cap[i] = out
        return h
    hs = [model.backbone.layers[i].mixer.in_proj.register_forward_hook(mk(i)) for i in range(L)]
    for ids, mask in batches:
        model.zero_grad(set_to_none=True)
        lg = model(input_ids=ids).logits
        (F.cross_entropy(lg[:, :-1].contiguous().transpose(1, 2), ids[:, 1:].contiguous(), reduction="none")
         * mask[:, 1:].float()).sum().backward()
        for i in range(L):
            c = cap[i]; m_ = mask.unsqueeze(-1).float()
            scores[i] += (c.detach()[..., :Dm] * c.grad[..., :Dm] * m_).abs().sum(dim=(0, 1))
        cap.clear()
    return scores


# ---------------- tests ----------------
def test_scores_match_original():
    A = G.scores_pass_A(base, batches, args, "cpu", G.PASS_A)
    B = G.scores_pass_B(base, batches, args, "cpu", G.PASS_B)
    for name, ref in [("LRP-CO", ref_lrp(False)), ("LRP-GI", ref_lrp(True)),
                      ("Act-CO", ref_act(False)), ("Act-GI", ref_act(True))]:
        print(f"  {name:10s} vs original: rel.err {close(A[name], ref, what=name):.1e}")
    for name, ref in [("GradMag-CO", ref_gradmag(False)), ("GradMag-GI", ref_gradmag(True)), ("GxI-CO", ref_gxi_content())]:
        print(f"  {name:10s} vs original: rel.err {close(B[name], ref, what=name):.1e}")
    # GI >= CO elementwise (it adds a non-negative gate term); the new scores are non-trivial and non-identical
    for k in ("LRP", "GxI"):
        co, gi = (A if k == "LRP" else B)[f"{k}-CO"], (A if k == "LRP" else B)[f"{k}-GI"]
        assert (gi >= co - 1e-6).all() and (gi > co).any(), k
    for k in ("LRP-Out", "Act-Out"):
        assert (A[k] > 0).all(), k
    assert (B["GxA-Out"] > 0).all()
    assert not torch.allclose(A["LRP-Out"], B["GxA-Out"]), "LRP rules should change the gradient at out_proj"
    return A, B


def test_act_out_is_post_gate():
    """Act-Out must equal mean|input of out_proj| measured on the plain model (no wrapper)."""
    model = copy.deepcopy(base).eval(); S = torch.zeros(L, Dm); cap = {}
    hs = [blk.mixer.out_proj.register_forward_pre_hook(lambda m, a, i=i: cap.__setitem__(i, a[0].detach()))
          for i, blk in enumerate(model.backbone.layers)]
    for ids, mask in batches:
        with torch.no_grad(): model(input_ids=ids, use_cache=False)
        for i in range(L): S[i] += (cap[i] * mask.unsqueeze(-1).float()).abs().sum((0, 1))
    for h in hs: h.remove()
    A = G.scores_pass_A(base, batches, args, "cpu", {"Act-Out"})
    print(f"  Act-Out vs plain-model out_proj input: rel.err {close(A['Act-Out'], S, what='Act-Out'):.1e}")


def test_lrp_wrapper_forward_equal():
    w = ModifiedMambaForCausalLM(base, is_fast_forward_available=False).eval()
    ids = batches[1][0]
    with torch.no_grad():
        a = w(input_ids=ids); a = a.logits if hasattr(a, "logits") else a
        b = base(input_ids=ids, use_cache=False).logits
    print(f"  LRP-wrapped forward vs stock forward: max|diff| {(a - b).abs().max().item():.1e}")
    assert (a - b).abs().max() < 1e-4


def test_pruning_exactly_equals_zeroed_channels():
    """Removing channel c must equal zeroing conv1d weight+bias of c (its conv/SiLU output becomes 0, so it feeds
    nothing into x_proj, the scan, or out_proj).  This is the correctness criterion for structural pruning."""
    g = torch.Generator().manual_seed(5)
    scores = torch.rand(L, Dm, generator=g)
    for sp in (0.3, 0.5, 0.7):
        pm = G.prune_model(base, scores, 1 - sp, "cpu")
        zm = copy.deepcopy(base).eval()
        keep = G.kept_indices(scores, 1 - sp, Dm)
        for i, blk in enumerate(zm.backbone.layers):
            drop = torch.ones(Dm, dtype=torch.bool); drop[keep[i]] = False
            blk.mixer.conv1d.weight.data[drop] = 0; blk.mixer.conv1d.bias.data[drop] = 0
        ids = batches[1][0]
        with torch.no_grad():
            d = (pm(input_ids=ids, use_cache=False).logits - zm(input_ids=ids, use_cache=False).logits).abs().max().item()
        print(f"  sparsity {sp:.0%}: pruned vs zeroed-channel model max|dlogit| = {d:.1e}")
        assert d < 1e-4
        n_keep = int(round(Dm * (1 - sp)))
        assert pm.backbone.layers[0].mixer.in_proj.out_features == 2 * n_keep
    pm = G.prune_model(base, scores, 1.0, "cpu")
    with torch.no_grad():
        d = (pm(input_ids=ids, use_cache=False).logits - base(input_ids=ids, use_cache=False).logits).abs().max().item()
    assert d < 1e-6, d


def test_finetune_both_protocols_and_resume():
    a = types.SimpleNamespace(protocol="main", ft_steps=12, ft_batch=2, ft_seq=16, ft_lr=2e-4, ft_warmup=3,
                              ft_micro=1, eval_at_list=[6])
    m = copy.deepcopy(base)
    m, curve, mids = G.fine_tune(m, D_, a, 0, "cpu", eval_cb=lambda mm: G.perplexity(mm, D_.eval_chunks, "cpu"))
    assert list(mids) == [6] and len(curve) > 0
    a.protocol = "legacy"; a.eval_at_list = []
    m, curve, mids = G.fine_tune(copy.deepcopy(base), D_, a, 0, "cpu")
    assert len(curve) > 0
    # pairing: the main-protocol data order must depend on the seed only (identical across methods)
    g1 = torch.randint(0, 100, (3, 2), generator=torch.Generator().manual_seed(7919 * 1 + 13))
    g2 = torch.randint(0, 100, (3, 2), generator=torch.Generator().manual_seed(7919 * 1 + 13))
    assert torch.equal(g1, g2)
    with tempfile.TemporaryDirectory() as d:
        argv = ["--tiny", "--out", d, "--mambalrp-path", LRP_PATH, "--seeds", "0", "--methods", "Random,LRP-GI,Dense",
                "--sparsities", "0.5", "--ft-steps", "4", "--ft-batch", "2", "--ft-seq", "16", "--ft-warmup", "2",
                "--eval-at", "", "--device", "cpu"]
        G.main(argv)
        n1 = len(os.listdir(os.path.join(d, "runs", "main")))
        mtimes = {f: os.path.getmtime(os.path.join(d, "runs", "main", f)) for f in os.listdir(os.path.join(d, "runs", "main"))}
        G.main(argv)                                                   # second call must be a no-op
        assert len(os.listdir(os.path.join(d, "runs", "main"))) == n1 == 3
        assert all(os.path.getmtime(os.path.join(d, "runs", "main", f)) == t for f, t in mtimes.items())
        # asking for an extra method later must reuse the cached scores and only run the new run
        G.main(argv[:argv.index("--methods") + 1] + ["Random,LRP-GI,Act-Out,Dense"] + argv[argv.index("--methods") + 2:])
        assert len(os.listdir(os.path.join(d, "runs", "main"))) == 4
    print("  resume / incremental behaviour OK")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name); fn()
    print("\nALL TESTS PASSED")
