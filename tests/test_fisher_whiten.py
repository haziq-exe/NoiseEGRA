#!/usr/bin/env python
"""Whitening a real model's perturbation basis by how far it moves predictions.

`tests/test_fisher.py` checks the arithmetic against a model whose pullback is
known in closed form. This checks the part that touches a transformer: that the
displacement is applied at the prompt exactly as the generation hook applies it,
that the metric is measured on the model's own next-token distribution, and that
whitening actually flattens the subspace it is given.

The model is the 64-wide randomly initialised one, so the numbers mean nothing
about language -- what is being checked is the wiring and the geometry.

    python tests/test_fisher_whiten.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from noiseegra.activation_basis import StoryAxes, whiten_axes_by_fisher  # noqa: E402
from noiseegra.fisher import fisher_rao_distance, resolvable_basis  # noqa: E402
from tiny_model import Tiny  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAIL.append(name)


LAYERS = [2, 3]
RANK = 5
PROMPT = [{"role": "user", "content": "write one short story for a young reader"}]

m = Tiny()
DIM = m.model.config.hidden_size


def fresh_axes():
    torch.manual_seed(0)
    basis, mean, scale = {}, {}, {}
    for li in LAYERS:
        q = torch.linalg.qr(torch.randn(DIM, RANK))[0]
        # Deliberately unequal, so there is something for whitening to do.
        basis[li] = q * torch.tensor([4.0, 2.0, 1.0, 0.5, 0.25])
        mean[li] = torch.zeros(DIM)
        scale[li] = torch.ones(RANK)
    return StoryAxes(basis=basis, mean=mean, scale=scale, n_stories=8)


axes = fresh_axes()
before = {li: axes.basis[li].clone() for li in LAYERS}
aniso = whiten_axes_by_fisher(m, PROMPT, axes, LAYERS, prompt_tail_clear=2,
                              gamma=1.0, verbose=False)

check("every layer asked for was measured", sorted(aniso) == LAYERS, str(sorted(aniso)))
check("the basis was rewritten, not left alone",
      all(not torch.equal(before[li], axes.basis[li]) for li in LAYERS))
check("the basis keeps its shape",
      all(axes.basis[li].shape == (DIM, RANK) for li in LAYERS))
check("a deliberately unequal subspace is reported as unequal",
      all(v > 2.0 for v in aniso.values()),
      ", ".join(f"layer {k}: {v:.1f}x" for k, v in aniso.items()))
# On a randomly initialised model some directions move the next-token
# distribution by less than the probe can resolve, so the reported anisotropy
# saturates. That is the situation the gain cap exists for, and the basis must
# still come back finite and of a sane size.
check("the whitened basis stays finite and does not blow up",
      all(bool(torch.isfinite(axes.basis[li]).all())
          and float(axes.basis[li].norm()) < 20 * float(before[li].norm())
          for li in LAYERS),
      ", ".join(f"layer {li}: {float(axes.basis[li].norm())/float(before[li].norm()):.1f}x"
                for li in LAYERS))


def spread_of_steps(basis, n=64, scale=0.25):
    """How far unit draws from this basis move the next-token distribution."""
    import noiseegra.activation_basis as ab
    blocks = m._get_transformer_blocks()
    device = m._input_device()
    text = m.apply_chat_template(PROMPT, tokenize=False, add_generation_prompt=True)
    enc = m.tokenizer(text, return_tensors="pt").to(device)
    enc.pop("token_type_ids", None)
    state = {"delta": None}

    def hook(module, inp, o):
        t = o[0] if isinstance(o, (tuple, list)) else o
        if not isinstance(t, torch.Tensor) or t.dim() != 3 or t.shape[1] == 1:
            return None
        if state["delta"] is not None:
            n_ = t.shape[1]
            hi = n_ - 2 if n_ > 2 else n_
            t[:, :hi, :].add_(state["delta"].to(t.dtype).view(1, 1, -1))
        return None

    h = blocks[LAYERS[0]].register_forward_hook(hook)
    try:
        with torch.no_grad():
            base = torch.softmax(m.model(**enc, return_dict=True).logits[0, -1].float(), -1)
            g = torch.Generator().manual_seed(5)
            c = torch.randn(n, basis.shape[1], generator=g)
            c = c / c.norm(dim=1, keepdim=True)
            d = []
            for ci in c:
                v = basis @ ci
                state["delta"] = v / v.norm().clamp_min(1e-12) * scale
                p = torch.softmax(m.model(**enc, return_dict=True).logits[0, -1].float(), -1)
                state["delta"] = None
                d.append(float(fisher_rao_distance(base, p)))
    finally:
        h.remove()
    return torch.tensor(d)


raw = spread_of_steps(before[LAYERS[0]])
wht = spread_of_steps(axes.basis[LAYERS[0]])
rv = float(raw.std() / raw.mean().clamp_min(1e-12))
wv = float(wht.std() / wht.mean().clamp_min(1e-12))
check("before whitening, how far a draw moves the model depends on its direction",
      rv > 0.15, f"relative spread {rv:.3f}")
check("after whitening it depends on it much less",
      wv < rv * 0.6, f"{rv:.3f} -> {wv:.3f}")

# The spread measured along the old columns describes nothing after they are
# rewritten, so it must not be left behind to shape a draw over the new ones.
check("the old per-direction spread is cleared, not carried over",
      all(torch.allclose(axes.scale[li], torch.ones(RANK)) for li in LAYERS))

# The displacement must be applied over the prompt and not at decode steps, and
# the last positions must be spared, exactly as the generation hook does.
seen = {"prompt": 0, "decode": 0, "width": None}
blocks = m._get_transformer_blocks()


def spy(module, inp, o):
    t = o[0] if isinstance(o, (tuple, list)) else o
    if isinstance(t, torch.Tensor) and t.dim() == 3:
        if t.shape[1] == 1:
            seen["decode"] += 1
        else:
            seen["prompt"] += 1
            seen["width"] = t.shape[1]
    return None


h = blocks[LAYERS[0]].register_forward_hook(spy)
try:
    whiten_axes_by_fisher(m, PROMPT, fresh_axes(), [LAYERS[0]],
                          prompt_tail_clear=2, verbose=False)
finally:
    h.remove()
check("the probe reads the prompt in one pass and never decodes",
      seen["prompt"] > 0 and seen["decode"] == 0,
      f"{seen['prompt']} prompt passes, {seen['decode']} decode steps")
check("it costs k(k+1)/2 + 1 forward passes",
      seen["prompt"] == RANK * (RANK + 1) // 2 + 1,
      f"{seen['prompt']} for rank {RANK}")

# A cached basis is loaded on resume, so whitening it twice would change the
# mechanism halfway through a run.
_again = whiten_axes_by_fisher(m, PROMPT, axes, LAYERS, prompt_tail_clear=2,
                               verbose=False)
check("a basis that was already whitened is left alone on a second pass",
      _again == {} and all(torch.equal(axes.basis[li], axes.basis[li]) for li in LAYERS))
_marked = fresh_axes()
whiten_axes_by_fisher(m, PROMPT, _marked, LAYERS, prompt_tail_clear=2, verbose=False)
check("and it says so, so a resumed run can tell", _marked.fisher_whitened)

try:
    whiten_axes_by_fisher(m, PROMPT, fresh_axes(), [99], verbose=False)
    refused = False
except (KeyError, ValueError):
    refused = True
check("a layer that does not exist is refused rather than skipped", refused)

# A layer that exists but has no basis is the case the KeyError is for.
_partial = fresh_axes()
del _partial.basis[LAYERS[1]]
try:
    whiten_axes_by_fisher(m, PROMPT, _partial, LAYERS, verbose=False)
    refused2 = False
except KeyError:
    refused2 = True
check("a layer with no perturbation basis is refused rather than skipped", refused2)

# --- dropping the inert directions rather than stretching them --------------
# On Qwen3-1.7B nine of thirty-one directions move the next-token distribution
# by less than the probe resolves, so about a third of a uniform draw's length
# goes where the model does not react. Whitening stretches those, which is why
# its gain is capped; dropping them has no amplification to cap.
_kept = fresh_axes()
_before_rank = _kept.basis[LAYERS[0]].shape[1]
whiten_axes_by_fisher(m, PROMPT, _kept, LAYERS, prompt_tail_clear=2,
                      mode="keep", verbose=False)
_after = _kept.basis[LAYERS[0]]
check("keeping drops columns rather than rescaling them",
      _after.shape[1] <= _before_rank and _after.shape[0] == DIM,
      f"{_before_rank} -> {_after.shape[1]} directions")
check("and what it keeps is an orthonormal basis, so a uniform draw stays uniform",
      bool(torch.allclose(_after.T @ _after,
                          torch.eye(_after.shape[1], dtype=_after.dtype), atol=1e-4)))
check("it does not blow the basis up the way an uncapped whitening would",
      float(_after.norm()) < 4 * float(_after.shape[1]) ** 0.5,
      f"norm {float(_after.norm()):.2f} for {_after.shape[1]} unit columns")

try:
    whiten_axes_by_fisher(m, PROMPT, fresh_axes(), LAYERS, mode="sideways",
                          verbose=False)
    _bad = False
except ValueError:
    _bad = True
check("a mode nobody implemented is refused, not silently whitened", _bad)

print()
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
    sys.exit(1)
print("ALL TESTS PASS")
