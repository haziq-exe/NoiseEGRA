#!/usr/bin/env python
"""The pulled-back Fisher-Rao metric, against a model whose answer is known.

Round 16 laid the whole set of per-story perturbations out by repulsion so it
covered the perturbation subspace evenly, and all four matched pairs came back
worse. The reason recorded at the time was that distance in the subspace does
not predict distance between the stories. This module answers that by measuring
distance in the model's own output distribution instead.

The checks use a synthetic categorical model, logits = A @ delta, because its
Fisher information is known in closed form: I = diag(p) - p p'. So the estimate
can be compared against the true pullback A' I A rather than against itself.

Offline, no GPU, no language model.

    python tests/test_fisher.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from noiseegra.fisher import (  # noqa: E402
    anisotropy, fisher_rao_distance, resolved_directions, subspace_metric,
    whiten_basis,
)

FAIL = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAIL.append(name)


# --- the closed form itself -------------------------------------------------
p = torch.tensor([0.5, 0.3, 0.2])
check("a distribution is no distance from itself",
      float(fisher_rao_distance(p, p)) < 1e-9)
q = torch.tensor([0.2, 0.3, 0.5])
check("the distance is symmetric",
      abs(float(fisher_rao_distance(p, q)) - float(fisher_rao_distance(q, p))) < 1e-12)
a = torch.tensor([1.0, 0.0, 0.0])
b = torch.tensor([0.0, 1.0, 0.0])
check("two distributions with no shared support are pi apart",
      abs(float(fisher_rao_distance(a, b)) - 3.14159265) < 1e-6,
      f"{float(fisher_rao_distance(a, b)):.6f}")
mid = torch.tensor([0.5, 0.5, 0.0])
check("a distribution between two others is nearer to each than they are to each other",
      float(fisher_rao_distance(a, mid)) < float(fisher_rao_distance(a, b)))

# --- a synthetic categorical model with a known pullback --------------------
torch.manual_seed(0)
DIM, VOCAB, K = 32, 40, 5
A = torch.randn(VOCAB, DIM, dtype=torch.float64) * 0.4
basis = torch.linalg.qr(torch.randn(DIM, K, dtype=torch.float64))[0]
# Deliberately unequal: the model is far more sensitive along some directions
# than others, which is the situation whitening exists for.
gains = torch.tensor([3.0, 1.6, 1.0, 0.45, 0.18], dtype=torch.float64)
basis = basis * gains


def predict(delta):
    z = torch.zeros(DIM, dtype=torch.float64) if delta is None else delta.double()
    return torch.softmax(A @ z, dim=-1)


def true_metric(B):
    pr = predict(None)
    info = torch.diag(pr) - torch.outer(pr, pr)
    J = A @ B
    return J.T @ info @ J


# Fisher-Rao distance squared is 4x the symmetrised second-order term, so the
# finite-difference estimate should match the analytic pullback up to that one
# constant. Checking the ratio is constant across entries is the real test: a
# constant factor cancels in whitening, a direction-dependent one does not.
est = subspace_metric(predict, basis, step=1e-4)
tru = true_metric(basis)
ratio = (est / tru)[tru.abs() > 1e-6]
check("the estimated metric matches the analytic pullback up to one constant",
      float(ratio.std() / ratio.mean().abs()) < 1e-3,
      f"ratio {float(ratio.mean()):.4f} +/- {float(ratio.std()):.2e}")
check("and the constant is one: the squared Fisher-Rao distance is the pullback",
      abs(float(ratio.mean()) - 1.0) < 1e-3, f"{float(ratio.mean()):.6f}")

# The remaining error is the finite difference, not a mistake: it falls in
# proportion to the step until double precision runs out underneath it.
errs = []
for st in (1e-2, 1e-3, 1e-4):
    r = (subspace_metric(predict, basis, step=st) / tru)[tru.abs() > 1e-6]
    errs.append(float(r.std() / r.mean().abs()))
check("the error shrinks with the probe step, as a finite difference should",
      errs[0] > errs[1] > errs[2] and errs[0] / errs[1] > 5,
      " -> ".join(f"{e:.1e}" for e in errs))

check("the metric is symmetric", float((est - est.T).abs().max()) < 1e-12)
check("the metric is positive definite",
      float(torch.linalg.eigvalsh(est).min()) > 0)

# --- the subspace really is anisotropic, and whitening fixes it -------------
before = anisotropy(est)
check("the raw subspace is strongly anisotropic", before > 5.0, f"{before:.1f}x")

# With the cap out of the way, whitening is exact.
W = whiten_basis(basis, est, max_gain=1e6)
after = anisotropy(subspace_metric(predict, W, step=1e-4))
check("whitening makes every direction move the predictions equally",
      after < 1.05, f"{before:.1f}x -> {after:.3f}x")

# The cap is there because a direction the model barely responds to would
# otherwise be stretched without limit. It trades some of the correction for
# that guarantee, and the amount it gives up is exactly the cap.
CAP = 4.0
Wc = whiten_basis(basis, est, max_gain=CAP)
capped = anisotropy(subspace_metric(predict, Wc, step=1e-4))
check("a capped whitening corrects only as far as the cap allows",
      before / CAP * 0.8 < capped < before / CAP * 1.25,
      f"{before:.1f}x -> {capped:.2f}x at a cap of {CAP:g}")
check("and it still corrects most of the way",
      capped < before / 2, f"{capped:.2f}x vs {before:.1f}x")

# A subspace with a direction the probe cannot resolve at all must not blow up.
deg = est.clone()
evals, evecs = torch.linalg.eigh(deg)
evals[0] = 0.0
deg = evecs @ torch.diag(evals) @ evecs.T
Wd = whiten_basis(basis, deg, max_gain=8.0)
check("a direction with no measurable response is not stretched without limit",
      bool(torch.isfinite(Wd).all()) and float(Wd.norm()) < 40 * float(basis.norm()),
      f"norm grew {float(Wd.norm()) / float(basis.norm()):.1f}x")
check("and the resolvable directions are counted",
      resolved_directions(deg) == K - 1 and resolved_directions(est) == K,
      f"{resolved_directions(deg)} of {K} resolvable when one is zeroed")


def spread(B, n=400, scale=1.0):
    """Fisher-Rao distances moved by unit-length draws from the span of B."""
    g = torch.Generator().manual_seed(3)
    c = torch.randn(n, B.shape[1], generator=g, dtype=torch.float64)
    c = c / c.norm(dim=1, keepdim=True)
    base = predict(None)
    d = torch.tensor([float(fisher_rao_distance(base, predict(B @ ci * scale)))
                      for ci in c])
    return d


raw = spread(basis, scale=0.05)
whit = spread(W, scale=0.05)
check("before whitening, how far a draw moves the model depends on its direction",
      float(raw.std() / raw.mean()) > 0.3, f"spread {float(raw.std()/raw.mean()):.3f}")
check("after whitening, every draw moves the model about equally",
      float(whit.std() / whit.mean()) < 0.05,
      f"spread {float(whit.std()/whit.mean()):.3f}")

# --- refusals ---------------------------------------------------------------
def refuses(fn):
    try:
        fn()
        return False
    except ValueError:
        return True


check("a one-dimensional basis argument is refused",
      refuses(lambda: subspace_metric(predict, basis[:, 0])))
check("a non-positive probe step is refused",
      refuses(lambda: subspace_metric(predict, basis, step=0.0)))
check("a metric of the wrong size is refused",
      refuses(lambda: whiten_basis(basis, torch.eye(K + 1, dtype=torch.float64))))
check("a cap below one is refused: it would mean shrinking the stiffest direction",
      refuses(lambda: whiten_basis(basis, est, max_gain=0.5)))

# --- the cost claim ---------------------------------------------------------
calls = {"n": 0}


def counted(delta):
    calls["n"] += 1
    return predict(delta)


subspace_metric(counted, basis, step=1e-4)
check("estimating the metric costs k(k+1)/2 + 1 forward passes",
      calls["n"] == K * (K + 1) // 2 + 1, f"{calls['n']} calls for k={K}")

print()
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
    sys.exit(1)
print("ALL TESTS PASS")
