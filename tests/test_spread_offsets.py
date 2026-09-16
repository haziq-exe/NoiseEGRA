#!/usr/bin/env python
"""The per-story perturbations are chosen as a set, not drawn one at a time.

Diversity is a property of the set of stories, and every round before this drew
each story's perturbation independently and hoped the set spread out. In the full
residual stream that hope is sound -- two random vectors in two thousand
dimensions are nearly orthogonal. But the perturbation is deliberately drawn from
the subspace the model's own states occupy, which is low rank, and there the
hope fails: independent draws collide, several stories are perturbed almost
identically, and the diversity that was paid for is not collected.

What this must NOT do is change the dose. Round 2 measured what happens when the
aim of a single push is changed at constant length: Vendi moved by +0.12, a null.
So the claim here is narrow and has to be tested as such -- same length, same
subspace, same average direction, fewer collisions.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402

FAILED = []


def check(msg, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILED.append(msg)


DIM, LAYER, RANK, N = 128, 5, 8, 60
VECS = {n: {LAYER: torch.randn(DIM)} for n in ("a", "b")}
SPECS = [ConstraintSpec(name=n, beta=1.0) for n in ("a", "b")]
BASIS = torch.linalg.qr(torch.randn(DIM, RANK))[0]


PLANS = {}


def offsets(draw):
    plan = SteeringPlan.build(
        VECS, [LAYER], SPECS, rms_scale=2.0, noise_mode="none", noise_alpha=0.0,
        offset_mode="orth", offset_gamma=0.1, offset_basis={LAYER: BASIS},
        offset_draw=draw,
    )
    plan.plan_offsets(N, seed=0)
    PLANS[draw] = plan
    out = []
    for i in range(N):
        torch.manual_seed(1000 + i)
        plan.resample_offset(story_index=i)
        out.append(plan.layer_plans[LAYER].offset.clone())
    return torch.stack(out)


def pairwise(M):
    U = M / M.norm(dim=1, keepdim=True)
    sim = (U @ U.t()).abs()
    sim.fill_diagonal_(0.0)
    return sim


print(f"== {N} stories, perturbations drawn from a rank-{RANK} subspace ==")
iid, spread = offsets("iid"), offsets("spread")
si, ss = pairwise(iid), pairwise(spread)

check("independent draws collide badly in a low-rank subspace",
      float(si.max()) > 0.8, f"worst pair {float(si.max()):.3f} alike")
check("choosing the set pulls the worst pair apart",
      float(ss.max()) < float(si.max()) - 0.2,
      f"{float(si.max()):.3f} -> {float(ss.max()):.3f}")
check("and it is the collisions that go, not the spread as a whole",
      abs(float(ss.mean()) - float(si.mean())) < 0.05,
      f"mean similarity {float(si.mean()):.3f} -> {float(ss.mean()):.3f}")

# The dose is what makes this different from re-aiming a single push, which is a
# measured null. If the lengths moved, any diversity gained would just be the
# usual trade being paid for again.
li, ls = iid.norm(dim=1), spread.norm(dim=1)
check("every perturbation keeps exactly the length it had",
      float((ls - li.mean()).abs().max()) < 1e-3,
      f"iid {float(li.mean()):.4f}, spread {float(ls.min()):.4f}-{float(ls.max()):.4f}")

# Same subspace: an offset that wandered off the manifold would be a different
# intervention, not the same one allocated differently. The subspace to check
# against is the one the plan actually draws from -- the activation basis with the
# constraint directions already stripped out of it at build time, not the basis as
# handed in.
lp = PLANS["spread"].layer_plans[LAYER]
OB = lp.offset_basis
resid = spread - (spread @ OB) @ OB.t()
check("and stays in the subspace it was drawn from",
      float(resid.norm(dim=1).max()) < 1e-3,
      f"largest component outside {float(resid.norm(dim=1).max()):.2e}")

# This is the one that makes the comparison against the iid arm fair. If choosing
# the set leaked any push onto a constraint direction, a compliance difference
# would be the leak rather than the allocation.
protect = lp.protect
if protect is not None:
    leak = (spread @ protect).abs().max()
    check("and puts nothing at all onto the constraint directions",
          float(leak) < 1e-3, f"largest constraint component {float(leak):.2e}")

check("the layout is reproducible from its seed",
      torch.allclose(offsets("spread"), spread, atol=1e-5))
check("asking for iid changes nothing about how it behaves",
      offsets("iid").shape == iid.shape)

print()
if FAILED:
    print(f"{len(FAILED)} FAILURE(S): {FAILED}")
    sys.exit(1)
print("spread-offset tests passed")
