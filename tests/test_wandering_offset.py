#!/usr/bin/env python
"""The displacement wanders during a story instead of standing still.

Two mechanisms existed here and they were separate things: one fixed
displacement held for a whole story, and fresh noise at every decode step. The
first moves stories apart and drags the whole story off register; the second is
a measured null for variety three times over, because it varies inside a story
and every story varies the same way.

They are the two ends of one axis. With a noise colour set, the displacement's
DIRECTION follows a 1/f^beta trajectory along the token axis while its LENGTH is
held at what the draw gave it. beta=0 redraws the direction every step; a large
exponent leaves it almost fixed. Holding the length constant is what keeps
`gamma` meaning the same number of story-distances at every exponent -- without
it, an exponent sweep would be a size sweep as well, which is a confound this
project has shipped before.

Offline, no GPU, no model.

    python tests/test_wandering_offset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAIL.append(name)


LAYER, DIM, RANK, STEPS = 0, 24, 6, 256


def a_plan(**kw):
    torch.manual_seed(0)
    basis = {LAYER: torch.linalg.qr(torch.randn(DIM, RANK))[0]}
    plan = SteeringPlan.build(
        vectors={"present_tense": {LAYER: torch.randn(DIM)}}, layers=[LAYER],
        specs=[ConstraintSpec("present_tense", 1.0)], rms_scale=1.0,
        steer_budget=2.0, offset_gamma=1.5, offset_mode="orth",
        offset_basis=basis, offset_prefill=True, offset_decode=True,
        offset_norm="energy", **kw)
    plan.noise_traj_steps = STEPS
    return plan


def walk(plan, story=0, steps=STEPS):
    torch.manual_seed(100 + story)
    plan.resample_offset(story_index=story)
    lp = plan.layer_plans[LAYER]
    # Exactly what `delta_for` does at each decode step, envelope included.
    return torch.stack([lp.offset_at(t, plan.envelope_at(t)) for t in range(steps)])


# --- the default is exactly the old mechanism -------------------------------
flat = walk(a_plan())
check("with no colour asked for, the displacement never moves",
      float((flat - flat[0]).abs().max()) == 0.0)
check("the displacement has the full dimension", flat.shape[1] == DIM)

_p = a_plan()
torch.manual_seed(100)
_p.resample_offset(story_index=0)
check("with no colour the step-0 value is the stored per-story displacement",
      torch.equal(_p.layer_plans[LAYER].offset_at(0), _p.layer_plans[LAYER].offset))
check("and no trajectory was built at all",
      _p.layer_plans[LAYER].offset_traj is None)


# --- length is held constant whatever the colour ----------------------------
def lengths(beta):
    w = walk(a_plan(noise_beta=beta))
    return w.norm(dim=1)


for b in (0.0, 1.0, 3.0):
    L = lengths(b)
    check(f"at exponent {b:g} the displacement keeps one length all through",
          float((L.max() - L.min()) / L.mean()) < 1e-4,
          f"{float(L.mean()):.4f} +/- {float(L.std()):.2e}")

ref = float(walk(a_plan()).norm(dim=1).mean())
check("and it is the same length as with no colour at all",
      all(abs(float(lengths(b).mean()) - ref) / ref < 1e-4 for b in (0.0, 1.0, 3.0)),
      f"no colour {ref:.4f}")


# --- but the direction wanders, and the colour says how fast ----------------
def wander(beta):
    """Mean angle between the displacement now and one step later."""
    w = walk(a_plan(noise_beta=beta, noise_fmin_cycles=0.25))
    u = w / w.norm(dim=1, keepdim=True)
    return float((1.0 - (u[:-1] * u[1:]).sum(dim=1)).mean())


w0, w1, w2, w4 = wander(0.0), wander(1.0), wander(2.0), wander(4.0)
check("a zero exponent redraws the direction every step",
      w0 > 0.5, f"step-to-step change {w0:.3f}")
check("a larger exponent makes it drift more slowly",
      w0 > w1 > w2 > w4, f"{w0:.3f} > {w1:.3f} > {w2:.3f} > {w4:.3f}")
check("a large exponent is nearly the fixed displacement again",
      w4 < 0.01, f"step-to-step change {w4:.5f}")


# --- and the whole point: stories must still differ from each other ---------
def between_story_spread(beta, stories=32):
    """How much the story-average displacement differs between stories."""
    plan_kw = dict(noise_beta=beta, noise_fmin_cycles=0.25) if beta is not None else {}
    means = []
    for i in range(stories):
        w = walk(a_plan(**plan_kw), story=i)
        m = w.mean(dim=0)
        means.append(m / max(float(w.norm(dim=1).mean()), 1e-12))
    M = torch.stack(means)
    return float(M.var(dim=0, unbiased=False).sum())


s_white = between_story_spread(0.0)
s_mid = between_story_spread(1.5)
s_fixed = between_story_spread(None)
check("at a zero exponent the stories' average displacements collapse together",
      s_white < 0.02, f"{s_white:.4f}")
check("a fixed displacement keeps them apart",
      s_fixed > 0.5, f"{s_fixed:.4f}")
check("an intermediate exponent keeps much of that while still wandering",
      s_white < s_mid < s_fixed and s_mid > 0.1,
      f"white {s_white:.3f} < mid {s_mid:.3f} < fixed {s_fixed:.3f}")


# --- the envelope -----------------------------------------------------------
rise = a_plan(noise_beta=1.0)
rise.offset_envelope = "rise"
r = walk(rise).norm(dim=1)
check("a rising envelope starts near nothing and grows",
      float(r[0]) < 1e-6 and float(r[-1]) > float(r[STEPS // 2]) > float(r[1]),
      f"{float(r[0]):.4f} -> {float(r[STEPS//2]):.4f} -> {float(r[-1]):.4f}")

dec = a_plan(noise_beta=1.0)
dec.offset_envelope = "decay"
d = walk(dec).norm(dim=1)
check("a decaying envelope does the opposite",
      float(d[0]) > float(d[STEPS // 2]) > float(d[-1]),
      f"{float(d[0]):.4f} -> {float(d[STEPS//2]):.4f} -> {float(d[-1]):.4f}")

bad = a_plan(noise_beta=1.0)
bad.offset_envelope = "sideways"
try:
    bad.envelope_at(3)
    refused = False
except ValueError:
    refused = True
check("an envelope nobody implemented is refused, not ignored", refused)


# --- a trajectory shorter than the story must not run off the end -----------
short = a_plan(noise_beta=1.0)
short.noise_traj_steps = 16
torch.manual_seed(1)
short.resample_offset(story_index=0)
lp = short.layer_plans[LAYER]
check("asking past the end of the trajectory holds the last step",
      torch.equal(lp.offset_at(15), lp.offset_at(999)))

print()
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
    sys.exit(1)
print("ALL TESTS PASS")
