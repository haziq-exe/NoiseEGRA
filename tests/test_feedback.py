"""Feedback steering: correct the story's own shortfall, not everyone's.

Constant steering adds the same vector to every story. That is why it buys
compliance by spending diversity -- forty stories shoved the same way end up more
alike, measured at 4.42 -> 3.88 requirements broken and 3.61 -> 3.12 Vendi.
Feedback steering reads where this story already sits on each constraint axis and
pushes only the shortfall, so a compliant story is left untouched and two stories
failing different constraints are corrected differently.

CPU only, 64-wide.  python tests/test_feedback.py
"""

import math, sys, warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))

from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


DIM, LAYERS, RMS = 64, [2, 3], 2.0
NAMES = ["present_tense", "simple_register", "dialogue"]
torch.manual_seed(0)
VECS = {n: {l: torch.randn(DIM) for l in LAYERS} for n in NAMES}
POS = {n: {l: torch.randn(DIM) * 0.5 for l in LAYERS} for n in NAMES}
L = LAYERS[0]


def plan(**kw):
    kw.setdefault("noise_mode", "none")
    kw.setdefault("noise_alpha", 0.0)
    return SteeringPlan.build(VECS, LAYERS, [ConstraintSpec(n, beta=1.0) for n in NAMES],
                              rms_scale=RMS, **kw)


p = plan(steer_mode="feedback", targets=POS, feedback_cap=1.0)
p.resample_offset()
lp = p.layer_plans[L]
tau, B = lp.target, lp.basis

print("== the target is where compliant text sits ==")
check("one target per constraint", tau.shape == (len(NAMES),), str(tuple(tau.shape)))
check("feedback without targets is refused",
      _raises(lambda: plan(steer_mode="feedback")))
check("an unknown mode is refused", _raises(lambda: plan(steer_mode="proportional")))

print("\n== a story that already complies is left alone ==")
check("no correction at all", p.feedback_delta(L, B @ (tau + 1.0)) is None)
on_target = p.feedback_delta(L, B @ tau)
check("and none for a story exactly on target",
      on_target is None or float(on_target.norm()) < 1e-3,
      "none" if on_target is None else f"{float(on_target.norm()):.2e}")

print("\n== a story that is short is pushed onto the target ==")
short = B @ (tau - 1.0)
d = p.feedback_delta(L, short)
check("a correction is produced", d is not None)
landed = (short + d) @ B
check("it lands on the target", float((landed - tau).abs().max()) < 1e-4,
      f"{[round(float(x), 3) for x in landed]}")
check("and it moves nothing outside the constraint span",
      float((d - B @ (B.t() @ d)).norm()) < 1e-4)

print("\n== different stories get different corrections ==")
s1 = B @ (tau - torch.tensor([2.0, 0.0, 0.0]))
s2 = B @ (tau - torch.tensor([0.0, 0.0, 2.0]))
d1, d2 = p.feedback_delta(L, s1), p.feedback_delta(L, s2)
cos = float((d1 @ d2) / (d1.norm() * d2.norm()))
check("two stories failing different constraints are corrected differently",
      abs(cos) < 0.05, f"cosine {cos:.3f}")
# the contrast with constant steering, which is the whole point
c = plan()
c.resample_offset()
cd1, cd2 = c.delta_for(L, 0), c.delta_for(L, 0)
check("constant steering gives every story the identical push",
      float((cd1 @ cd2) / (cd1.norm() * cd2.norm())) > 0.999)

print("\n== only the failing constraints move ==")
one_short = B @ (tau - torch.tensor([0.0, 3.0, 0.0]))
d3 = p.feedback_delta(L, one_short)
moved = (d3 @ B).abs()
check("the satisfied constraints are untouched",
      float(moved[0]) < 1e-5 and float(moved[2]) < 1e-5,
      f"{[round(float(x), 4) for x in moved]}")
check("the failing one is corrected", float(moved[1]) > 2.9)

print("\n== the cap stops a correction breaking the text ==")
capped = plan(steer_mode="feedback", targets=POS, feedback_cap=0.05)
capped.resample_offset()
far = capped.layer_plans[L].basis @ (capped.layer_plans[L].target - 50.0)
dc = capped.feedback_delta(L, far)
limit = 0.05 * RMS * math.sqrt(DIM)
check("a wildly non-compliant story gets a bounded correction",
      float(dc.norm()) <= limit + 1e-4, f"{float(dc.norm()):.3f} <= {limit:.3f}")
check("an uncapped plan would have pushed far harder",
      float(p.feedback_delta(L, far).norm()) > float(dc.norm()) * 3)

print("\n== beta scales the correction ==")
half = SteeringPlan.build(VECS, LAYERS, [ConstraintSpec(n, beta=0.5) for n in NAMES],
                          rms_scale=RMS, noise_mode="none", noise_alpha=0.0,
                          steer_mode="feedback", targets=POS, feedback_cap=1.0)
half.resample_offset()
check("beta 0.5 closes half the gap",
      abs(float(half.feedback_delta(L, short).norm()) - float(d.norm()) / 2) < 1e-3)

print("\n== end to end on a 64-wide model ==")
from tiny_model import Tiny  # noqa: E402

egra = Tiny()
egra.tokenizer.decode = lambda ids, skip_special_tokens=True: " ".join(str(int(i)) for i in ids)
PROMPT = [{"role": "system", "content": "s"}, {"role": "user", "content": "write a story"}]
vecs = {n: {l: torch.randn(DIM) for l in LAYERS} for n in NAMES}
pos = {n: {l: torch.randn(DIM) * 3.0 for l in LAYERS} for n in NAMES}


def tiny(**kw):
    kw.setdefault("noise_mode", "none")
    kw.setdefault("noise_alpha", 0.0)
    return SteeringPlan.build(vecs, LAYERS, [ConstraintSpec(n, beta=1.0) for n in NAMES],
                              rms_scale=1.0, **kw)


base = egra.generate_with_orthogonal_steering(PROMPT, tiny(), max_new_tokens=12, seed=7)
fb = egra.generate_with_orthogonal_steering(
    PROMPT, tiny(steer_mode="feedback", targets=pos, feedback_cap=0.5),
    max_new_tokens=12, seed=7)
check("feedback steering changes what is written", fb != base, f"{base!r} vs {fb!r}")
outs = {egra.generate_with_orthogonal_steering(
    PROMPT, tiny(steer_mode="feedback", targets=pos, feedback_cap=0.5),
    max_new_tokens=12, seed=s) for s in range(6)}
check("and different seeds still give different stories", len(outs) > 1,
      f"{len(outs)} of 6 distinct")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("all feedback-steering tests passed")
