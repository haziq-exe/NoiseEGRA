#!/usr/bin/env python
"""f(S_c)'s aim must wander over decode steps without changing how hard it pushes.

The one axis the method does not win is variety of what happens: raised
temperature with top-k sampling varies at every token and leads it by 5.2 on an
interval that clears zero, and nothing that changes where the perturbation is
applied, how large it is, which layers it acts on, which steps it is allowed
through at, or how its directions are drawn has closed that. Those are all
once-per-story mechanisms, and a once-per-story mechanism cannot buy token-level
variety.

The two things that do vary per token are both unusable as they stand. Raising
the sampling temperature is ruled out by the task. Adding fresh noise at every
step is the published method's own mechanism and on this model it broke every
opening story at 0.4 and matched the baseline's variety exactly at 0.2.

This is the object in between: the aim of the constraint push performs a
correlated random walk, so adjacent tokens are pushed in almost the same
direction and nothing breaks locally, while over a story the aim explores. In
rotate mode the length of the push is preserved exactly, so the model is pushed
as hard as ever and only where it is aimed moves -- a function of the constraint
vector rather than a term added beside it.

Offline, no GPU, no model.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from noiseegra.setup_experiment import _ortho_tag  # noqa: E402
from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402

LAYER = 0
DIM = 32


def a_plan(walk: float, mode: str = "rotate", kappa: float = 0.3) -> SteeringPlan:
    torch.manual_seed(0)
    vectors = {n: {LAYER: torch.randn(DIM)} for n in ("present_tense", "sensory")}
    plan = SteeringPlan.build(
        vectors=vectors, layers=[LAYER],
        specs=[ConstraintSpec(n, 1.0) for n in ("present_tense", "sensory")],
        rms_scale=1.0, steer_budget=2.0,
        jitter_mode=mode, jitter_kappa=kappa, jitter_walk=walk,
    )
    plan.resample_offset(0)
    return plan


def aims(plan: SteeringPlan, steps: int):
    """The unit aim of the push at each decode step."""
    out = []
    for t in range(steps):
        d = plan.delta_for(LAYER, t, with_noise=False, with_offset=False)
        out.append(d / d.norm())
    return torch.stack(out)


def test_a_still_aim_does_not_move() -> None:
    """Walk 0 is what every run before this did: one f per story."""
    a = aims(a_plan(0.0), 12)
    assert torch.allclose(a[0], a[-1], atol=1e-6), (
        "the aim moved with the walk switched off")
    print("  [PASS] with no walk the aim is fixed for the whole story, as before")


def test_the_aim_wanders_and_neighbouring_steps_stay_close() -> None:
    """The point of a walk rather than a redraw: local smoothness, global drift."""
    a = aims(a_plan(0.15), 60)
    adjacent = float((a[:-1] * a[1:]).sum(-1).mean())
    far = float((a[0] * a[-1]).sum())
    assert adjacent > 0.9, f"adjacent steps are not close: cosine {adjacent:.3f}"
    assert far < adjacent - 0.02, (
        f"the aim did not drift: start-to-end cosine {far:.3f} against "
        f"adjacent {adjacent:.3f}")
    print(f"  [PASS] the aim wanders  adjacent cosine {adjacent:.3f}, "
          f"start to end {far:.3f}")


def test_a_full_walk_is_white_noise_on_the_aim() -> None:
    a = aims(a_plan(1.0), 40)
    adjacent = float((a[:-1] * a[1:]).sum(-1).mean())
    still = float((aims(a_plan(0.15), 40)[:-1] * aims(a_plan(0.15), 40)[1:]).sum(-1).mean())
    assert adjacent < still, (
        f"a full walk should decorrelate faster than a partial one: "
        f"{adjacent:.3f} against {still:.3f}")
    print(f"  [PASS] a full walk decorrelates adjacent steps  {adjacent:.3f}")


def test_rotating_the_aim_leaves_the_push_as_hard() -> None:
    """The whole reason to rotate rather than add: the dose does not change."""
    plan = a_plan(0.2, mode="rotate")
    still = a_plan(0.0, mode="rotate", kappa=0.0)
    want = float(still.delta_for(LAYER, 0, with_noise=False, with_offset=False).norm())
    for t in range(20):
        got = float(plan.delta_for(LAYER, t, with_noise=False,
                                   with_offset=False).norm())
        assert math.isclose(got, want, rel_tol=1e-4), (
            f"step {t}: the push changed length, {got:.4f} against {want:.4f}")
    print("  [PASS] rotating the aim leaves the push exactly as hard at every step")


def test_the_walk_restarts_for_each_story() -> None:
    plan = a_plan(0.3)
    first = aims(plan, 30)[-1].clone()
    plan.resample_offset(1)
    second = aims(plan, 1)[0]
    assert not torch.allclose(first, second, atol=1e-4), (
        "the next story inherited the previous one's wandered aim")
    print("  [PASS] each story starts its own walk")


def test_the_walk_is_in_the_run_id() -> None:
    class Spec:
        def __init__(self, plan):
            self.steering_plan = plan

    still = _ortho_tag("M", Spec(a_plan(0.0)))
    moving = _ortho_tag("M", Spec(a_plan(0.15)))
    assert still != moving, "two walk rates produced the same run id"
    assert "__jw" in moving, f"the walk rate is missing from {moving}"
    print("  [PASS] the walk rate is in the run id")


if __name__ == "__main__":
    test_a_still_aim_does_not_move()
    test_the_aim_wanders_and_neighbouring_steps_stay_close()
    test_a_full_walk_is_white_noise_on_the_aim()
    test_rotating_the_aim_leaves_the_push_as_hard()
    test_the_walk_restarts_for_each_story()
    test_the_walk_is_in_the_run_id()
    print("\nok")
