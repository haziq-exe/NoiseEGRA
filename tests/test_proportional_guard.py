#!/usr/bin/env python
"""A story displaced further must get more of the push on the guarded direction.

The formatting ceiling binds per story, not on the average. Varying the size of
the per-story displacement took headings from 2% of stories to 9% while the mean
displacement stayed exactly where it was, so the stories drawn near the top of
the range are crossing the threshold individually while the ones near the bottom
are nowhere near it. A push that is identical for both spends protection where
it is not needed and withholds it where it is.

This scales the guarded direction's share by how far the story was displaced.
The total is renormalised to the same budget afterwards, so a far-displaced
story trades some of its other steering for formatting and a near one keeps it.

If the coupling does not reach the coefficients, every story is pushed the same
way and an arm that asks for it is the old arm -- a null that looks like a
result, which this project has hit five times.

Offline, no GPU, no model.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402

LAYER = 0
DIM = 16
NAMES = ["present_tense", "story_format"]


def a_plan(guard: str, spread: float = 0.5) -> SteeringPlan:
    torch.manual_seed(0)
    return SteeringPlan.build(
        vectors={n: {LAYER: torch.randn(DIM)} for n in NAMES}, layers=[LAYER],
        specs=[ConstraintSpec(n, 1.0) for n in NAMES], rms_scale=1.0,
        steer_budget=2.0, offset_gamma=0.15, offset_mode="orth",
        offset_basis={LAYER: torch.linalg.qr(torch.randn(DIM, 4))[0]},
        offset_prefill=True, offset_gamma_spread=spread, guard_direction=guard,
    )


def pairs(plan: SteeringPlan, n: int = 60):
    """(how far this story was displaced, what share the guarded direction got)."""
    out = []
    for i in range(n):
        torch.manual_seed(50 + i)
        plan.resample_offset(i)
        share = 1.0 if plan.gains is None else plan.gains[NAMES.index("story_format")]
        out.append((float(plan.layer_plans[LAYER].offset.norm()), share))
    return out


def test_the_guard_follows_the_displacement() -> None:
    got = pairs(a_plan("story_format"))
    got.sort()
    first = sum(s for _, s in got[:15]) / 15
    last = sum(s for _, s in got[-15:]) / 15
    assert last > first * 1.5, (
        f"the guard did not follow the displacement: the least-displaced stories "
        f"got {first:.2f} and the most-displaced {last:.2f}")
    # and it should be monotone in the displacement, not merely different
    shares = [s for _, s in got]
    assert shares == sorted(shares), "the guard is not monotone in the displacement"
    print(f"  [PASS] the guard follows the displacement  {first:.2f} for the nearest, "
          f"{last:.2f} for the furthest")


def test_no_guard_leaves_every_story_alike() -> None:
    plan = a_plan("")
    for _, share in pairs(plan, 20):
        assert share == 1.0, f"a guard was applied when none was asked for: {share}"
    print("  [PASS] with no guard named every story is pushed alike, as before")


def test_an_unknown_name_is_ignored_rather_than_silently_wrong() -> None:
    """Naming a direction that is not steered must not quietly guard nothing else."""
    for _, share in pairs(a_plan("not_a_direction"), 20):
        assert share == 1.0, f"an unknown guard changed the push: {share}"
    print("  [PASS] naming a direction that is not steered changes nothing")


if __name__ == "__main__":
    test_the_guard_follows_the_displacement()
    test_no_guard_leaves_every_story_alike()
    test_an_unknown_name_is_ignored_rather_than_silently_wrong()
    print("\nok")
