#!/usr/bin/env python
"""The per-story displacement's size must vary between stories when asked.

Every run in this project gives every story a displacement of the same size,
which puts them all on a shell around the unperturbed state rather than filling
the ball inside it. Vendi measures spread, and a shell has less of it: on the
geometry alone, a hundred points at this basis's rank scored with the same
Vendi, drawing the radius uniformly over [0, 2r] scores 2.3 against 1.9 for a
fixed radius, with the mean displacement unchanged.

It is the one property of the displacement never varied. Where it points has
been swept, how it is drawn has been swept, where it is applied has been swept,
how far it goes has been swept in the mean -- never in the spread.

The failure to guard is the usual one: if the size never actually varies, an
arm that asks for spread is the old arm and the sweep reads as a null.

Offline, no GPU, no model.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from noiseegra.setup_experiment import _ortho_tag  # noqa: E402
from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402

LAYER = 0
DIM = 16
GAMMA = 0.15


def a_plan(spread: float) -> SteeringPlan:
    torch.manual_seed(0)
    return SteeringPlan.build(
        vectors={"present_tense": {LAYER: torch.randn(DIM)}}, layers=[LAYER],
        specs=[ConstraintSpec("present_tense", 1.0)], rms_scale=1.0,
        steer_budget=2.0, offset_gamma=GAMMA, offset_mode="orth",
        offset_basis={LAYER: torch.linalg.qr(torch.randn(DIM, 4))[0]},
        offset_prefill=True, offset_gamma_spread=spread,
    )


def sizes(plan: SteeringPlan, n: int = 300):
    out = []
    for i in range(n):
        torch.manual_seed(1000 + i)
        plan.resample_offset(i)
        out.append(float(plan.layer_plans[LAYER].offset.norm()))
    return out


def test_no_spread_gives_every_story_the_same_size() -> None:
    s = sizes(a_plan(0.0))
    assert statistics.pstdev(s) < 1e-5, f"sizes varied with spread off: {s[:3]}"
    print(f"  [PASS] with no spread every story is displaced the same, {s[0]:.3f}")


def test_spread_varies_the_size_and_keeps_the_mean() -> None:
    fixed = sizes(a_plan(0.0))
    varied = sizes(a_plan(1.0))
    assert statistics.pstdev(varied) > 0.3 * statistics.mean(fixed), (
        f"the size barely varied: sd {statistics.pstdev(varied):.4f}")
    assert abs(statistics.mean(varied) - statistics.mean(fixed)) < 0.15 * statistics.mean(fixed), (
        f"the mean size moved: {statistics.mean(varied):.3f} against "
        f"{statistics.mean(fixed):.3f}; the spread is meant to be about it, not "
        f"instead of it")
    assert min(varied) < 0.3 * statistics.mean(fixed), "no story was left near the centre"
    assert max(varied) > 1.7 * statistics.mean(fixed), "no story was sent near the edge"
    print(f"  [PASS] the size varies and the mean holds  "
          f"{min(varied):.3f} to {max(varied):.3f}, mean {statistics.mean(varied):.3f} "
          f"against {statistics.mean(fixed):.3f}")


def test_the_spread_is_in_the_run_id() -> None:
    class Spec:
        def __init__(self, plan):
            self.steering_plan = plan

    off = _ortho_tag("M", Spec(a_plan(0.0)))
    on = _ortho_tag("M", Spec(a_plan(0.5)))
    assert off != on, "two spreads produced the same run id"
    assert "__gs" in on, f"the spread is missing from {on}"
    print("  [PASS] the spread is in the run id")


if __name__ == "__main__":
    test_no_spread_gives_every_story_the_same_size()
    test_spread_varies_the_size_and_keeps_the_mean()
    test_the_spread_is_in_the_run_id()
    print("\nok")
