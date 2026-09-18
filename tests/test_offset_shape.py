#!/usr/bin/env python
"""A manifold-shaped perturbation must weight directions by how far stories spread.

Variety of what happens rises with the size of the per-story perturbation up to
0.15 and falls after it -- 67.2, then 64.3 at 0.2, then 61.8 at 0.25 -- while
variety of wording keeps climbing, 17.9 to 24.1 to 26.5. Past the peak the
displacement stops sending the model to a different story and starts disturbing
the wording of the one it was going to write anyway.

A uniform draw is a candidate for why. The basis is ordered by how much
between-story variation each direction carries, and the last directions carry
almost none, so weighting them equally with the first sends a step of a given
length far outside anything the model does.

The failure to guard against is the usual one here: if the shaping never reaches
the draw, the arm is the old one and a sweep over it reads as a null. Three
suites once ran isotropic noise under run ids claiming an estimated basis.

Offline, no GPU, no model.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from noiseegra.setup_experiment import _ortho_tag  # noqa: E402
from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402

LAYER = 0
DIM = 16
RANK = 6
# Stories spread a lot along the first direction and almost none along the last.
SPREAD = torch.tensor([4.0, 2.0, 1.0, 0.5, 0.2, 0.05])


def a_plan(shape: str) -> SteeringPlan:
    torch.manual_seed(0)
    vectors = {"present_tense": {LAYER: torch.randn(DIM)}}
    basis = {LAYER: torch.linalg.qr(torch.randn(DIM, RANK))[0]}
    plan = SteeringPlan.build(
        vectors=vectors, layers=[LAYER],
        specs=[ConstraintSpec("present_tense", 1.0)], rms_scale=1.0,
        steer_budget=2.0, offset_gamma=0.1, offset_mode="orth",
        offset_basis=basis, offset_scale={LAYER: SPREAD},
        offset_draw_shape=shape, offset_prefill=True,
    )
    plan.plan_offsets(64, seed=0)
    return plan


def weight_per_direction(plan: SteeringPlan) -> torch.Tensor:
    """Root-mean-square weight on each basis direction, over many stories.

    Read off the offsets the plan actually produces, not from the laid-out
    coefficient table: that table is only built for the spread draw, and the
    draw every run uses is the independent one. Testing the table would have
    passed while the path in use went unshaped.
    """
    lp = plan.layer_plans[LAYER]
    rows = []
    for i in range(300):
        torch.manual_seed(1000 + i)
        plan.resample_offset(i)
        rows.append(lp.offset_basis.t() @ lp.offset)
    return torch.stack(rows).pow(2).mean(dim=0).sqrt()


def test_the_shaping_follows_the_spread() -> None:
    shaped = weight_per_direction(a_plan("manifold"))
    assert shaped[0] > shaped[-1] * 5, (
        f"the draw did not follow the spread: first direction {shaped[0]:.3f}, "
        f"last {shaped[-1]:.3f}, for spreads {SPREAD[0]:.2f} and {SPREAD[-1]:.2f}"
    )
    # Ordering should match the spread's ordering.
    assert torch.all(shaped[:-1] >= shaped[1:] - 1e-6), f"not monotone: {shaped}"
    print(f"  [PASS] the shaped draw follows the spread  "
          f"{shaped[0]:.3f} -> {shaped[-1]:.3f}")


def test_the_sphere_draw_is_still_even() -> None:
    even = weight_per_direction(a_plan("sphere"))
    assert even.max() / even.min() < 2.0, (
        f"the unshaped draw is not even across directions: {even}")
    print(f"  [PASS] the default draw stays even  "
          f"{even.max() / even.min():.2f}x between most and least")


def test_the_two_draws_differ() -> None:
    """The point of the option is that it changes the perturbation itself."""
    def first_offset(shape: str) -> torch.Tensor:
        plan = a_plan(shape)
        torch.manual_seed(7)
        plan.resample_offset(0)
        return plan.layer_plans[LAYER].offset.clone()

    assert not torch.allclose(first_offset("sphere"), first_offset("manifold"),
                              atol=1e-6), (
        "shaping the draw produced the same perturbation as not shaping it")
    print("  [PASS] shaping changes the perturbation")


def test_the_shape_is_in_the_run_id() -> None:
    class Spec:
        def __init__(self, plan):
            self.steering_plan = plan

    even = _ortho_tag("M", Spec(a_plan("sphere")))
    shaped = _ortho_tag("M", Spec(a_plan("manifold")))
    assert even != shaped, "two draw shapes produced the same run id"
    assert "__odmanifold" in shaped, f"the shape is missing from {shaped}"
    print("  [PASS] the draw shape is in the run id")


if __name__ == "__main__":
    test_the_shaping_follows_the_spread()
    test_the_sphere_draw_is_still_even()
    test_the_two_draws_differ()
    test_the_shape_is_in_the_run_id()
    print("\nok")
