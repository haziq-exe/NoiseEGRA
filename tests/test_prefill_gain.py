#!/usr/bin/env python
"""The prompt-side push strength must actually change what reaches the prompt.

Rounds 26 and 27 measured that the two places a constraint push can be applied
do different jobs: applied while writing it enforces a sustained property and
shortens the prose, applied to the prompt positions it does neither but raises
variety of wording from 9.7 to 16.7. Every run before this tied the two
strengths together, so they could not be dosed apart.

The failure this guards against is silent. If the multiplier never reaches the
vector the prompt is given, every arm of a sweep over it generates identical
stories, and the sweep reads as "the prompt strength does not matter" -- which
is exactly the wrong conclusion, and indistinguishable from a real null. A
sibling fault has already happened here once: three suites never built the
activation basis they named, so their arms silently ran isotropic noise.

Offline, no GPU, no model.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from noiseegra.setup_experiment import _ortho_tag  # noqa: E402
from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402

LAYERS = [2, 3]
DIM = 16
NAMES = ["present_tense", "sensory"]


def a_plan(prefill_gain: float) -> SteeringPlan:
    torch.manual_seed(0)
    vectors = {n: {l: torch.randn(DIM) for l in LAYERS} for n in NAMES}
    return SteeringPlan.build(
        vectors=vectors, layers=LAYERS, specs=[ConstraintSpec(n, 1.0) for n in NAMES],
        rms_scale=1.0, steer_budget=2.0, steer_prefill=True,
        prefill_gain=prefill_gain,
    )


def test_the_multiplier_scales_the_vector_the_prompt_is_given() -> None:
    one = a_plan(1.0)
    four = a_plan(4.0)
    for layer in LAYERS:
        base = one.steering_only(layer, 0)
        scaled = four.steering_only(layer, 0)
        assert base is not None and scaled is not None, "no vector at the prompt"
        assert torch.allclose(scaled, base * 4.0, atol=1e-5), (
            f"layer {layer}: the prompt vector did not scale with the multiplier"
        )
        assert scaled.norm() > base.norm() * 3.5, "the prompt push did not grow"
    print("  [PASS] the prompt-side multiplier scales the vector the prompt is given")


def test_the_writing_vector_is_left_alone() -> None:
    """Only the prompt siting is scaled; the strength used while writing is not."""
    one = a_plan(1.0)
    four = a_plan(4.0)
    for layer in LAYERS:
        base = one.delta_for(layer, 0, with_noise=False, with_offset=False)
        scaled = four.delta_for(layer, 0, with_noise=False, with_offset=False)
        if base is None:
            continue
        assert torch.allclose(scaled, base, atol=1e-6), (
            f"layer {layer}: the multiplier changed the push used while writing"
        )
    print("  [PASS] the strength used while writing is unchanged")


def test_the_multiplier_is_in_the_run_id() -> None:
    """Two runs at different prompt strengths must not share a run id.

    They would overwrite each other's stories in the checkpoint, which is a
    silent wrong answer rather than a crash.
    """
    class Spec:
        def __init__(self, plan):
            self.steering_plan = plan

    one = _ortho_tag("M", Spec(a_plan(1.0)))
    four = _ortho_tag("M", Spec(a_plan(4.0)))
    assert one != four, "two prompt strengths produced the same run id"
    assert "__pg" in four, f"the prompt strength is missing from {four}"
    assert "__pg" not in one, f"the default prompt strength should add no tag: {one}"
    print(f"  [PASS] the prompt strength is in the run id  ({four.split('__')[-1]})")


if __name__ == "__main__":
    test_the_multiplier_scales_the_vector_the_prompt_is_given()
    test_the_writing_vector_is_left_alone()
    test_the_multiplier_is_in_the_run_id()
    print("\nok")
