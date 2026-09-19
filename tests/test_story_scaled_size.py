#!/usr/bin/env python
"""The displacement's size must be measurable against the model's own stories.

For fifty rounds the size was measured against the hidden state's own length,
which has no relation to how far the model's stories sit from one another. On
Qwen3-1.7B at layer 6 a real story sits 6.9 from the average of the sampled
stories, and the size every good arm was run at -- 0.15 -- works out at 15.0.
Every arm, including the ones that kept every story coherent, was displacing to
somewhere at least twice as far out as any story the model had written. That is
what the refusals are: the model finding itself outside its own distribution and
falling back on being an assistant.

Measured against the stories instead, 1.0 means as far out as a typical story,
below 1.0 is inside the cloud and above it is outside.

python tests/test_story_scaled_size.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_orthosteer_experiment as ro  # noqa: E402
from noiseegra.steering_vectors import SteeringVectorSet  # noqa: E402

# The real hidden size, because the comparison between the two units is a
# ratio of sqrt(dim) to a story's distance and a toy dimension does not
# reproduce it.
DIM, LAYERS, N_STORIES, RANK = 2048, [6, 7], 16, 8
NAMES = ["present_tense", "sensory", "named_character", "no_heading"]
RADIUS = 7.0


def _build(norm, gamma):
    torch.manual_seed(0)
    vecs = SteeringVectorSet(
        vectors={n: {l: torch.randn(DIM) for l in LAYERS} for n in NAMES},
        components={n: {l: torch.linalg.qr(torch.randn(DIM, 4))[0] for l in LAYERS}
                    for n in NAMES},
        positives={n: {l: torch.randn(DIM) for l in LAYERS} for n in NAMES},
    )
    # Sampled stories all sitting exactly RADIUS from their average, so the
    # expected displacement length is exactly gamma * RADIUS and the test does
    # not depend on a sampling accident.
    raw = torch.randn(N_STORIES, DIM)
    raw = raw - raw.mean(dim=0, keepdim=True)
    raw = raw / raw.norm(dim=1, keepdim=True) * RADIUS
    anchors = {l: raw.clone() for l in LAYERS}
    basis = {l: torch.linalg.qr(raw.t())[0][:, :RANK] for l in LAYERS}
    ro.RUN_DEFAULTS["offset_anchors"] = anchors
    ro.RUN_DEFAULTS["offset_norm"] = norm
    try:
        return ro.make_plan(
            vecs, LAYERS, NAMES, beta=1.0, rms_scale=2.215,
            offset_gamma=gamma, offset_mode="orth", offset_basis=basis,
            offset_basis_kind="story", offset_prefill=True, offset_decode=False,
            steer_budget=2.0, steer_prefill=True,
        )
    finally:
        ro.RUN_DEFAULTS["offset_anchors"] = None
        ro.RUN_DEFAULTS["offset_norm"] = "energy"


def test_one_story_out_is_one_story_out() -> None:
    plan = _build("story", 1.0)
    for layer in LAYERS:
        plan.resample_offset(story_index=0)
        got = float(plan.layer_plans[layer].offset.norm())
        assert abs(got - RADIUS) < 0.2 * RADIUS, (
            f"layer {layer}: a size of 1.0 displaced by {got:.1f} when a story "
            f"of the model's own sits {RADIUS:.1f} from the average"
        )
    print(f"a size of 1.0 displaces by about {RADIUS:.1f}, one story's distance")


def test_half_a_story_is_inside_the_cloud() -> None:
    plan = _build("story", 0.5)
    plan.resample_offset(story_index=0)
    got = float(plan.layer_plans[6].offset.norm())
    assert got < RADIUS, f"a size of 0.5 displaced by {got:.1f}, outside the stories"
    print(f"a size of 0.5 displaces by {got:.1f}, inside the cloud of real stories")


def test_the_old_unit_was_far_outside_it() -> None:
    """The size every good arm ran at, in the old unit, against the data."""
    plan = _build("energy", 0.15)
    plan.resample_offset(story_index=0)
    got = float(plan.layer_plans[6].offset.norm())
    assert got > 2 * RADIUS, (
        f"a size of 0.15 in the old unit displaced by only {got:.1f}; this test "
        f"records that it is far outside the stories' own {RADIUS:.1f}, and if "
        "that is no longer true the history in the docstring is wrong"
    )
    print(f"the old unit at 0.15 displaces by {got:.1f}, "
          f"{got / RADIUS:.1f} times a real story's distance")


def test_the_two_units_get_different_run_ids() -> None:
    from noiseegra.setup_experiment import ExperimentSpec, _ortho_tag

    def tag(norm):
        return _ortho_tag("M", ExperimentSpec(use_orthogonal_steering=True,
                                              steering_plan=_build(norm, 0.5)))
    a, b = tag("energy"), tag("story")
    assert a != b, "the two size units share a run id, so one overwrites the other"
    print("the two units get different run ids")


if __name__ == "__main__":
    test_one_story_out_is_one_story_out()
    test_half_a_story_is_inside_the_cloud()
    test_the_old_unit_was_far_outside_it()
    test_the_two_units_get_different_run_ids()
    print("\nok")
