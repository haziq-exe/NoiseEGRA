#!/usr/bin/env python
"""A shielded direction must leave the push alone and leave the perturbation unable to use it.

The per-story perturbation's measured cost is coherence, and the stories it
breaks are not garbled: the model refuses, or narrates its own planning. That is
a direction in the residual stream, so it can be named and removed from the
subspace the perturbation is drawn from.

Three things have to hold, and each has broken on this project before:

* the steering push is byte-identical with and without the shield, or the arm is
  not measuring the shield;
* no perturbation has any component along the shielded direction, which is the
  whole claim;
* the two arms get different run ids, or the second silently overwrites the
  first and both are reported as one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from noiseegra.steering_vectors import SteeringVectorSet  # noqa: E402
from noiseegra.setup_experiment import _ortho_tag  # noqa: E402
from noiseegra.setup_experiment import ExperimentSpec  # noqa: E402

DIM, LAYERS, RANK = 48, [6, 7], 16
NAMES = ["present_tense", "sensory", "named_character", "no_heading"]


def _vectors(shield, rank=0):
    torch.manual_seed(0)
    every = NAMES + ["in_story"]
    return SteeringVectorSet(
        vectors={n: {l: torch.randn(DIM) for l in LAYERS} for n in every},
        components={n: {l: torch.linalg.qr(torch.randn(DIM, 4))[0] for l in LAYERS}
                    for n in every},
        positives={n: {l: torch.randn(DIM) for l in LAYERS} for n in NAMES},
        shield=list(shield), shield_rank=rank,
    )


def _build(shield, rank=0):
    import run_orthosteer_experiment as ro

    vecs = _vectors(shield, rank)
    torch.manual_seed(1)
    basis = {l: torch.linalg.qr(torch.randn(DIM, RANK))[0] for l in LAYERS}
    return ro.make_plan(
        vecs, LAYERS, NAMES, beta=1.0, rms_scale=1.5,
        offset_gamma=0.15, offset_mode="orth", offset_basis=basis,
        offset_basis_kind="story", offset_prefill=True, offset_decode=False,
        steer_budget=2.0, steer_prefill=True,
    )


def test_the_push_is_unchanged_by_shielding() -> None:
    off, on = _build([]), _build(["in_story"])
    for layer in LAYERS:
        a = off.layer_plans[layer].basis
        b = on.layer_plans[layer].basis
        assert torch.allclose(a, b, atol=1e-6), (
            f"layer {layer}: the steering basis moved when a direction was shielded; "
            "the arm would be measuring a different push, not the shield"
        )
    print(f"push identical at layers {LAYERS} with and without the shield")


def test_the_perturbation_cannot_move_along_the_shielded_direction() -> None:
    off, on = _build([]), _build(["in_story"])
    vecs = _vectors(["in_story"])
    for layer in LAYERS:
        d = vecs.vectors["in_story"][layer]
        d = d / d.norm()
        leaks = []
        for plan, tag in ((off, "unshielded"), (on, "shielded")):
            worst = 0.0
            for story in range(40):
                torch.manual_seed(100 + story)
                plan.resample_offset(story_index=story)
                vec = plan.layer_plans[layer].offset
                assert vec is not None
                worst = max(worst, abs(float(d @ (vec / vec.norm()))))
            leaks.append((tag, worst))
        (_, free), (_, held) = leaks
        assert held < 1e-5, (
            f"layer {layer}: a shielded perturbation still has {held:.4f} of its "
            "length along the direction it is supposed to be held clear of"
        )
        assert free > 1e-3, (
            f"layer {layer}: the unshielded perturbation is already clear of the "
            f"direction ({free:.2e}), so this test would pass without the shield "
            "doing anything"
        )
        print(f"layer {layer}: worst overlap {free:.3f} unshielded -> {held:.1e} shielded")


def test_the_two_arms_get_different_run_ids() -> None:
    def tag(shield):
        spec = ExperimentSpec(use_orthogonal_steering=True,
                              steering_plan=_build(shield))
        return _ortho_tag("Qwen3-1.7B", spec)

    a, b = tag([]), tag(["in_story"])
    assert a != b, (
        "a shielded arm and an unshielded one produce the same run id, so the "
        "second would overwrite the first and both would be reported as one"
    )
    assert "__sh" in b and "__sh" not in a, f"the shield is not named in the id: {b}"
    print(f"ids differ:\n  {a}\n  {b}")


def test_a_wider_shield_removes_more() -> None:
    """The subspace shield must hold the perturbation clear of more than one axis.

    The mean direction alone removed every refusal at a perturbation of 0.15 and
    none of them at 0.25, which is what a region bounded in many directions
    looks like. So the shield has to be able to span several.
    """
    vecs = _vectors(["in_story"], 3)
    thin, wide = _build(["in_story"], 0), _build(["in_story"], 3)
    assert wide.protect_rank > thin.protect_rank, (
        f"a rank-3 shield protects {wide.protect_rank} directions and the mean "
        f"alone protects {thin.protect_rank}; the components are not being added"
    )
    for layer in LAYERS:
        comps = vecs.components["in_story"][layer][:, :3]
        for plan, tag, want in ((thin, "mean only", True), (wide, "rank 3", False)):
            torch.manual_seed(7)
            plan.resample_offset(story_index=0)
            vec = plan.layer_plans[layer].offset
            share = float((comps.t() @ (vec / vec.norm())).norm())
            if want:
                assert share > 1e-3, (
                    f"layer {layer}: even the mean-only shield already holds the "
                    f"perturbation clear of the components ({share:.2e}); the wider "
                    "shield would have nothing to do and this test would be hollow"
                )
            else:
                assert share < 1e-5, (
                    f"layer {layer}: a rank-3 shield leaves {share:.4f} of the "
                    "perturbation inside the subspace it is meant to exclude"
                )
    print(f"protected rank {thin.protect_rank} -> {wide.protect_rank} at shield rank 3")


if __name__ == "__main__":
    test_the_push_is_unchanged_by_shielding()
    test_the_perturbation_cannot_move_along_the_shielded_direction()
    test_the_two_arms_get_different_run_ids()
    test_a_wider_shield_removes_more()
    print("\nok")
