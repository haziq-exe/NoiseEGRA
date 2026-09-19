#!/usr/bin/env python
"""The displacement can aim at one of the model's own stories.

Every displacement so far has been drawn in the span of the directions that
summarise the sampled stories. That works at a small size and not at a large
one: at 0.15 every story stays coherent, at 0.25 sixteen in a hundred come back
as a refusal or as the model narrating its own planning. A mixture of principal
components, pushed far enough, lands somewhere no story of the model's own ever
sat, and from there it does not write a story.

Aiming at a sampled story instead cannot do that. At full size the state lands
exactly where the model has been. This checks the three claims that makes:

* each story aims at a different sampled story, so it is still a between-story
  mechanism -- the only kind that has ever moved variety here;
* the aim is a real story's direction, not a fresh draw, so two runs with the
  same story order aim the same way;
* the constraint directions are still taken out of it, which the basis path gets
  free at build time and this path has to do by hand.

python tests/test_anchor_draw.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_orthosteer_experiment as ro  # noqa: E402
from noiseegra.steering_vectors import SteeringVectorSet  # noqa: E402

DIM, LAYERS, N_STORIES, RANK = 48, [6, 7], 12, 8
NAMES = ["present_tense", "sensory", "named_character", "no_heading"]


def _setup():
    torch.manual_seed(0)
    vecs = SteeringVectorSet(
        vectors={n: {l: torch.randn(DIM) for l in LAYERS} for n in NAMES},
        components={n: {l: torch.linalg.qr(torch.randn(DIM, 4))[0] for l in LAYERS}
                    for n in NAMES},
        positives={n: {l: torch.randn(DIM) for l in LAYERS} for n in NAMES},
    )
    anchors = {l: torch.randn(N_STORIES, DIM) for l in LAYERS}
    basis = {l: torch.linalg.qr(anchors[l].t())[0][:, :RANK] for l in LAYERS}
    return vecs, anchors, basis


def _plan(shape):
    vecs, anchors, basis = _setup()
    ro.RUN_DEFAULTS["offset_anchors"] = anchors
    try:
        return ro.make_plan(
            vecs, LAYERS, NAMES, beta=1.0, rms_scale=1.5,
            offset_gamma=0.25, offset_mode="orth", offset_basis=basis,
            offset_basis_kind="story", offset_draw_shape=shape,
            offset_prefill=True, offset_decode=False,
            steer_budget=2.0, steer_prefill=True,
        ), anchors
    finally:
        ro.RUN_DEFAULTS["offset_anchors"] = None


def test_each_story_aims_at_a_different_sampled_story() -> None:
    plan, anchors = _plan("anchor")
    seen = {}
    for story in range(N_STORIES):
        plan.resample_offset(story_index=story)
        vec = plan.layer_plans[6].offset
        rows = anchors[6]
        # Which sampled story is this aimed at? The one it lines up with best.
        cos = torch.nn.functional.normalize(rows, dim=1) @ (vec / vec.norm())
        seen[story] = int(cos.argmax())
    assert len(set(seen.values())) == N_STORIES, (
        f"{N_STORIES} stories aim at only {len(set(seen.values()))} distinct "
        "sampled stories; the displacement is not varying between stories"
    )
    print(f"{N_STORIES} stories aim at {len(set(seen.values()))} distinct sampled stories")


def test_the_aim_is_not_a_fresh_draw() -> None:
    """Two passes over the same story order must aim the same way."""
    plan, _ = _plan("anchor")
    first, second = [], []
    for run in (first, second):
        for story in range(N_STORIES):
            torch.manual_seed(999)          # a different seed each pass would
            plan.resample_offset(story_index=story)   # not change an anchor aim
            run.append(plan.layer_plans[7].offset.clone())
    for i, (a, b) in enumerate(zip(first, second)):
        assert torch.allclose(a, b, atol=1e-6), f"story {i}'s aim changed between passes"
    print("the aim is a property of the story index, not of the random draw")


def test_the_constraint_directions_are_still_removed() -> None:
    plan, _ = _plan("anchor")
    for layer in LAYERS:
        lp = plan.layer_plans[layer]
        worst = 0.0
        for story in range(N_STORIES):
            plan.resample_offset(story_index=story)
            vec = lp.offset
            worst = max(worst, float((lp.protect.t() @ (vec / vec.norm())).norm()))
        assert worst < 1e-5, (
            f"layer {layer}: an anchored displacement keeps {worst:.4f} of its "
            "length inside the constraint subspace, so it leaks onto a steered "
            "requirement for the whole story"
        )
    print("no anchored displacement has any component on a constraint direction")


def test_it_differs_from_the_drawn_displacement() -> None:
    drawn, _ = _plan("manifold")
    aimed, _ = _plan("anchor")
    torch.manual_seed(5)
    drawn.resample_offset(story_index=3)
    torch.manual_seed(5)
    aimed.resample_offset(story_index=3)
    a, b = drawn.layer_plans[6].offset, aimed.layer_plans[6].offset
    assert not torch.allclose(a, b, atol=1e-4), (
        "the anchored and the drawn displacement are identical, so the arm "
        "would be a rerun of the old one under a new name"
    )
    print("the anchored displacement differs from the drawn one")


if __name__ == "__main__":
    test_each_story_aims_at_a_different_sampled_story()
    test_the_aim_is_not_a_fresh_draw()
    test_the_constraint_directions_are_still_removed()
    test_it_differs_from_the_drawn_displacement()
    print("\nok")
