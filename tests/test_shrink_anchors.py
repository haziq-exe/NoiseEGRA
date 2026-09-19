#!/usr/bin/env python
"""A story can be travelled towards only part of the way.

Measured at 200 stories: the eight sampled stories that break the writing are
the same eight that carry the variety. Removing them took coherence from 176 of
200 to 189 and turned a win on variety of what happens (+3.5 over top-k, clear
of zero) into a tie (+0.6). Their directions were doing both jobs.

So keep the direction and shorten the distance. A story aiming at one of them
travels half as far; a story aiming at any other travels the full distance.

python tests/test_shrink_anchors.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_orthosteer_experiment as ro  # noqa: E402
from noiseegra.steering_vectors import SteeringVectorSet  # noqa: E402

DIM, LAYERS, N, RANK = 2048, [6, 7], 16, 8
NAMES = ["present_tense", "sensory", "named_character", "no_heading"]
SHRINK, FACTOR, RADIUS = [1, 5, 6], 0.5, 7.0


def _plan(scale):
    torch.manual_seed(0)
    vecs = SteeringVectorSet(
        vectors={n: {l: torch.randn(DIM) for l in LAYERS} for n in NAMES},
        components={n: {l: torch.linalg.qr(torch.randn(DIM, 4))[0] for l in LAYERS}
                    for n in NAMES},
        positives={n: {l: torch.randn(DIM) for l in LAYERS} for n in NAMES},
    )
    raw = torch.randn(N, DIM)
    raw = raw - raw.mean(dim=0, keepdim=True)
    raw = raw / raw.norm(dim=1, keepdim=True) * RADIUS
    ro.RUN_DEFAULTS["offset_anchors"] = {l: raw.clone() for l in LAYERS}
    ro.RUN_DEFAULTS["anchor_scale"] = scale
    ro.RUN_DEFAULTS["offset_norm"] = "story"
    try:
        return ro.make_plan(
            vecs, LAYERS, NAMES, beta=1.0, rms_scale=2.215,
            offset_gamma=1.0, offset_mode="orth",
            offset_basis={l: torch.linalg.qr(raw.t())[0][:, :RANK] for l in LAYERS},
            offset_basis_kind="story", offset_draw_shape="anchor",
            offset_prefill=True, offset_decode=False,
            steer_budget=2.0, steer_prefill=True,
        )
    finally:
        for k, v in (("offset_anchors", None), ("anchor_scale", None),
                     ("offset_norm", "energy")):
            ro.RUN_DEFAULTS[k] = v


def _lengths(plan):
    out = {}
    for story in range(N):
        plan.resample_offset(story_index=story)
        out[story] = float(plan.layer_plans[6].offset.norm())
    return out


def test_the_named_stories_are_travelled_towards_less_far() -> None:
    scale = torch.ones(N)
    for i in SHRINK:
        scale[i] = FACTOR
    short, full = _lengths(_plan(scale)), _lengths(_plan(None))
    for i in range(N):
        want = full[i] * (FACTOR if i in SHRINK else 1.0)
        assert abs(short[i] - want) < 0.05 * max(want, 1e-9), (
            f"story {i} travelled {short[i]:.2f} where {want:.2f} was asked for")
    print(f"the {len(SHRINK)} named stories are travelled towards at "
          f"{FACTOR:g} of the distance, the other {N - len(SHRINK)} at full")


def test_their_direction_is_unchanged() -> None:
    """Shortening must not turn the story, or the variety it carries is lost."""
    scale = torch.ones(N)
    for i in SHRINK:
        scale[i] = FACTOR
    a, b = _plan(scale), _plan(None)
    for i in SHRINK:
        a.resample_offset(story_index=i)
        b.resample_offset(story_index=i)
        va, vb = a.layer_plans[6].offset, b.layer_plans[6].offset
        cos = float((va / va.norm()) @ (vb / vb.norm()))
        assert cos > 0.999, (
            f"story {i}'s aim turned by {cos:.4f} when it was only meant to be "
            "shortened")
    print("the shortened stories point exactly where they did")


def test_it_is_not_the_same_as_dropping_them() -> None:
    scale = torch.ones(N)
    for i in SHRINK:
        scale[i] = FACTOR
    lens = _lengths(_plan(scale))
    assert all(lens[i] > 0 for i in SHRINK), (
        "a shrunk story is not displaced at all, which is dropping it")
    print("every story is still displaced, so no direction is lost")


def test_each_story_can_have_its_own_fraction() -> None:
    """One factor for all of them is blunt: some break the writing far more often.

    The fraction comes from the calibration's own count -- one over one plus the
    number of rejections that story caused -- so a story that broke it five
    times is travelled towards at a sixth of the distance and one that broke it
    once at a half.
    """
    counts = {1: 5, 5: 2, 6: 1}
    scale = torch.ones(N)
    for i, c in counts.items():
        scale[i] = 1.0 / (1.0 + c)
    short, full = _lengths(_plan(scale)), _lengths(_plan(None))
    for i, c in counts.items():
        want = full[i] / (1.0 + c)
        assert abs(short[i] - want) < 0.05 * want, (
            f"story {i} broke the writing {c} times and travelled {short[i]:.2f} "
            f"where {want:.2f} was asked for")
    assert short[1] < short[5] < short[6], (
        "the story that broke the writing most is not the one travelled towards "
        "least far")
    print("each story is travelled towards by its own fraction, shortest for the "
          "one that broke the writing most")


if __name__ == "__main__":
    test_the_named_stories_are_travelled_towards_less_far()
    test_their_direction_is_unchanged()
    test_it_is_not_the_same_as_dropping_them()
    test_each_story_can_have_its_own_fraction()
    print("\nok")
