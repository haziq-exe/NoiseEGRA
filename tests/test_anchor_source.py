#!/usr/bin/env python
"""Aiming at the untouched model's stories, confined to where the steered ones vary.

Measured at 200 stories, the two clouds are good at different things.

Aiming at the *untouched* model's stories beats raised-temperature top-k on both
variety measures on intervals clear of zero -- what happens +3.3 [+0.8, +5.7],
wording +9.0 [+6.4, +11.8] -- and keeps 176 stories of 200.

Aiming at *steered* stories keeps 198 of 200 and wins neither variety measure:
the push makes stories alike, which is its job, so the cloud it produces has
less variation in what happens for a displacement to amplify.

So take the untouched stories' differences, which carry the variety, and drop
the part of each that points out of the subspace the steered stories occupy,
which is what takes the state somewhere the model does not write from. What is
left is a difference the model expresses and can still write from.

python tests/test_anchor_source.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The real shapes: two 31-direction subspaces inside a 2048-wide stream.
# On this model the projection keeps about 58% of each story and the two
# subspaces line up at 0.38, so it is neither a no-op nor a wipe-out --
# and a toy rank does not reproduce that.
DIM, N_STORIES, RANK = 2048, 32, 31


def _clouds(relatedness: float = 0.42):
    """Two related clouds, which is what they are.

    The steered stories are the same model writing the same task under a push,
    so the directions they vary along are a moved version of the untouched
    ones, not an independent subspace. Drawing the two independently is the
    wrong model and gives a projection that keeps 12% where the real one keeps
    58%: two independent 31-direction subspaces of a 2048-wide stream are very
    nearly orthogonal, and these are not. The default here is set to
    reproduce the 58% measured on Qwen3-1.7B.
    """
    torch.manual_seed(0)
    untouched = torch.randn(N_STORIES, DIM)
    untouched = untouched - untouched.mean(dim=0, keepdim=True)
    moved = untouched + (1.0 - relatedness) / relatedness * torch.randn(N_STORIES, DIM)
    steered = torch.linalg.qr(moved.t())[0][:, :RANK]
    return steered, untouched


def _project(untouched, steered):
    return (untouched @ steered) @ steered.t()


def test_the_projection_lands_inside_the_steered_subspace() -> None:
    steered, untouched = _clouds()
    a = _project(untouched, steered)
    # Everything left must be reachable from the steered directions alone.
    outside = a - (a @ steered) @ steered.t()
    worst = float(outside.norm(dim=1).max())
    assert worst < 1e-4, (
        f"a projected story keeps {worst:.4f} of its length outside the subspace "
        "the steered stories occupy, which is the part that was to be dropped"
    )
    print("every projected story lies inside the steered subspace")


def test_the_stories_stay_distinct_from_one_another() -> None:
    """A projection that collapsed them would destroy the variety it is for."""
    steered, untouched = _clouds()
    a = _project(untouched, steered)
    unit = torch.nn.functional.normalize(a, dim=1)
    sim = (unit @ unit.t()).abs()
    sim.fill_diagonal_(0.0)
    worst = float(sim.max())
    assert worst < 0.95, (
        f"two projected stories are {worst:.2f} alike, so the displacement would "
        "aim two stories the same way and the variety would be lost"
    )
    print(f"the most alike pair of projected stories is {worst:.2f}")


def test_it_keeps_a_real_share_of_each_story() -> None:
    """If almost nothing survived, this would be the steered cloud by another name."""
    steered, untouched = _clouds()
    a = _project(untouched, steered)
    share = float((a.norm(dim=1) / untouched.norm(dim=1)).mean())
    assert 0.3 < share < 0.9, (
        f"{share:.0%} of a story's difference survives the projection. Below a "
        "third the aim is decided by the steered directions and not by the "
        "story; above nine tenths nothing is being dropped and the arm is "
        "the untouched cloud under a new name. On this model it is 58%."
    )
    print(f"{share:.0%} of each story's difference survives the projection")


def test_it_is_not_the_same_as_aiming_at_steered_stories() -> None:
    steered, untouched = _clouds()
    a = _project(untouched, steered)
    torch.manual_seed(1)
    own = torch.randn(N_STORIES, DIM) @ steered @ steered.t()   # steered stories
    unit_a = torch.nn.functional.normalize(a, dim=1)
    unit_b = torch.nn.functional.normalize(own, dim=1)
    worst = float((unit_a * unit_b).sum(dim=1).abs().max())
    assert worst < 0.95, (
        "the projected untouched stories point the same way as the steered ones, "
        "so this arm would be a rerun of that one under a new name"
    )
    print(f"the two aims differ; the closest pair lines up at {worst:.2f}")


if __name__ == "__main__":
    test_the_projection_lands_inside_the_steered_subspace()
    test_the_stories_stay_distinct_from_one_another()
    test_it_keeps_a_real_share_of_each_story()
    test_it_is_not_the_same_as_aiming_at_steered_stories()
    print("\nok")
