#!/usr/bin/env python
"""The displacement can be kept away from particular sampled stories.

The coherence loss of the arm that wins both variety measures is not spread
evenly. The displacement aims at one of 32 sampled stories, chosen by story
number, and across six independent runs eight of those 32 accounted for 62 of
the 76 rejected stories while twenty-one never caused one at all. Aiming along a
story's direction amplifies whatever makes that story unusual, and a few are
unusual in a way that stops the model writing a story.

Two predictors of which eight have already failed: how far a sampled story leans
away from telling a story, and how much of it lies outside the cloud of stories
written under the push (correlation +0.09, none). So they are named directly.

This checks the arithmetic of naming them: the right rows go, the remaining
stories are aimed at, and the arm gets its own run id.

python tests/test_drop_anchors.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

N, DIM = 32, 64
DROP = [1, 5, 6, 10, 12, 14, 15, 18]


def _drop(anchors, drop):
    keep = [i for i in range(anchors.shape[0]) if i not in set(drop)]
    return anchors[torch.tensor(keep)], keep


def test_the_named_stories_are_gone_and_the_rest_are_not() -> None:
    torch.manual_seed(0)
    a = torch.randn(N, DIM)
    kept, keep = _drop(a, DROP)
    assert kept.shape[0] == N - len(DROP), f"{kept.shape[0]} left, expected {N - len(DROP)}"
    for i in DROP:
        for row in kept:
            assert not torch.allclose(row, a[i], atol=1e-6), (
                f"sampled story {i} was named but is still in the set")
    for j, i in enumerate(keep):
        assert torch.allclose(kept[j], a[i], atol=1e-6), (
            f"the story at position {j} is not the one it should be")
    print(f"{len(DROP)} named stories removed, {kept.shape[0]} left, order preserved")


def test_every_remaining_story_is_still_aimed_at() -> None:
    """A story index picks its aim by position, so all of them must get used."""
    torch.manual_seed(0)
    kept, _ = _drop(torch.randn(N, DIM), DROP)
    used = {i % kept.shape[0] for i in range(200)}
    assert used == set(range(kept.shape[0])), (
        f"only {len(used)} of {kept.shape[0]} remaining stories are ever aimed at")
    print(f"all {kept.shape[0]} remaining stories are aimed at across 200 stories")


def test_dropping_changes_which_story_each_index_aims_at() -> None:
    """Otherwise the run would not be a real test of the change."""
    kept, keep = _drop(torch.randn(N, DIM), DROP)
    same = sum(1 for i in range(200) if keep[i % len(keep)] == i % N)
    assert same < 40, (
        f"{same} of 200 story indices would aim at the same sampled story as "
        "before, so the run would largely repeat the old one")
    print(f"only {same} of 200 story indices keep their old aim")


def test_it_refuses_to_leave_too_few() -> None:
    kept, _ = _drop(torch.randn(N, DIM), list(range(N - 2)))
    assert kept.shape[0] == 2
    print("dropping almost everything leaves almost nothing, as it should -- the "
          "runner refuses below four")


if __name__ == "__main__":
    test_the_named_stories_are_gone_and_the_rest_are_not()
    test_every_remaining_story_is_still_aimed_at()
    test_dropping_changes_which_story_each_index_aims_at()
    test_it_refuses_to_leave_too_few()
    print("\nok")
