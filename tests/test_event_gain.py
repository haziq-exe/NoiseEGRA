#!/usr/bin/env python
"""How hard one direction is pushed can vary between stories.

The per-story displacement buys variety by moving the hidden state off the
region the model writes stories from, and that is exactly what it pays for in
coherence: at one and a half stories out the method keeps 190 of 200, at two it
keeps 170.

Varying how hard a single steered direction is pushed does not leave that region
at all. Every story is written from a state the push itself produced, with more
or less of one thing in it.

This was measured flat when every steered direction was about how a sentence is
formed -- present tense, sensory words, a named character, the format. Varying
those varies the style, not what the story is about. The event direction is the
first that changes what happens, so varying that one is a different experiment
with the same mechanism.

python tests/test_event_gain.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_orthosteer_experiment as ro  # noqa: E402
from noiseegra.setup_experiment import ExperimentSpec, _ortho_tag  # noqa: E402
from noiseegra.steering_vectors import SteeringVectorSet  # noqa: E402

DIM, LAYERS = 64, [6, 7]
NAMES = ["present_tense", "sensory", "named_character", "something_happens"]
TARGET = "something_happens"


def _plan(names_varied, kappa=0.6):
    torch.manual_seed(0)
    vecs = SteeringVectorSet(
        vectors={n: {l: torch.randn(DIM) for l in LAYERS} for n in NAMES},
        components={n: {l: torch.linalg.qr(torch.randn(DIM, 4))[0] for l in LAYERS}
                    for n in NAMES},
        positives={n: {l: torch.randn(DIM) for l in LAYERS} for n in NAMES},
    )
    return ro.make_plan(
        vecs, LAYERS, NAMES, beta=1.0, rms_scale=2.0, steer_budget=2.0,
        steer_prefill=True, jitter_mode="gain", jitter_kappa=kappa,
        jitter_names=names_varied,
    )


def test_only_the_named_direction_varies() -> None:
    plan = _plan([TARGET])
    k = NAMES.index(TARGET)
    seen = set()
    for story in range(30):
        torch.manual_seed(story)
        plan.resample_offset(story_index=story)
        g = plan.gains
        assert g is not None, "no gains were drawn"
        for i, name in enumerate(NAMES):
            if i == k:
                seen.add(round(g[i], 6))
            else:
                assert abs(g[i] - 1.0) < 1e-9, (
                    f"{name} was varied, but only {TARGET} should be")
    assert len(seen) > 20, f"{TARGET} took only {len(seen)} values over 30 stories"
    print(f"only {TARGET} varies; it took {len(seen)} distinct values over 30 stories")


def test_it_averages_to_the_push_that_was_asked_for() -> None:
    """Lognormal with mean one, so the expected push is unchanged."""
    plan = _plan([TARGET])
    k = NAMES.index(TARGET)
    vals = []
    for story in range(4000):
        torch.manual_seed(story)
        plan.resample_offset(story_index=story)
        vals.append(plan.gains[k])
    mean = sum(vals) / len(vals)
    assert 0.94 < mean < 1.06, (
        f"the varied push averages {mean:.3f} of the one asked for; a mean away "
        "from one would make this a change of strength rather than a spread")
    assert min(vals) > 0, "a draw flipped the direction's sign and pushed against it"
    print(f"the varied push averages {mean:.3f} of the one asked for, never negative")


def test_varying_one_direction_is_its_own_arm() -> None:
    def tag(names_varied):
        return _ortho_tag("M", ExperimentSpec(use_orthogonal_steering=True,
                                              steering_plan=_plan(names_varied)))
    one, all_of_them = tag([TARGET]), tag(None)
    assert one != all_of_them, (
        "varying one direction and varying every direction share a run id")
    assert "__jn" in one and "__jn" not in all_of_them
    print("varying one direction gets its own run id")


if __name__ == "__main__":
    test_only_the_named_direction_varies()
    test_it_averages_to_the_push_that_was_asked_for()
    test_varying_one_direction_is_its_own_arm()
    print("\nok")
