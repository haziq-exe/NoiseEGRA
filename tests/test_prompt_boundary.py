#!/usr/bin/env python
"""The final prompt positions must really be left unperturbed when asked.

The prompt does not end with the instruction; it ends with the chat template's
own tokens, which say the user has stopped and the assistant is answering.
Perturbing those along with everything else weakens the only signal that this is
a reply, and the best-scoring arm opens 29% of its stories with a title -- which
the instruction forbids and which no baseline produces at all.

This checks the mechanics of sparing them, with a stand-in for the residual
stream, because the failure is silent in both directions: if the slice is wrong
the boundary stays perturbed and the sweep reads as a null, and if it spares too
much the perturbation never reaches the instruction it is meant to move.

Offline, no GPU, no model.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from noiseegra.setup_experiment import _ortho_tag  # noqa: E402
from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402

LAYERS = [1]
DIM = 8
NAMES = ["present_tense", "sensory"]


def a_plan(keep: int) -> SteeringPlan:
    torch.manual_seed(0)
    vectors = {n: {l: torch.randn(DIM) for l in LAYERS} for n in NAMES}
    return SteeringPlan.build(
        vectors=vectors, layers=LAYERS, specs=[ConstraintSpec(n, 1.0) for n in NAMES],
        rms_scale=1.0, steer_budget=2.0, steer_prefill=True, prompt_tail_clear=keep,
    )


def apply_like_the_hook(plan: SteeringPlan, n_positions: int) -> torch.Tensor:
    """The slicing the prefill hook does, over a zeroed stand-in stream."""
    target = torch.zeros(1, n_positions, DIM)
    delta = plan.steering_only(LAYERS[0], 0)
    assert delta is not None, "no vector at the prompt"
    keep = plan.prompt_tail_clear
    d = delta.view(1, 1, -1)
    if keep > 0 and target.shape[1] > keep:
        target[:, :-keep, :].add_(d)
    else:
        target.add_(d)
    return target


def test_the_last_positions_are_untouched() -> None:
    n = 20
    for keep in (2, 4, 8):
        out = apply_like_the_hook(a_plan(keep), n)
        moved = (out.abs().sum(-1) > 0)[0]
        assert moved[: n - keep].all(), f"keep={keep}: the instruction was not perturbed"
        assert not moved[n - keep:].any(), f"keep={keep}: the boundary was perturbed"
    print("  [PASS] the last N prompt positions are left alone, the rest are moved")


def test_zero_keeps_the_old_behaviour() -> None:
    out = apply_like_the_hook(a_plan(0), 20)
    assert (out.abs().sum(-1) > 0).all(), "every position should move when nothing is spared"
    print("  [PASS] sparing nothing perturbs every prompt position, as before")


def test_a_short_prompt_is_not_wiped_out() -> None:
    """Sparing more positions than the prompt has must not silently do nothing.

    The guard leaves the perturbation applied everywhere rather than nowhere, so
    a mis-set value cannot quietly disable the mechanism it is configuring.
    """
    out = apply_like_the_hook(a_plan(8), 4)
    assert (out.abs().sum(-1) > 0).all(), "a short prompt lost its perturbation entirely"
    print("  [PASS] a prompt shorter than the spared tail still gets perturbed")


def test_the_spared_count_is_in_the_run_id() -> None:
    class Spec:
        def __init__(self, plan):
            self.steering_plan = plan

    none = _ortho_tag("M", Spec(a_plan(0)))
    four = _ortho_tag("M", Spec(a_plan(4)))
    assert none != four, "two settings produced the same run id"
    assert "__tail4" in four, f"the spared count is missing from {four}"
    assert "__tail" not in none, f"sparing nothing should add no tag: {none}"
    print("  [PASS] the spared count is in the run id")


if __name__ == "__main__":
    test_the_last_positions_are_untouched()
    test_zero_keeps_the_old_behaviour()
    test_a_short_prompt_is_not_wiped_out()
    test_the_spared_count_is_in_the_run_id()
    print("\nok")
