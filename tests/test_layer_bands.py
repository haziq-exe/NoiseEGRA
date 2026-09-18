#!/usr/bin/env python
"""The push and the perturbation must act only on the layers they are given.

They have no reason to want the same band. The constraint push controls
properties of the sentence being written and works at layers 6-13 of 28 while
fragmenting the text at 14-22. The perturbation's job is to send the model to a
different story, which need not be decided where a sentence is worded. Every run
before this used one band for both because one flag set both.

The failure this guards is silent in the worst way: if the gating does not work,
every arm of a band sweep is identical, the sweep reads as "where the
perturbation is applied does not matter", and that is indistinguishable from a
real null. Three suites in this project once ran isotropic noise while their run
ids claimed an estimated basis, which is the same shape of fault.

Offline, no GPU, no model.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from noiseegra.setup_experiment import _ortho_tag  # noqa: E402
from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402

ALL = [6, 7, 8, 9, 10, 11]
PUSH = [6, 7, 8]
OFFSET = [9, 10, 11]
DIM = 12
NAMES = ["present_tense", "sensory"]


def a_plan(push=None, offset=None, gamma=0.1) -> SteeringPlan:
    torch.manual_seed(0)
    vectors = {n: {l: torch.randn(DIM) for l in ALL} for n in NAMES}
    basis = {l: torch.linalg.qr(torch.randn(DIM, 4))[0] for l in ALL}
    plan = SteeringPlan.build(
        vectors=vectors, layers=ALL, specs=[ConstraintSpec(n, 1.0) for n in NAMES],
        rms_scale=1.0, steer_budget=2.0, steer_prefill=True,
        offset_gamma=gamma, offset_mode="orth", offset_basis=basis,
        offset_prefill=True, offset_decode=True,
        push_layers=push, offset_layers=offset,
    )
    plan.plan_offsets(4, seed=0)
    plan.resample_offset(0)
    return plan


def test_the_push_is_silent_outside_its_band() -> None:
    plan = a_plan(push=PUSH, offset=OFFSET)
    for layer in PUSH:
        assert plan.steering_only(layer, 0) is not None, f"no push at layer {layer}"
    for layer in OFFSET:
        assert plan.steering_only(layer, 0) is None, (
            f"the push reached layer {layer}, which is outside its band")
    print("  [PASS] the push acts only on the layers it was given")


def test_the_perturbation_is_silent_outside_its_band() -> None:
    plan = a_plan(push=PUSH, offset=OFFSET)
    quiet = a_plan(push=PUSH, offset=OFFSET, gamma=0.0)
    for layer in OFFSET:
        with_offset = plan.delta_for(layer, 0, with_noise=False)
        without = quiet.delta_for(layer, 0, with_noise=False)
        moved = with_offset if without is None else (
            with_offset - without if with_offset is not None else None)
        assert moved is not None and moved.norm() > 0, (
            f"no perturbation at layer {layer}, which is inside its band")
    for layer in PUSH:
        a = plan.delta_for(layer, 0, with_noise=False)
        b = quiet.delta_for(layer, 0, with_noise=False)
        if a is None and b is None:
            continue
        assert torch.allclose(a, b, atol=1e-6), (
            f"the perturbation reached layer {layer}, which is outside its band")
    print("  [PASS] the perturbation acts only on the layers it was given")


def test_empty_bands_mean_every_layer() -> None:
    """The default must be what every earlier run did: both act everywhere."""
    plan = a_plan()
    for layer in ALL:
        assert plan.steering_only(layer, 0) is not None, (
            f"the default silenced the push at layer {layer}")
    print("  [PASS] giving no bands leaves both acting on every layer, as before")


def apply_like_the_prefill_hook(plan, layer, n_positions=12):
    """The slicing and gating the prefill hook does, over a zeroed stand-in stream.

    This exists because testing the plan's own methods was not enough. The band
    gating was added to the decode path only, and every run in this project
    perturbs the prompt and leaves decoding alone -- so a sweep over three bands
    produced three byte-identical arms and read as a clean null, with the run ids
    faithfully recording bands that had no effect.
    """
    target = torch.zeros(1, n_positions, DIM)
    delta = plan.steering_only(layer, 0)
    bands = plan.offset_layers or ()
    if plan.offset_prefill and (not bands or layer in bands):
        off = plan.layer_plans[layer].offset
        if off is not None:
            delta = off if delta is None else delta + off
    if delta is not None:
        target.add_(delta.view(1, 1, -1))
    return target


def test_the_prompt_perturbation_honours_its_band() -> None:
    plan = a_plan(push=PUSH, offset=OFFSET)
    quiet = a_plan(push=PUSH, offset=OFFSET, gamma=0.0)
    for layer in OFFSET:
        moved = apply_like_the_prefill_hook(plan, layer)
        flat = apply_like_the_prefill_hook(quiet, layer)
        assert not torch.allclose(moved, flat, atol=1e-6), (
            f"layer {layer} is inside the perturbation's band but the prompt "
            f"was not perturbed there")
    for layer in PUSH:
        moved = apply_like_the_prefill_hook(plan, layer)
        flat = apply_like_the_prefill_hook(quiet, layer)
        assert torch.allclose(moved, flat, atol=1e-6), (
            f"layer {layer} is outside the perturbation's band but the prompt "
            f"was perturbed there")
    print("  [PASS] the prompt perturbation acts only on the layers it was given")


def test_the_bands_are_in_the_run_id() -> None:
    class Spec:
        def __init__(self, plan):
            self.steering_plan = plan

    same = _ortho_tag("M", Spec(a_plan(push=ALL, offset=ALL)))
    split = _ortho_tag("M", Spec(a_plan(push=PUSH, offset=OFFSET)))
    assert same != split, "two band settings produced the same run id"
    assert "__ol9-11" in split, f"the perturbation's band is missing from {split}"
    print("  [PASS] a split band is recorded in the run id")


if __name__ == "__main__":
    test_the_push_is_silent_outside_its_band()
    test_the_perturbation_is_silent_outside_its_band()
    test_the_prompt_perturbation_honours_its_band()
    test_empty_bands_mean_every_layer()
    test_the_bands_are_in_the_run_id()
    print("\nok")
