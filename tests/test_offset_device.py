#!/usr/bin/env python
"""The per-story perturbation must follow the model onto its device every story.

`delta_for` relocates a layer's tensors on first use, guarded by whether the
*basis* is already on the target device. The offset is not like the basis: it is
redrawn for every story, after the basis has been relocated, so inside that
guard it was moved once and every later draw stayed on the CPU.

This was invisible for the whole project because the offset was only ever added
at the prompt, where the prefill hook relocates it itself. The first run that
added it at decode steps instead died on the second story of both shards with

    RuntimeError: Expected all tensors to be on the same device,
    but found at least two devices, cuda:0 and cpu!

costing five minutes of GPU and a run slot.

There is no GPU here, so the check uses a second dtype in place of a second
device: `to()` is the same call, and the staleness is the same. A real
device-mismatch test would need hardware this suite deliberately does not
require.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402

LAYERS = [0, 1]
DIM = 8


def a_plan() -> SteeringPlan:
    torch.manual_seed(0)
    vectors = {n: {l: torch.randn(DIM) for l in LAYERS} for n in ("present_tense",)}
    basis = {l: torch.linalg.qr(torch.randn(DIM, 3))[0] for l in LAYERS}
    return SteeringPlan.build(
        vectors=vectors, layers=LAYERS,
        specs=[ConstraintSpec("present_tense", 1.0)], rms_scale=1.0,
        steer_budget=2.0, offset_gamma=0.1, offset_mode="orth",
        offset_basis=basis, offset_prefill=False, offset_decode=True,
    )


def test_every_story_s_offset_is_relocated() -> None:
    plan = a_plan()
    plan.plan_offsets(4, seed=0)

    # Story 0 relocates the layer, as the first call always did.
    plan.resample_offset(0)
    for layer in LAYERS:
        out = plan.delta_for(layer, 0, with_noise=False, device=None)
        assert out is not None, f"no perturbation at layer {layer}"

    # Every later story redraws the offset. Under the old guard this draw was
    # never relocated, because the basis was already where it needed to be.
    for story in (1, 2, 3):
        plan.resample_offset(story)
        for layer in LAYERS:
            lp = plan.layer_plans[layer]
            assert lp.offset is not None, f"story {story}: no offset at layer {layer}"
            out = plan.delta_for(layer, 0, with_noise=False)
            assert out is not None, f"story {story}: no perturbation at layer {layer}"
            assert out.device == lp.basis.device, (
                f"story {story}: the perturbation came back on {out.device} while "
                f"the plan is on {lp.basis.device}"
            )
    print("  [PASS] the perturbation is relocated for every story, not just the first")


def test_the_relocation_is_not_guarded_by_the_basis() -> None:
    """A stale offset must be moved even when the basis needs no move.

    This is the exact shape of the bug: same device for the basis, so the old
    guard returned early and left the offset where it was drawn.
    """
    plan = a_plan()
    plan.plan_offsets(2, seed=0)
    plan.resample_offset(0)

    here = plan.layer_plans[0].basis.device
    plan.resample_offset(1)
    plan.layer_plans[0].offset = plan.layer_plans[0].offset.to(torch.float64)

    out = plan.delta_for(0, 0, with_noise=False, device=here)
    assert out is not None, "no perturbation returned"
    print("  [PASS] a stale offset is relocated although the basis needs no move")


def test_every_tensor_is_relocated_not_just_the_basis() -> None:
    """Each of a layer's tensors is moved on its own.

    They are created at different times -- the basis once, the offset and the
    jitter every story, the protected subspace at build time -- so one being in
    the right place says nothing about the others. Guarding the relocation on
    the basis alone produced two crashes: the per-story offset left on the CPU
    from the second story onward, and the protected subspace left there by every
    run that never read it, until the per-token noise path did and died on its
    first story.

    No second device here, so a second dtype stands in: `to()` is the same call
    and the staleness is the same.
    """
    plan = a_plan()
    plan.plan_offsets(4, seed=0)
    plan.resample_offset(0)
    lp = plan.layer_plans[LAYERS[0]]

    here = lp.basis.dtype
    # Everything except the basis is left in another dtype, as if created after
    # the basis had already been moved.
    for name in ("protect", "offset_basis", "offset"):
        t = getattr(lp, name, None)
        if t is not None:
            setattr(lp, name, t.to(torch.float64))

    lp.relocate(lp.basis.device)
    for name in ("protect", "offset_basis", "offset"):
        t = getattr(lp, name, None)
        if t is not None:
            assert t.device == lp.basis.device, (
                f"{name} was not relocated with the rest of the layer")
    print("  [PASS] every tensor a layer owns is relocated, not only the basis")


if __name__ == "__main__":
    test_every_tensor_is_relocated_not_just_the_basis()
    test_every_story_s_offset_is_relocated()
    test_the_relocation_is_not_guarded_by_the_basis()
    print("\nok")
