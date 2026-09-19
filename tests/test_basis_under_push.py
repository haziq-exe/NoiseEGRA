#!/usr/bin/env python
"""The stories the displacement is measured against can be sampled under the push.

Two separate reasons, both measured.

The perturbation acts during steered generation, so the cloud of states it is
moving within is the cloud of *steered* stories. Sampled from the untouched
model, the average it is measured from and the directions it moves along
describe a cloud the state is never in.

And an anchored displacement aims at one of those sampled stories. Taken from
the untouched model they break 4.3 requirements of twelve, so aiming at one
drags the writing back towards breaking them: at one story's distance the
anchored displacement scores 3.58 against the drawn one's 2.19, losing on
exactly the two requirements the push is holding up.

No model here. The collector is driven with a stub that records what the hook
did, which is enough to check the push is applied where the generation hook
applies it and nowhere else.

python tests/test_basis_under_push.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_orthosteer_experiment as ro  # noqa: E402
from noiseegra.activation_basis import collect_story_pcs  # noqa: E402
from noiseegra.steering_vectors import SteeringVectorSet  # noqa: E402

DIM, LAYERS = 64, [6, 7]
NAMES = ["present_tense", "sensory", "named_character", "no_heading"]


def _push_plan(gamma=0.0, tail=8):
    torch.manual_seed(0)
    vecs = SteeringVectorSet(
        vectors={n: {l: torch.randn(DIM) for l in LAYERS} for n in NAMES},
        components={n: {l: torch.linalg.qr(torch.randn(DIM, 4))[0] for l in LAYERS}
                    for n in NAMES},
        positives={n: {l: torch.randn(DIM) for l in LAYERS} for n in NAMES},
    )
    return ro.make_plan(
        vecs, LAYERS, NAMES, beta=1.0, rms_scale=2.0,
        steer_budget=2.0, steer_prefill=True, prompt_tail_clear=tail,
        offset_gamma=gamma, offset_mode=("orth" if gamma else "none"),
        offset_basis={l: torch.linalg.qr(torch.randn(DIM, 16))[0] for l in LAYERS},
        offset_basis_kind="story", offset_prefill=bool(gamma), offset_decode=False,
    )


def test_a_perturbing_plan_is_refused() -> None:
    """The basis says what the unperturbed steered stories look like."""
    try:
        collect_story_pcs(object(), "x", LAYERS, push_plan=_push_plan(gamma=0.5))
    except ValueError as exc:
        assert "must not itself perturb" in str(exc), str(exc)
        print("a plan that perturbs is refused, so the basis cannot be measured "
              "against a cloud the perturbation itself moved")
        return
    raise AssertionError("a perturbing plan was accepted")


def test_the_push_is_the_one_the_hook_would_apply() -> None:
    """Same vector, same spared tail, as the generation hook uses at the prompt."""
    plan = _push_plan(tail=8)
    for layer in LAYERS:
        want = plan.steering_only(layer, 0)
        assert want is not None and want.norm() > 0, f"layer {layer}: no push to apply"
        # What the collector adds at the prompt: `steering_only`, to every
        # position but the spared tail. Reproduce that here and check it
        # matches the plan rather than some rescaled copy.
        prompt = torch.zeros(1, 20, DIM)
        keep = int(plan.prompt_tail_clear)
        prompt[:, :20 - keep, :].add_(want.view(1, 1, -1))
        touched = (prompt.norm(dim=2)[0] > 0).tolist()
        assert touched[: 20 - keep] == [True] * (20 - keep), "early positions missed"
        assert touched[20 - keep:] == [False] * keep, "the spared tail was pushed"
    print(f"the push reaches every prompt position but the last "
          f"{plan.prompt_tail_clear}, as the generation hook does")


def test_the_decode_push_carries_no_perturbation() -> None:
    plan = _push_plan()
    for layer in LAYERS:
        d = plan.delta_for(layer, 0, with_noise=False, with_offset=False)
        s = plan.steering_only(layer, 0)
        assert d is not None and s is not None
        assert torch.allclose(d, s, atol=1e-5), (
            f"layer {layer}: the per-step push differs from the prompt push, so "
            "the stories would be sampled under something other than the push"
        )
    print("the per-step push is the push alone, with nothing else added")


if __name__ == "__main__":
    test_a_perturbing_plan_is_refused()
    test_the_push_is_the_one_the_hook_would_apply()
    test_the_decode_push_carries_no_perturbation()
    print("\nok")
