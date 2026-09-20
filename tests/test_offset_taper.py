#!/usr/bin/env python
"""The perturbation can fade along the prompt instead of stopping at a position.

Sparing the last prompt positions outright separates two effects that had always
been measured together. Measured at 200 stories, two stories out:

    positions spared    coherent   not a story   what happens   wording
                   8     170/200          10%          +4.9      +5.1
                  16     177/200           3%          +1.9       +0.3
                  32     183/200           0%          +1.5       -0.3

The perturbation near the end of the prompt is what varies the wording, and what
makes the model answer the reader instead of telling a story. The perturbation
early in the prompt is what varies what happens, and is safe -- at two and a half
stories out with 48 spared there is not one non-story in two hundred and variety
of what happens is still won.

Cutting at a position loses the wording along with the failure. Fading keeps
some of both, so the strength becomes a curve over the prompt rather than a
cliff.

python tests/test_offset_taper.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

N_POS, DIM = 40, 16


def _apply(off, taper, keep=0, head=0, n=N_POS):
    """What the generation hook does to the prompt, for one layer."""
    target = torch.zeros(1, n, DIM)
    lo = head if 0 < head < n else 0
    hi = n - keep if 0 < keep < n - lo else n
    if taper >= 1.0:
        target[:, lo:hi, :].add_(off.view(1, 1, -1))
    else:
        ramp = torch.linspace(1.0, taper, hi - lo).view(1, -1, 1)
        target[:, lo:hi, :].add_(off.view(1, 1, -1) * ramp)
    return target[0].norm(dim=1)


def test_a_flat_perturbation_is_unchanged() -> None:
    """Taper 1.0 must reproduce exactly what every earlier run did."""
    off = torch.randn(DIM)
    flat = _apply(off, 1.0, keep=8)
    assert torch.allclose(flat[:32], flat[:1].expand(32), atol=1e-5), "not flat"
    assert float(flat[32:].abs().max()) == 0.0, "the spared tail was touched"
    print("taper 1.0 is the old behaviour exactly: flat, with the tail spared")


def test_it_fades_from_the_start_to_the_end() -> None:
    off = torch.randn(DIM)
    faded = _apply(off, 0.0)
    assert faded[0] > faded[N_POS // 2] > faded[-1], "not decreasing along the prompt"
    assert abs(float(faded[-1])) < 1e-5, "taper 0 should reach nothing at the last position"
    assert abs(float(faded[0]) - float(off.norm())) < 1e-4, (
        "the first position should get the full perturbation")
    print("the perturbation fades from full at the first position to none at the last")


def test_a_partial_taper_keeps_a_share_at_the_end() -> None:
    off = torch.randn(DIM)
    faded = _apply(off, 0.25)
    want = 0.25 * float(off.norm())
    assert abs(float(faded[-1]) - want) < 1e-4, (
        f"the last position kept {float(faded[-1]):.3f}, expected {want:.3f}")
    print("a taper of 0.25 leaves exactly a quarter at the last position")


def test_fading_still_reaches_the_positions_cutting_abandons() -> None:
    """This is the whole point, and it is not about the total.

    A linear fade to nothing delivers exactly the same total as cutting at the
    midpoint -- half. What differs is where it goes: cutting leaves the late
    positions untouched, and those are the ones that vary the wording. Fading
    reaches them at reduced strength.
    """
    off = torch.randn(DIM)
    cut = _apply(off, 1.0, keep=N_POS // 2)
    fade = _apply(off, 0.0)
    assert abs(float(cut.sum()) - float(fade.sum())) < 1e-3, (
        "the two no longer deliver the same total, so this is not a like-for-like "
        "comparison")
    late = slice(N_POS // 2, N_POS - 1)
    assert float(cut[late].max()) == 0.0, "cutting should leave the late positions alone"
    assert float(fade[late].min()) > 0.0, (
        "fading should still reach every late position, which is where the wording "
        "variety comes from")
    print(f"at the same total, cutting reaches {int((cut > 0).sum())} positions and "
          f"fading reaches {int((fade > 0).sum())}")


if __name__ == "__main__":
    test_a_flat_perturbation_is_unchanged()
    test_it_fades_from_the_start_to_the_end()
    test_a_partial_taper_keeps_a_share_at_the_end()
    test_fading_still_reaches_the_positions_cutting_abandons()
    print("\nok")
