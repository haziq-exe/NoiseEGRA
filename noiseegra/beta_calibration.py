"""Choose each steering direction's sign from which way its requirement fails.

Round 1's diagnosis was that steering hurt because nothing it steered was a
requirement that failed: one of the four directions was not a scored requirement
at all and the other three targeted requirements already passing at 98-100%.
Round 2 fixed the first half -- every direction is now a scored requirement --
and hit the second half of the same problem from the other side. `terse` is the
direction "end the sentence and start another one", chosen to serve the rule that
no sentence may run over ten words. The model never breaks that rule. What it
breaks is the four-word floor: it writes about ten sentences of four words each
when six to eight of four to ten are wanted, so steering *toward* terser prose
drove it further into the violation, and the steered arm broke 5.05 requirements
against the baseline's 4.55.

A fixed positive coefficient assumes the model errs on one particular side of
every rule. With two-sided rules that assumption is wrong half the time, and
being wrong costs more than not steering at all.

The rule here is the obvious one, stated so it can be applied rather than
guessed: **steer a direction only if its requirement actually fails, and in the
direction of the side that is failing.** The measurement is taken on unsteered
generations, so nothing about the conditions being compared enters into it.

    +1  the requirement fails, and more of this property fixes it
    -1  the requirement fails on the other side, and less of it fixes it
     0  the requirement already passes; pushing a direction that is not needed
        spends the perturbation budget and moves other constraints for nothing

``PROBES`` maps a direction to the failure side it is measured on. A direction
with no probe keeps its nominal +1, so this can never silently zero out a
direction it does not know about.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Sequence, Tuple


def _rate(rows, predicate: Callable) -> float:
    return sum(1 for r in rows if predicate(r)) / max(len(rows), 1)


def failure_sides(rows, checker) -> Dict[str, Tuple[float, float]]:
    """(too-little rate, too-much rate) per steerable direction.

    "Too little" is the share of stories that would be fixed by more of the
    property the direction adds; "too much" the share that would be fixed by
    less. A one-sided rule has a zero in the second slot.
    """
    lo_s, hi_s = checker.sentence_word_range
    return {
        "present_tense": (
            _rate(rows, lambda r: r.checks.get("present_tense") is False), 0.0),
        "simple_register": (
            _rate(rows, lambda r: r.grade_level > checker.max_grade_level), 0.0),
        "dialogue": (
            _rate(rows, lambda r: r.n_quoted_spans < checker.n_quotes),
            _rate(rows, lambda r: r.n_quoted_spans > checker.n_quotes)),
        # The direction shortens sentences, so a story with a sentence over the
        # ceiling wants more of it and one under the floor wants less.
        "terse": (
            _rate(rows, lambda r: r.max_sentence_words > hi_s),
            _rate(rows, lambda r: r.min_sentence_words < lo_s)),
        "varied_openers": (
            _rate(rows, lambda r: r.max_opener_uses > checker.max_opener_uses), 0.0),
    }


def calibrate(
    stories: Sequence[str],
    checker,
    names: Sequence[str],
    beta: float = 1.0,
    *,
    tolerance: float = 0.05,
) -> Tuple[Dict[str, float], List[str]]:
    """Per-direction coefficients, plus lines explaining every one of them.

    ``tolerance`` is the pass-rate slack below which a requirement counts as not
    failing: a rule broken by one story in forty is not worth spending a steering
    direction on, and signing off a difference that small would make the
    coefficient a coin flip between runs.
    """
    rows = checker.evaluate_all(list(stories))["stories"]
    sides = failure_sides(rows, checker)

    betas: Dict[str, float] = {}
    notes: List[str] = []
    for name in names:
        if name not in sides:
            betas[name] = float(beta)
            notes.append(f"  {name:<16} no probe, kept at beta {beta:+.2g}")
            continue
        need_more, need_less = sides[name]
        if max(need_more, need_less) <= tolerance:
            betas[name] = 0.0
            notes.append(f"  {name:<16} passes ({need_more:.0%} / {need_less:.0%} "
                         f"either side), not steered")
        elif need_more >= need_less:
            betas[name] = float(beta)
            notes.append(f"  {name:<16} {need_more:.0%} of stories want more of it, "
                         f"beta {beta:+.2g}")
        else:
            betas[name] = -float(beta)
            notes.append(f"  {name:<16} {need_less:.0%} of stories want less of it, "
                         f"beta {-beta:+.2g}")
    return betas, notes


def describe(betas: Mapping[str, float]) -> str:
    return ", ".join(f"{n}={b:+.2g}" for n, b in betas.items())
