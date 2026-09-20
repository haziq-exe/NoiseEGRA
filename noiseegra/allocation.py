"""Dividing a fixed steering budget among the requirements by measured need.

Steering every requirement at once runs into a fixed total: the push is
renormalised so that adding a direction redistributes strength rather than
pushing harder. Under that constraint, which requirements get a direction is a
real decision, and until now it was made by hand -- five directions for twelve
requirements, chosen by judgement.

Measuring what the model actually gets wrong shows the hand-picked set was the
wrong set. On Qwen3-1.7B, across 200 stories under the middle-school
instruction, the picked set included the one requirement that never fails
(writing only the story: 100%) and left out the second worst (varied sentence
openings: 30%). Budget was being spent where nothing was wrong and withheld
where something was.

It also shows why a picked set cannot carry to another model. The same twelve
requirements on Qwen3-8B fail in a different pattern -- varied openings 99%
rather than 30%, dialogue 97% rather than 62% -- and the two shortfall profiles
point only 0.79 of the way in the same direction. A set chosen on one model
spends its budget on the other model's solved problems.

So the weights are read off the model's own failures on a calibration sample:
each direction is weighted by how often its requirement is broken, and the
plan's budget machinery renormalises the total as before. A requirement the
model already satisfies asks for nothing and costs nothing. This replaces a
judgement with a measurement, and it re-measures on whatever model it is run on.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Optional


def shortfall_weights(
    pass_rates: Mapping[str, float],
    names: Iterable[str],
    *,
    floor: float = 0.0,
    power: float = 1.0,
    aliases: Optional[Mapping[str, str]] = None,
) -> Dict[str, float]:
    """Weight each steered direction by how often its requirement is broken.

    `pass_rates` maps a requirement to the share of calibration stories that
    satisfied it. `names` are the directions to be steered, which need not be
    named exactly as the requirements are -- `aliases` maps a direction to the
    requirement it serves.

    `floor` keeps a direction alive even where its requirement always passes,
    which is worth having when the calibration sample is small enough that a
    100% pass rate is weak evidence. `power` sharpens or flattens the
    allocation: above one concentrates the budget on the worst requirements,
    below one spreads it.

    The weights are scaled so the largest is one. The absolute size is set by
    the plan's own budget, so this function decides proportions only.
    """
    if floor < 0:
        raise ValueError("floor must not be negative")
    if power <= 0:
        raise ValueError("power must be positive")
    names = list(names)
    if not names:
        raise ValueError("no directions to weight")
    alias = dict(aliases or {})

    weights: Dict[str, float] = {}
    for n in names:
        rule = alias.get(n, n)
        if rule in pass_rates:
            need = max(0.0, 1.0 - float(pass_rates[rule]))
        else:
            # A direction with no requirement of its own -- a shield, or one
            # added for a property nothing scores -- cannot be allocated by
            # measurement, so it keeps a neutral share rather than being
            # silently switched off.
            need = None
        weights[n] = need

    measured = [v for v in weights.values() if v is not None]
    typical = (sum(measured) / len(measured)) if measured else 1.0
    for n, v in list(weights.items()):
        weights[n] = typical if v is None else v

    out = {n: (max(0.0, weights[n]) ** power) + floor for n in names}
    top = max(out.values())
    if top <= 0:
        # Nothing is failing. Steering all of them equally is the honest
        # fallback: the measurement says the push has no work to do.
        return {n: 1.0 for n in names}
    return {n: v / top for n, v in out.items()}


def allocation_cosine(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    """How far two allocations point in the same direction, 0 to 1.

    Used to say whether a set of weights measured on one model means anything
    on another. One means the two models fail in the same pattern and a picked
    set could carry over; well below one means it cannot.
    """
    keys = sorted(set(a) | set(b))
    va = [float(a.get(k, 0.0)) for k in keys]
    vb = [float(b.get(k, 0.0)) for k in keys]
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(x * x for x in vb))
    if na <= 0 or nb <= 0:
        return 0.0
    return sum(x * y for x, y in zip(va, vb)) / (na * nb)
