"""How much of the noise ends up along the rule directions.

The per-story noise is projected clear of the rule directions at every layer it
is added to, so where it is added it has no component along them at all. That
is protection at the point of injection, not downstream: noise added at layer 6
passes through layers 7 to 13, whose attention and MLPs are nonlinear, and what
they write back can lie along the rule directions of the layer that follows.

This measures that directly. Along the reference passage, it compares each
steered layer's output with the noise and without, and projects the difference
onto that layer's protected directions. Because the layer's own noise is
orthogonal to them, whatever is found there came from the noise added earlier.
It is reported three ways: as a share of the whole difference, against the
share a random vector would have by chance, and against the size of the push
the steering applies along the same directions -- the number that says whether
the noise can overpower a rule.
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch

from .fisher_calibration import _logits


@torch.no_grad()
def rule_leakage(egra, plan, passage: torch.Tensor, n_prompt: int,
                 names: List[str]) -> Dict[int, Dict[str, object]]:
    """Per steered layer: where the noise's downstream effect lands.

    Returns, for each layer, ``share`` (fraction of the change lying in the
    protected directions), ``chance`` (that fraction for a random vector),
    ``vs_push`` (size of the change along the protected directions over the size
    of the steering push, averaged over the continuation), and ``per_rule``
    (the change along each rule direction over the push along it).
    """
    with_n: Dict[int, torch.Tensor] = {}
    without: Dict[int, torch.Tensor] = {}
    _logits(egra, plan, passage, n_prompt, with_offset=True, capture=with_n)
    _logits(egra, plan, passage, n_prompt, with_offset=False, capture=without)
    out: Dict[int, Dict[str, object]] = {}
    n = int(passage.shape[-1])
    for li in sorted(with_n):
        lp = plan.layer_plans[li]
        dh = (with_n[li] - without[li])[n_prompt:n]           # (T, dim)
        prot = lp.protect.to(dh.device, torch.float32)
        rules = lp.basis.to(dh.device, torch.float32)
        saved = getattr(plan, "_jitter_step", None)
        rows = []
        for j in range(n - n_prompt):
            d = plan.delta_for(li, j, with_noise=False, with_offset=False,
                               device=dh.device)
            rows.append(torch.zeros(dh.shape[1], device=dh.device) if d is None
                        else d.float())
        if saved is not None:
            plan._jitter_step = saved
        push = torch.stack(rows)                              # (T, dim)
        in_prot = dh @ prot                                   # (T, k)
        share = float((in_prot.norm(dim=1) / dh.norm(dim=1).clamp_min(1e-12)).mean())
        push_prot = (push @ prot).norm(dim=1)
        vs_push = float((in_prot.norm(dim=1) / push_prot.clamp_min(1e-12)).mean())
        along = (dh @ rules).abs().mean(dim=0)                # (C,)
        pushed = (push @ rules).abs().mean(dim=0).clamp_min(1e-12)
        per_rule = {names[i] if i < len(names) else f"rule{i}": float(along[i] / pushed[i])
                    for i in range(rules.shape[1])}
        out[li] = {"share": share, "chance": math.sqrt(prot.shape[1] / dh.shape[1]),
                   "vs_push": vs_push, "per_rule": per_rule}
    return out
