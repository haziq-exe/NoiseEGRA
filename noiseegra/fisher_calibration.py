"""Sizing a random perturbation by how far it moves the model's predictions.

The perturbation this project adds is random: a direction drawn fresh for every
story, in a random subspace that is also drawn fresh, and nothing about it is
learned from the model's outputs. What is left to choose is its size, and every
unit this project has used for that has been wrong in its own way.

A fixed activation length means different things in different directions. The
pullback of the output's Fisher metric onto the residual stream is extremely
uneven -- its spectrum spans seven orders of magnitude on GPT-2, and only 2-17%
of directions move the output at all (FishBack, arXiv 2605.17231) -- so two random
draws of the same length can differ by orders of magnitude in what they do to
the story. One barely registers; the next lands on a direction the model is
sensitive to and breaks the text. Measuring the size against a story's distance
from the average story fixed the scale only by sampling stories first, which the
method must not do.

The unit here is the change in the model's own predictions. For a categorical
distribution the Fisher-Rao distance has a closed form, the angle between the
square roots of the two probability vectors, so the effect of a perturbation of
any length can be measured exactly with one forward pass. The length is then
chosen per story, by bisection, so that the perturbation moves the model's
next-token distribution by a set amount averaged over a short reference passage.
Every story is changed by the same amount of information, whichever direction
its draw happened to point in.

The set amount is expressed against the baseline this method has to beat. The
reference unit is the Fisher-Rao distance between the model's distribution at
temperature 1 and the distribution nucleus sampling actually draws from
(temperature 1.8, top-p 0.95), averaged over the same passage. A size of 1.0 is
a perturbation that moves the predictions as far as high-temperature nucleus
sampling does -- the difference being that nucleus sampling spends that change
independently at every token, flattening each distribution, while this spends it
on one coherent direction per story.

The reference passage is the model's own greedy continuation under the rule
steering alone, computed once per prompt. It is not a sampled story and nothing
is learned from it: it is the fixed text along which the perturbation's effect
is measured, the way a ruler is laid along something to measure it.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch

from .fisher import fisher_rao_distance


def _prompt_range(plan, n: int) -> tuple:
    keep = int(getattr(plan, "prompt_tail_clear", 0) or 0)
    head = int(getattr(plan, "prompt_head_clear", 0) or 0)
    lo = head if 0 < head < n else 0
    hi = n - keep if 0 < keep < n - lo else n
    return lo, hi


def _layers(egra, plan) -> List[int]:
    blocks = egra._get_transformer_blocks()
    return sorted({egra._normalize_layer_index(int(i), len(blocks))
                   for i in plan.layers})


@torch.no_grad()
def _logits(egra, plan, ids: torch.Tensor, n_prompt: int, *,
            with_offset: bool) -> torch.Tensor:
    """Logits at every position of ``ids`` with the plan applied as generation
    applies it: the push (and, if asked, the offset) across the prompt, and the
    decode-step delta at each continuation position.

    One full forward pass without a cache. The continuation positions are
    processed together rather than one step at a time, which gives the same
    result: each position's delta depends only on its own step number.
    """
    blocks = egra._get_transformer_blocks()
    layers = _layers(egra, plan)
    n = int(ids.shape[-1])
    lo, hi = _prompt_range(plan, n_prompt)
    bands = getattr(plan, "offset_layers", None) or ()
    taper = float(getattr(plan, "offset_taper", 1.0) or 1.0)
    offset_prefill = bool(getattr(plan, "offset_prefill", False))

    def make_hook(li):
        def hook(module, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if not isinstance(t, torch.Tensor) or t.dim() != 3:
                return None
            if plan.steer_prefill and hi > lo:
                d = plan.steering_only(li, 0, device=t.device)
                if d is not None:
                    t[:, lo:hi, :].add_(d.to(t.dtype).view(1, 1, -1))
            if with_offset and offset_prefill and (not bands or li in bands) and hi > lo:
                off = plan.layer_plans[li].offset
                if off is not None:
                    off = off.to(device=t.device, dtype=t.dtype).view(1, 1, -1)
                    if taper >= 1.0:
                        t[:, lo:hi, :].add_(off)
                    else:
                        ramp = torch.linspace(1.0, taper, hi - lo, device=t.device,
                                              dtype=t.dtype).view(1, -1, 1)
                        t[:, lo:hi, :].add_(off * ramp)
            for j in range(n - n_prompt):
                d = plan.delta_for(li, j, with_noise=False, with_offset=with_offset,
                                   device=t.device)
                if d is not None:
                    t[:, n_prompt + j, :].add_(d.to(t.dtype))
            return None
        return hook

    saved_step = getattr(plan, "_jitter_step", None)
    handles = [blocks[li].register_forward_hook(make_hook(li)) for li in layers]
    try:
        res = egra.model(input_ids=ids, use_cache=False, return_dict=True)
    finally:
        for h in handles:
            h.remove()
        if saved_step is not None:
            plan._jitter_step = saved_step
    return res.logits[0].float()


@torch.no_grad()
def reference_passage(egra, plan, prompt_ids: torch.Tensor,
                      n_tokens: int = 32) -> torch.Tensor:
    """The model's greedy continuation under the rule steering alone.

    Deterministic and computed once per prompt. It is the text along which a
    perturbation's effect is measured, not a sample of anything.
    """
    ids = prompt_ids
    n_prompt = int(prompt_ids.shape[-1])
    for _ in range(int(n_tokens)):
        logits = _logits(egra, plan, ids, n_prompt, with_offset=False)
        nxt = logits[-1].argmax().view(1, 1).to(ids.device)
        ids = torch.cat([ids, nxt], dim=-1)
    return ids


def _probs(logits: torch.Tensor, n_prompt: int, temperature: float = 1.0,
           top_p: Optional[float] = None) -> torch.Tensor:
    """Next-token distributions from the last prompt position onward."""
    x = logits[n_prompt - 1: logits.shape[0] - 1] / float(temperature)
    p = torch.softmax(x, dim=-1)
    if top_p is not None and top_p < 1.0:
        sp, idx = p.sort(dim=-1, descending=True)
        keep = (sp.cumsum(dim=-1) - sp) < float(top_p)
        sp = sp * keep
        p = torch.zeros_like(p).scatter(-1, idx, sp)
        p = p / p.sum(dim=-1, keepdim=True)
    return p


def nucleus_unit(egra, plan, passage: torch.Tensor, n_prompt: int, *,
                 temperature: float = 1.8, top_p: float = 0.95) -> float:
    """How far nucleus sampling moves the distribution it draws from, per token.

    Mean Fisher-Rao distance between the steered model's distribution at
    temperature 1 and the truncated, sharpened-then-flattened distribution
    nucleus sampling at ``temperature``/``top_p`` samples from, along the
    reference passage.
    """
    logits = _logits(egra, plan, passage, n_prompt, with_offset=False)
    p1 = _probs(logits, n_prompt)
    pn = _probs(logits, n_prompt, temperature=temperature, top_p=top_p)
    return float(fisher_rao_distance(p1, pn).mean())


def set_offset_length(plan, length: float) -> None:
    """Give every layer's offset the same length, keeping its direction.

    The direction is kept separately the first time it is seen, so a length of
    zero -- which a search may try -- does not erase it.
    """
    for lp in plan.layer_plans.values():
        if lp.offset is None:
            continue
        unit = getattr(lp, "_offset_unit", None)
        if unit is None or unit.shape != lp.offset.shape:
            unit = lp.offset / lp.offset.norm().clamp_min(1e-12)
            lp._offset_unit = unit
        lp.offset = unit * float(length)
        lp.offset_length = float(length)


def offset_distance(egra, plan, passage: torch.Tensor, n_prompt: int,
                    reference: Optional[torch.Tensor] = None) -> float:
    """Mean Fisher-Rao distance the current offset causes along the passage."""
    if reference is None:
        reference = _probs(_logits(egra, plan, passage, n_prompt, with_offset=False),
                           n_prompt)
    moved = _probs(_logits(egra, plan, passage, n_prompt, with_offset=True), n_prompt)
    return float(fisher_rao_distance(reference, moved).mean())


def calibrate_offset(egra, plan, prompt_ids: torch.Tensor, target: float, *,
                     max_length: float, n_tokens: int = 32, iters: int = 12,
                     cache: Optional[Dict] = None) -> Dict[str, float]:
    """Set the offset's length so it moves the predictions by ``target``.

    ``target`` is in units of :func:`nucleus_unit`. Bisection on a log scale
    between ``max_length / 1000`` and ``max_length``: the distance rises with
    the length, not always smoothly, and a log scale spends the iterations
    evenly across the three orders of magnitude a random draw's sensitivity can
    span. When even ``max_length`` falls short the offset is left at
    ``max_length`` and the shortfall is reported rather than hidden.
    """
    key = tuple(int(i) for i in prompt_ids.view(-1).tolist())
    if cache is not None and key in cache:
        passage, unit, reference = cache[key]
    else:
        passage = reference_passage(egra, plan, prompt_ids, n_tokens)
        n_prompt = int(prompt_ids.shape[-1])
        unit = nucleus_unit(egra, plan, passage, n_prompt)
        reference = _probs(_logits(egra, plan, passage, n_prompt, with_offset=False),
                           n_prompt)
        if cache is not None:
            cache[key] = (passage, unit, reference)
    n_prompt = int(prompt_ids.shape[-1])
    goal = float(target) * unit

    lo, hi = math.log(max_length / 1000.0), math.log(max_length)
    set_offset_length(plan, max_length)
    top = offset_distance(egra, plan, passage, n_prompt, reference)
    if top < goal:
        return {"length": max_length, "distance": top / unit, "unit": unit,
                "reached": 0.0}
    best = (max_length, top)
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        set_offset_length(plan, math.exp(mid))
        d = offset_distance(egra, plan, passage, n_prompt, reference)
        if abs(d - goal) < abs(best[1] - goal):
            best = (math.exp(mid), d)
        if d < goal:
            lo = mid
        else:
            hi = mid
    set_offset_length(plan, best[0])
    return {"length": best[0], "distance": best[1] / unit, "unit": unit,
            "reached": 1.0}
