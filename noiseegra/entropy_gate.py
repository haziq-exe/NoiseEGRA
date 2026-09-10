"""Apply the perturbation only where the model actually has a choice.

A decode step where the next-token distribution is nearly a point mass is a step
where the model is finishing a word, closing a quotation, or agreeing a verb.
Perturbing there buys no variety and costs grammar. A step where the distribution
is spread out is a branch point: which noun, which turn the plot takes. That is
where variety lives.

The gate therefore lets the perturbation through only at steps whose next-token
entropy is above a threshold, and leaves the steering running at every step, so a
gate sweep changes *where* the perturbation lands without changing the constraint
pressure.

Two practical points.

The entropy of the step being generated is not known until after the forward pass
that the hook is inside, so the gate uses the entropy of the step before -- "was
the model uncertain about the token it just emitted". That is causal, free, and
in a text where uncertainty is locally correlated it is a good proxy.

The threshold is a quantile of the model's own entropy distribution, measured
once from a short unsteered sample, for the same reason the noise scale is a
multiple of the model's own block RMS: absolute nats are not comparable across
models or tokenisers.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch
from transformers import LogitsProcessor

# Fraction of decode steps the perturbation is *blocked* at. "none" perturbs
# everywhere, which is the behaviour of every run so far.
GATE_LEVELS: Dict[str, Optional[float]] = {
    "none": None,   # every step
    "median": 0.5,  # the more uncertain half of steps
    "high": 0.9,    # only the most uncertain tenth
}


class EntropyProbe(LogitsProcessor):
    """Records each decode step's next-token entropy into a shared dict.

    Registered as a logits *processor*, not a warper, so it sees the model's raw
    scores rather than the post-temperature, post-top-p distribution: the question
    is how uncertain the model is, not how uncertain the sampler was made.
    Returns the scores untouched.
    """

    def __init__(self, state: dict, keep_history: bool = False):
        self.state = state
        self.keep_history = keep_history
        state.setdefault("entropy", float("inf"))
        state.setdefault("history", [])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        logp = torch.log_softmax(scores.float(), dim=-1)
        h = float(-(logp.exp() * logp).sum(-1).mean())
        if math.isfinite(h):
            self.state["entropy"] = h
            if self.keep_history:
                self.state["history"].append(h)
        return scores


def quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile, with no numpy dependency at import time."""
    vals = sorted(v for v in values if v == v)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def collect_decode_entropies(
    egra,
    prompt,
    *,
    n_samples: int = 6,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    seed: int = 0,
) -> List[float]:
    """Next-token entropies over a few unsteered generations from ``prompt``.

    Unsteered on purpose: the threshold describes the model, so it must not depend
    on which condition is being run, or the gate would mean something different in
    every arm.
    """
    from transformers import LogitsProcessorList

    state: dict = {"entropy": float("inf"), "history": []}
    probe = EntropyProbe(state, keep_history=True)
    chat_text = egra.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    device = egra._input_device()
    inputs = egra.tokenizer(chat_text, return_tensors="pt").to(device)
    inputs.pop("token_type_ids", None)

    for i in range(n_samples):
        torch.manual_seed(seed + i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + i)
        with torch.no_grad():
            egra.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                logits_processor=LogitsProcessorList([probe]),
                **egra._sampling_kwargs(do_sample=True, temperature=temperature),
            )
    return list(state["history"])


def gate_threshold(entropies: Sequence[float], level: str) -> float:
    """Entropy in nats above which the perturbation is allowed through."""
    if level not in GATE_LEVELS:
        raise ValueError(f"unknown gate level {level!r}; have {sorted(GATE_LEVELS)}")
    q = GATE_LEVELS[level]
    if q is None:
        return 0.0
    return quantile(entropies, q)


def describe(level: str, threshold: float) -> str:
    q = GATE_LEVELS.get(level)
    if q is None:
        return "no entropy gate: the perturbation is applied at every decode step"
    return (f"entropy gate '{level}': the perturbation is applied only where the "
            f"model's next-token entropy exceeds {threshold:.3f} nats, which is the "
            f"top {1 - q:.0%} of its own decode steps")
