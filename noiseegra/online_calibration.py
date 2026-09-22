"""Sizing the per-story noise while the story is written.

``fisher_calibration`` sizes each story's noise before the story starts: a
32-token reference passage per prompt, then about 13 forward passes of bisection
per story. This does the same job during generation, with no passes of its own.

The size is set against top-p sampling at high temperature, the decoder the
method is compared with, and both quantities that comparison needs are available
at every step:

- How far that decoder would move this step's next-token distribution. Its
  distribution is a function of the model's own (sharpened or flattened by the
  temperature, then cut to the top-p nucleus), so the distance is exact and costs
  one sort over the vocabulary.
- How far the noise has moved it. A second row in the batch writes the same
  words with the same steering and no noise -- the shadow row, used here only to
  measure -- so its distribution is what this step would have been without the
  noise. The Fisher-Rao distance between the two rows is the noise's effect,
  including everything it did at earlier positions through the cache.

After every step a controller scales the noise for the next token so that,
averaged over the recent steps of the story, the second is ``target`` times the
first. This is the adaptive parameter-noise scaling of Plappert et al. (ICLR
2018), which matches a perturbation's output divergence to the per-step
baseline's, run inside one generation instead of across training, and against
the high-temperature top-p decoder instead of epsilon-greedy.

The prompt's share of the noise is written by the first forward pass, before
anything has been measured, so it stays at the starting length times the prompt's
gain. Only the noise while writing follows the controller.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
from transformers import LogitsProcessor

from .fisher import fisher_rao_distance


def sampler_probs(logits: torch.Tensor, temperature: float,
                  top_p: Optional[float]) -> torch.Tensor:
    """The distribution temperature-then-top-p sampling draws from, per row.

    The same cut as :func:`noiseegra.fisher_calibration._probs`: a token is kept
    while the mass before it is still short of ``top_p``.
    """
    p = torch.softmax(logits.float() / float(temperature), dim=-1)
    if top_p is not None and top_p < 1.0:
        sp, idx = p.sort(dim=-1, descending=True)
        keep = (sp.cumsum(dim=-1) - sp) < float(top_p)
        p = torch.zeros_like(p).scatter(-1, idx, sp * keep)
        p = p / p.sum(dim=-1, keepdim=True)
    return p


class OnlineSizer(LogitsProcessor):
    """Scales the per-story noise while the story is written.

    Reads row 0 (the story) and row 1 (its noise-free shadow) of every step's
    scores and sets ``plan.online_gain``, the multiple of the starting length the
    noise is written at on the next step. The scores are returned unchanged, so
    sampling is exactly what it would be without it.

    ``halflife`` is how many steps the running averages remember; ``rate`` damps
    each correction (1 would jump straight to the size the last averages ask
    for); ``max_step`` caps the change in one step; ``bounds`` keeps the gain
    inside a range around the starting length, so a stretch the noise cannot
    move -- a name being spelt out, say -- does not wind it up without limit.
    """

    def __init__(self, plan, target: float, *, temperature: float = 1.8,
                 top_p: Optional[float] = 0.95, halflife: float = 16.0,
                 warmup: int = 2, rate: float = 0.5, max_step: float = 1.25,
                 bounds: Tuple[float, float] = (0.25, 2.5)):
        self.plan = plan
        self.target = float(target)
        self.temperature = float(temperature)
        self.top_p = top_p
        self.keep = 0.5 ** (1.0 / max(float(halflife), 1e-6))
        self.warmup = int(warmup)
        self.rate = float(rate)
        self.max_step = float(max_step)
        self.lo, self.hi = float(bounds[0]), float(bounds[1])
        self.moved: Optional[float] = None    # recent average of the noise's effect
        self.budget: Optional[float] = None   # recent average of the decoder's
        # (noise's effect, decoder's, gain set for the next step), one per step
        self.history: List[Tuple[float, float, float]] = []
        plan.online_gain = 1.0

    def __call__(self, input_ids, scores):
        if scores.dim() != 2 or scores.shape[0] < 2:
            return scores
        with torch.no_grad():
            story = torch.softmax(scores[0].float(), dim=-1)
            clean = torch.softmax(scores[1].float(), dim=-1)
            moved = float(fisher_rao_distance(clean, story))
            decoder = sampler_probs(scores[1:2], self.temperature, self.top_p)[0]
            budget = float(fisher_rao_distance(clean, decoder))
        if self.moved is None:
            self.moved, self.budget = moved, budget
        else:
            self.moved = self.keep * self.moved + (1.0 - self.keep) * moved
            self.budget = self.keep * self.budget + (1.0 - self.keep) * budget
        gain = float(self.plan.online_gain)
        if len(self.history) + 1 > self.warmup and self.moved > 1e-12:
            ratio = self.target * self.budget / self.moved
            step = min(max(ratio ** self.rate, 1.0 / self.max_step), self.max_step)
            gain = min(max(gain * step, self.lo), self.hi)
            self.plan.online_gain = gain
        self.history.append((moved, budget, gain))
        return scores

    def summary(self) -> Dict[str, float]:
        """What the story got: the size it reached, and how far it moved things.

        ``achieved`` is the noise's effect over the whole story in the decoder's
        units, the number the Fisher calibration aims at before the story.
        ``late_gain`` averages the second half, after the controller settles.
        """
        h = self.history
        if not h:
            return {}
        n = len(h)
        moved = sum(x[0] for x in h) / n
        budget = sum(x[1] for x in h) / n
        gains = [x[2] for x in h]
        late = gains[n // 2:] or gains
        return {
            "steps": float(n),
            "achieved": moved / budget if budget > 0 else math.nan,
            "moved": moved,
            "budget": budget,
            "final_gain": gains[-1],
            "late_gain": sum(late) / len(late),
            "min_gain": min(gains),
            "max_gain": max(gains),
        }
