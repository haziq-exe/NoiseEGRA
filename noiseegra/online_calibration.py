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


class RuleTilt(LogitsProcessor):
    """Tilts the story's next-token distribution toward the tokens the rules favour.

    ``profile`` is a vector over the vocabulary: the constraints' output
    profiles (``SteeringVectorSet.output_profile``), read off the same forward
    passes the steering directions come from -- which tokens the model predicts
    more in the rule-following continuations of the contrast pairs than in the
    rule-breaking ones. Each step the story's scores become ``scores + beta *
    profile``, an exponential tilt, with ``beta`` chosen so that the tilt moves
    the distribution a Fisher-Rao distance of ``share`` times what top-p
    sampling at high temperature would move it at the same step. So the tilt is
    sized in the noise's units, spent where the next token is open (top-p moves
    a confident step very little) and left alone where it is not.

    To first order the tilt's distance is ``beta`` times the standard deviation
    of the profile under the current distribution, which gives ``beta``
    directly; a few secant or bisection steps on the exact distance then correct
    it, since a rare token the profile favours can make the first-order guess far
    too large. Only row 0 (the story) is tilted: a noise-free shadow row, when
    there is one, is where the decoder's distortion is measured.
    """

    def __init__(self, profile: torch.Tensor, share: float, *, temperature: float = 1.8,
                 top_p: Optional[float] = 0.95, max_beta: float = 20.0,
                 tolerance: float = 0.03, iters: int = 10):
        self.profile = profile.detach().float()
        self.share = float(share)
        self.temperature = float(temperature)
        self.top_p = top_p
        self.max_beta = float(max_beta)
        self.tolerance = float(tolerance)
        self.iters = int(iters)
        # (beta, tilt's distance, decoder's distance), one per step
        self.history: List[Tuple[float, float, float]] = []

    def _distance(self, logits: torch.Tensor, p: torch.Tensor, beta: float) -> float:
        q = torch.softmax(logits + beta * self.profile, dim=-1)
        return float(fisher_rao_distance(p, q))

    def _solve(self, logits, p, target: float, beta: float):
        """The tilt strength whose distance is ``target``: bracket, then close in.

        The distance rises with the strength, faster than linearly where the
        profile favours tokens the model gives little chance, so scaling the
        first guess by target / distance overshoots and zig-zags. Instead the
        guess is doubled or halved until the target lies between two strengths,
        and false position (Illinois variant) closes the bracket.
        """
        d = self._distance(logits, p, beta)
        if abs(d - target) <= self.tolerance * target:
            return beta, d
        lo, dlo, hi, dhi = 0.0, 0.0, None, None
        for _ in range(12):                          # find a bracket
            if d < target:
                lo, dlo = beta, d
                if hi is not None or beta >= self.max_beta:
                    break
                beta = min(beta * 2.0, self.max_beta)
            else:
                hi, dhi = beta, d
                if lo > 0 or beta < 1e-6:
                    break
                beta = beta / 2.0
            d = self._distance(logits, p, beta)
            if abs(d - target) <= self.tolerance * target:
                return beta, d
        if hi is None:                               # even the limit falls short
            return lo, dlo
        side = 0
        for _ in range(self.iters):                  # close it
            beta = lo + (target - dlo) * (hi - lo) / max(dhi - dlo, 1e-12)
            d = self._distance(logits, p, beta)
            if abs(d - target) <= self.tolerance * target:
                break
            if d < target:
                lo, dlo = beta, d
                if side == -1:
                    dhi = target + (dhi - target) / 2.0
                side = -1
            else:
                hi, dhi = beta, d
                if side == 1:
                    dlo = target - (target - dlo) / 2.0
                side = 1
        return beta, d

    def __call__(self, input_ids, scores):
        if self.share <= 0 or scores.dim() != 2:
            return scores
        with torch.no_grad():
            if self.profile.device != scores.device:
                self.profile = self.profile.to(scores.device)
            ref = scores[1:2] if scores.shape[0] >= 2 else scores[0:1]
            clean = torch.softmax(ref[0].float(), dim=-1)
            decoder = sampler_probs(ref, self.temperature, self.top_p)[0]
            budget = float(fisher_rao_distance(clean, decoder))
            target = self.share * budget
            logits = scores[0].float()
            if target <= 1e-9:
                self.history.append((0.0, 0.0, budget))
                return scores
            p = torch.softmax(logits, dim=-1)
            mean = float((p * self.profile).sum())
            std = float((p * (self.profile - mean) ** 2).sum().clamp_min(0).sqrt())
            beta, d = self._solve(logits, p, target, min(target / max(std, 1e-9), self.max_beta))
            scores[0] = scores[0] + (beta * self.profile).to(scores.dtype)
        self.history.append((beta, d, budget))
        return scores

    def summary(self) -> Dict[str, float]:
        """How hard the story was tilted, and how far that moved its predictions.

        ``achieved`` is the tilt's distance over the story as a share of the
        decoder's, the number ``share`` asks for.
        """
        h = self.history
        if not h:
            return {}
        n = len(h)
        moved = sum(x[1] for x in h) / n
        budget = sum(x[2] for x in h) / n
        return {"steps": float(n), "mean_beta": sum(x[0] for x in h) / n,
                "achieved": moved / budget if budget > 0 else math.nan,
                "moved": moved, "budget": budget}


# --------------------------------------------------------------------------- #
#  Choosing the target before the stories                                      #
# --------------------------------------------------------------------------- #

def target_from_top_share(unit: float, top_prob: float, k: float = 0.43) -> float:
    """The target, in top-p's units, that caps the noise's per-step shift.

    A Fisher-Rao distance d between two next-token distributions bounds the
    probability mass that can move between them: their Bhattacharyya coefficient
    is cos(d/2), and total variation is at most sin(d/2). Coherence needs the
    noise to vary how the model says things without overturning what it would
    say, so the mass it may move is capped at a share ``k`` of the probability
    the model gives its most likely next word, ``top_prob``: sin(d/2) = k *
    top_prob. A less certain model tolerates less.

    The cap is then expressed in the unit the controller uses -- top-p at 1.8's
    own shift per step, ``unit`` -- which is larger on a flatter model: the
    relative unit alone over-noises exactly the models that tolerate least.

    k = 0.43 sits within 3% of the best target found on both models tried:
    Qwen3-1.7B (unit 0.685, top word 0.762: best 1.0, rule 0.98) and
    Llama-3.2-3B (unit 1.345, top word 0.678: best 0.43, rule 0.44).
    """
    s = min(1.0, float(k) * float(top_prob))
    return 2.0 * math.asin(s) / float(unit)


def measure_for_rule(egra, plan, prompt_ids, n_tokens: int = 48) -> Dict[str, float]:
    """Top-p's shift per step and the top word's probability, before any story.

    Along the model's greedy continuation of the prompt under the rule steering
    alone -- the calibration's reference passage -- so nothing is sampled; one
    pass per prompt.
    """
    from .fisher_calibration import _logits, _probs, nucleus_unit, reference_passage

    n_prompt = int(prompt_ids.shape[-1])
    passage = reference_passage(egra, plan, prompt_ids, n_tokens)
    clean = _probs(_logits(egra, plan, passage, n_prompt, with_offset=False), n_prompt)
    return {"unit": nucleus_unit(egra, plan, passage, n_prompt),
            "top_prob": float(clean.max(-1).values.mean())}
