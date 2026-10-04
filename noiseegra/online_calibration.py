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
                 bounds: Tuple[float, float] = (0.25, 2.5),
                 absolute: Optional[float] = None, hold: int = 0):
        self.plan = plan
        self.target = float(target)
        self.temperature = float(temperature)
        self.top_p = top_p
        self.keep = 0.5 ** (1.0 / max(float(halflife), 1e-6))
        self.warmup = int(warmup)
        self.rate = float(rate)
        self.max_step = float(max_step)
        self.lo, self.hi = float(bounds[0]), float(bounds[1])
        # A per-step Fisher-Rao distance to hold the noise's effect at, instead
        # of ``target`` times the decoder's own shift at each step. The two agree
        # when the text being written moves the decoder as much as the reference
        # passage did (within 8% on stories); on text the model is far surer of
        # than its reference passage (maths), the relative target runs away.
        self.absolute = None if absolute is None else float(absolute)
        # Steps it only watches before it starts: a noise faded in from zero
        # moves things little on the way up, and a controller running then would
        # wind the size up and undo the fade.
        self.hold = int(hold)
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
        mode = str(getattr(self.plan, "offset_envelope", "flat"))
        if mode == "budget" and self.plan.change_end is None:
            # The chance the noise changed this step's word.
            self.plan.change_spent += 0.5 * float((story - clean).abs().sum())
            if self.plan.change_spent >= float(self.plan.offset_change_budget):
                self.plan.change_end = len(self.history) + 1
        if len(self.history) < self.hold:
            self.history.append((moved, budget, float(self.plan.online_gain)))
            return scores
        if self.moved is None:
            self.moved, self.budget = moved, budget
        else:
            self.moved = self.keep * self.moved + (1.0 - self.keep) * moved
            self.budget = self.keep * self.budget + (1.0 - self.keep) * budget
        gain = float(self.plan.online_gain)
        # With a plateau envelope the target falls with it, so the controller
        # follows the fade instead of winding the size up against it; where
        # the envelope is nearly off it holds still.
        env = 1.0
        if mode in ("plateau", "budget"):
            env = float(self.plan.envelope_at(len(self.history)))
        if (len(self.history) - self.hold + 1 > self.warmup and self.moved > 1e-12
                and env > 0.05):
            ratio = ((self.absolute if self.absolute is not None
                      else self.target * self.budget) * env / self.moved)
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
    Qwen3-1.7B (unit 0.685, top word 0.762: best 1.0, rule 0.97) and
    Llama-3.2-3B (unit 1.345, top word 0.678: best 0.43, rule 0.44).
    """
    s = min(1.0, float(k) * float(top_prob))
    return 2.0 * math.asin(s) / float(unit)


def _reference(egra, plan, prompt_ids, n_tokens: int):
    from .fisher_calibration import _logits, _probs, reference_passage

    n_prompt = int(prompt_ids.shape[-1])
    passage = reference_passage(egra, plan, prompt_ids, n_tokens)
    clean = _probs(_logits(egra, plan, passage, n_prompt, with_offset=False), n_prompt)
    return passage, clean


def measure_for_rule(egra, plan, prompt_ids, n_tokens: int = 48,
                     reference=None) -> Dict[str, float]:
    """Top-p's shift per step and the top word's probability, before any story.

    Along the model's greedy continuation of the prompt under the rule steering
    alone -- the calibration's reference passage -- so nothing is sampled; one
    pass per prompt.
    """
    from .fisher_calibration import nucleus_unit

    n_prompt = int(prompt_ids.shape[-1])
    passage, clean = reference or _reference(egra, plan, prompt_ids, n_tokens)
    return {"unit": nucleus_unit(egra, plan, passage, n_prompt),
            "top_prob": float(clean.max(-1).values.mean())}


def scale_offsets(plan, length: float) -> None:
    """Give this story's noise ``length`` at every layer, keeping its direction."""
    for lp in plan.layer_plans.values():
        if lp.offset is None:
            continue
        now = lp.offset_length if lp.offset_length else float(lp.offset.norm())
        lp.offset = lp.offset * (float(length) / max(float(now), 1e-12))
        lp.offset_length = float(length)


def start_for_target(egra, plan, prompt_ids, target: float, unit: float, *,
                     reference=None, n_tokens: int = 48, seeds=(0, 1, 2, 3),
                     iters: int = 10) -> Dict[str, float]:
    """The starting length at which the noise already moves things by ``target``.

    The controller multiplies the starting length by at most ``max_step`` a
    step, and its averages lag the noise's effect, so a start far below the
    size a model needs makes it overshoot over the opening words -- on
    Llama-3.2-3B, at 0.10 of the residual norm against the 0.24 its target
    needed, 17 of 40 openings garbled. Measured once per prompt instead, along
    the same greedy passage as the target: a few random draws of the noise,
    made exactly as a story makes them, at a common length, their mean
    Fisher-Rao shift of the passage's predictions set to ``target * unit`` by
    bisection on a log scale between 1/500 of the residual norm and all of it.
    The same draws at every length, so the shift rises smoothly with it.

    Changes the plan's noise draw and the torch random state, so it runs
    before the story is seeded and draws its own noise.
    """
    from .fisher_calibration import offset_distance

    n_prompt = int(prompt_ids.shape[-1])
    passage, clean = reference or _reference(egra, plan, prompt_ids, n_tokens)
    norm = float(plan.rms_scale) * math.sqrt(plan.dim)
    goal = float(target) * float(unit)

    def moved(length: float) -> float:
        out = []
        for s in seeds:
            torch.manual_seed(1_000_003 + int(s))
            plan.resample_offset(story_index=0)
            scale_offsets(plan, length)
            out.append(offset_distance(egra, plan, passage, n_prompt, clean))
        return sum(out) / len(out)

    lo, hi = math.log(norm / 500.0), math.log(norm)
    top = moved(norm)
    if top < goal:
        return {"start": norm, "fraction": 1.0, "moves": top / unit, "reached": 0.0}
    best = (norm, top)
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        d = moved(math.exp(mid))
        if abs(d - goal) < abs(best[1] - goal):
            best = (math.exp(mid), d)
        lo, hi = (mid, hi) if d < goal else (lo, mid)
    return {"start": best[0], "fraction": best[0] / norm, "moves": best[1] / unit,
            "reached": 1.0}


# --------------------------------------------------------------------------- #
#  Sizing the rule steering by its effect                                      #
# --------------------------------------------------------------------------- #

def _with_budget(plan, budget: float):
    """Context: the plan's steering at ``budget`` (0 turns it off), put back after."""
    class _Ctx:
        def __enter__(self):
            self.saved = plan.steer_budget
            plan.steer_budget = float(budget)
        def __exit__(self, *exc):
            plan.steer_budget = self.saved
    return _Ctx()


def steer_target_from_top_share(top_prob: float, share: float) -> float:
    """The steering's effect target from the unsteered model's top-word probability.

    The same rule as the noise's target (target_from_top_share), with its own
    share: the steering may move at most ``share`` of the probability the
    unsteered model gives its most likely next word, sin(d/2) = share * top_prob.
    A model less sure of its next word is pushed less -- a fixed distance asks
    for twice the budget on a prompt where Qwen3-8B's top word is 0.64 than
    where it is 0.76, and held that long the push breaks the text.
    """
    return 2.0 * math.asin(min(1.0, float(share) * float(top_prob)))


def budget_for_effect(egra, plan, prompt_ids, effect: Optional[float] = None, *,
                      share: Optional[float] = None,
                      n_tokens: int = 48, iters: int = 12) -> Dict[str, float]:
    """The steering budget at which the push moves the predictions by ``effect``.

    A budget is a length in units of the per-coordinate activation scale, so the
    same budget is a smaller push relative to a wider model's residual stream
    (whose norm grows with sqrt(dim)), and models differ in how far a push of a
    given relative size moves their words. Sized by effect instead -- the mean
    Fisher-Rao distance between the steered and unsteered next-word
    distributions -- the push means the same thing on every model, in the same
    currency the noise is sized in.

    Measured once per prompt along the model's greedy continuation with the
    steering off, so nothing is sampled. The distance rises with the budget, so
    the budget is found by bisection on a log scale around the given one. With
    ``effect`` None only the given budget's effect is measured. With ``share``
    the effect is set from the unsteered model's top-word probability instead
    (steer_target_from_top_share).
    """
    from .fisher_calibration import _logits, _probs, reference_passage

    n_prompt = int(prompt_ids.shape[-1])
    given = float(plan.steer_budget or 0.0)
    with _with_budget(plan, 0.0):
        passage = reference_passage(egra, plan, prompt_ids, n_tokens)
        base = _probs(_logits(egra, plan, passage, n_prompt, with_offset=False), n_prompt)

    def moved(budget: float) -> float:
        with _with_budget(plan, budget):
            p = _probs(_logits(egra, plan, passage, n_prompt, with_offset=False), n_prompt)
        return float(fisher_rao_distance(base, p).mean())

    top_prob = float(base.max(-1).values.mean())
    if share is not None and share > 0:
        effect = steer_target_from_top_share(top_prob, share)
    at_given = moved(given) if given > 0 else 0.0
    out = {"given": given, "at_given": at_given, "budget": given, "moves": at_given,
           "reached": 1.0, "top_prob": top_prob,
           "effect": float(effect) if effect is not None else 0.0}
    if effect is None or given <= 0:
        return out
    lo, hi = math.log(given / 20.0), math.log(given * 20.0)
    best = (given, at_given)
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        d = moved(math.exp(mid))
        if abs(d - effect) < abs(best[1] - effect):
            best = (math.exp(mid), d)
        lo, hi = (mid, hi) if d < float(effect) else (lo, mid)
    out.update(budget=best[0], moves=best[1],
               reached=float(abs(best[1] - effect) <= 0.05 * float(effect)))
    return out


def noise_divergence(egra, plan, prompt_ids, passage, length: float, *,
                     seeds=(0, 1, 2, 3)) -> Dict[str, float]:
    """How far random draws of the noise at ``length`` change what is written.

    The per-step shift the noise is sized by says how much each next-word
    distribution moves; this says whether the words chosen change: the greedy
    continuation with each draw against the one without, token by token, over
    the reference passage's length: where each draw first writes a different
    word (median over draws; the passage length if it never does). Only the
    first departure is counted -- after it the two texts are no longer aligned.
    A diagnostic, printed before the stories.
    Changes the plan's noise draw and the torch random state, like
    start_for_target.
    """
    from .fisher_calibration import _logits

    n_prompt = int(prompt_ids.shape[-1])
    n = int(passage.shape[-1]) - n_prompt
    clean = passage[0, n_prompt:].tolist()
    firsts = []
    for s in seeds:
        torch.manual_seed(1_000_003 + int(s))
        plan.resample_offset(story_index=0)
        scale_offsets(plan, length)
        ids = prompt_ids
        for _ in range(n):
            logits = _logits(egra, plan, ids, n_prompt, with_offset=True)
            ids = torch.cat([ids, logits[-1].argmax().view(1, 1).to(ids.device)], dim=-1)
        got = ids[0, n_prompt:].tolist()
        diff = [a != b for a, b in zip(got, clean)]
        firsts.append(diff.index(True) if any(diff) else n)
    firsts.sort()
    return {"first": float(firsts[len(firsts) // 2]),
            "departed": sum(f < n for f in firsts) / len(firsts), "tokens": float(n)}


# --------------------------------------------------------------------------- #
#  The noise's push on each decision, scaled by how sure the clean model is     #
# --------------------------------------------------------------------------- #

def neutral_scale(clean_probs: torch.Tensor, ref: float, *, power: float, lo: float,
                  hi: float, word_start: Optional[torch.Tensor] = None) -> float:
    """Average of the margin scaling along the reference passage, so dividing by
    it leaves the noise's average push where it was: the push moved from unsure
    steps to sure ones, not added. Same rule as MarginScaler.__call__."""
    lp = clean_probs.clamp_min(1e-30).log()
    top = lp.topk(2, dim=-1)
    m = (top.values[:, 0] - top.values[:, 1]).float()
    s = ((m / max(float(ref), 1e-3)) ** float(power)).clamp(float(lo), float(hi))
    if word_start is not None:
        first = top.indices[:, 0].cpu()
        inside = ~word_start[first.clamp_max(word_start.numel() - 1)]
        s = torch.where(inside.to(s.device), s.clamp_max(1.0), s)
    return float(s.mean())


def reference_margin(clean_probs: torch.Tensor) -> float:
    """Median gap between the clean model's top two log-probabilities along the
    reference passage: the typical sureness of a step for this prompt."""
    top = clean_probs.clamp_min(1e-30).log().topk(2, dim=-1).values
    return float((top[:, 0] - top[:, 1]).median())


class MarginScaler(LogitsProcessor):
    """Scales the residual noise's effect on this step's decision by the clean
    model's sureness at this step.

    A vector added to the hidden state shifts the logits by about the same
    amount at every step, but how much that moves the choice depends on the
    gap between the clean model's top two words: where it is unsure, the shift
    reorders words that mean the same; where it is sure -- who the story is
    about, what happens -- the same shift cannot cross the gap. A fixed-size
    noise therefore spends itself on wording and leaves the plot to the model's
    defaults, at any size small enough to keep the wording intact.

    Row 0 (the story) and row 1 (its noise-free shadow) come from the same
    forward pass; their difference is the noise's own push on this decision,
    first-order linear in its size. That push is scaled by (margin / median
    margin)^power, held to [lo, hi]: amplified where the clean model is sure,
    damped where it is not, so the noise has the same chance to change a
    decision wherever it is made. The noise in the hidden states -- what later
    words attend to -- is left as the controller sized it; only how hard it
    pushes each choice is equalised. Placed after the controller, which reads
    the raw push.
    """

    def __init__(self, ref_margin: float, *, power: float = 1.0,
                 lo: float = 0.25, hi: float = 4.0,
                 word_start: Optional[torch.Tensor] = None, norm: float = 1.0,
                 steps: int = 0):
        self.ref = max(float(ref_margin), 1e-3)
        # Only the first `steps` decisions (0 = all): the opening, where the
        # story's setting and premise are chosen. Later decisions mostly keep
        # the story consistent with what is already written, and amplifying
        # the push there breaks that consistency.
        self.steps = int(steps)
        self.calls = 0
        # Divides every step's scale: the average of the scaling on the
        # reference passage (neutral_scale), so the average push is unchanged.
        self.norm = max(float(norm), 1e-6)
        self.power = float(power)
        self.lo, self.hi = float(lo), float(hi)
        # Which vocabulary entries begin a word. A step whose clean choice
        # continues a word is spelling, not a decision -- the model is sure of
        # the rest of a word once it has started it -- and amplifying the push
        # there breaks words ("skitter-shing", "musits"). Such steps are never
        # amplified. None treats every step as a decision.
        self.word_start = word_start
        self.history: List[float] = []

    def __call__(self, input_ids, scores):
        if scores.dim() != 2 or scores.shape[0] < 2:
            return scores
        self.calls += 1
        if self.steps and self.calls > self.steps:
            return scores
        clean = scores[1].float()
        top = torch.log_softmax(clean, dim=-1).topk(2).values
        margin = float(top[0] - top[1])
        s = min(max((margin / self.ref) ** self.power, self.lo), self.hi)
        if self.word_start is not None:
            first = int(clean.argmax())
            if first < self.word_start.numel() and not bool(self.word_start[first]):
                s = min(s, 1.0)
        s = s / self.norm
        self.history.append(s)
        out = scores.clone()
        out[0] = (clean + s * (scores[0].float() - clean)).to(scores.dtype)
        return out


class FlipRecorder(LogitsProcessor):
    """Records, per step, the clean model's top-two margin and whether the
    story's scores -- after every other processor -- still pick the clean
    model's top word. The evidence for where the noise acts: binned by margin,
    the share of steps whose choice the noise changed.

    Last in the list, after any margin scaling, so it sees what is sampled from.
    Accumulates in ``stats`` across stories: bin -> [steps, changed].
    """

    EDGES = (0.5, 1.0, 2.0, 4.0, 8.0)

    def __init__(self, stats: Dict[str, List[int]]):
        self.stats = stats

    def __call__(self, input_ids, scores):
        if scores.dim() != 2 or scores.shape[0] < 2:
            return scores
        clean = torch.log_softmax(scores[1].float(), dim=-1)
        top = clean.topk(2)
        margin = float(top.values[0] - top.values[1])
        changed = int(int(scores[0].float().argmax()) != int(top.indices[0]))
        lo = 0.0
        for hi in (*self.EDGES, float("inf")):
            if margin < hi:
                key = f"{lo:g}-{hi:g}"
                break
            lo = hi
        rec = self.stats.setdefault(key, [0, 0])
        rec[0] += 1
        rec[1] += changed
        return scores

    @classmethod
    def summary(cls, stats: Dict[str, List[int]]) -> str:
        keys = sorted(stats, key=lambda k: float(k.split("-")[0]))
        return "  ".join(f"gap {k}: {stats[k][1] / max(stats[k][0], 1):.1%} of {stats[k][0]}"
                         for k in keys)
