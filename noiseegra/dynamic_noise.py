"""Noise whose direction changes while the story is written, and the B-Trans baseline.

Three things, all used by ``EGRA.generate_with_orthogonal_steering``:

``attach_btrans``
    B-Trans (Yang & Zhang, arXiv 2512.25063): every hidden-size normalisation
    layer gets its own Gaussian offset z ~ N(0, sigma^2 I), drawn once per story
    and added to that layer's output at every position ("y = Norm(x) + z", their
    Figure 1). The paper gives sigma = 0.02 and no other value. Query and key
    norms (per attention head, head-dim wide) are left alone: the paper's wrapper
    reads the input as (batch, tokens, hidden).

``CleanGuard``
    The story samples only among tokens its noise-free shadow row gives at least
    ``alpha`` times the probability of the shadow's top token. When the story's
    own top token falls outside that set (the noise has pushed a token the clean
    model rules out), ``on_violation`` is called -- the direction correction.

``RuleDebt``
    Reads, each step, which content rules the text so far still owes
    (dialogue, a he and a she, a simile) and whether the noise has made the
    story less likely than its shadow to write an owed rule's words; if so it
    calls ``on_suppressed`` -- the noise's direction is then turned off the
    gradient that lowers them. The directions the noise is kept clear of change
    as the story pays its rules.

``DefaultAvoid`` / ``rotate_away``
    At each step where the noise-free shadow is choosing (its top word below
    a probability), the step is replayed with a gradient and each layer's
    noise direction is turned a fixed angle away from the direction that
    would make the story write the shadow's top word: the noise learns,
    while the story is written, which way leads off the model's default.
    Only the direction changes; the length is kept.

``feedback_update``
    Turns each layer's per-story noise by the displacement it caused downstream
    (the story's state minus its shadow's at a later layer): toward the part of
    that displacement not along the noise itself (resonance), away from it, or
    to the opposite of the whole displacement (cancel).
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

import torch
from transformers import LogitsProcessor


def attach_btrans(model, sigma: float) -> List:
    """One N(0, sigma^2) offset per hidden-size norm layer, added to its output.

    Drawn from the ambient RNG, so a story seeded before this call gets the same
    draws every time. Returns the hook handles; removing them removes the noise.
    """
    cfg = getattr(model, "config", None)
    cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    dim = int(getattr(cfg, "hidden_size"))
    handles = []
    for _, module in model.named_modules():
        if "norm" not in type(module).__name__.lower():
            continue
        w = getattr(module, "weight", None)
        if not isinstance(w, torch.Tensor) or w.dim() != 1 or w.numel() != dim:
            continue
        z = torch.randn(dim, dtype=torch.float32) * float(sigma)
        z = z.to(device=w.device)

        def hook(mod, inp, out, z=z):
            if isinstance(out, torch.Tensor):
                return out + z.to(out.dtype)
            return out

        handles.append(module.register_forward_hook(hook))
    if not handles:
        raise RuntimeError("B-Trans: found no normalisation layer of the hidden size")
    return handles


def prefix_cache(cache, row: int, length: int):
    """A new cache holding one row's first ``length`` positions, for a replay.

    Works on the layered DynamicCache (``cache.layers[i].keys``) and on the older
    one (``cache.key_cache[i]``). The generation's own cache is not touched.
    """
    from transformers import DynamicCache

    layers = getattr(cache, "layers", None)
    if layers is not None:
        pairs = [(l.keys, l.values) for l in layers]
    else:
        pairs = list(zip(cache.key_cache, cache.value_cache))
    out = DynamicCache()
    for i, (k, v) in enumerate(pairs):
        out.update(k[row:row + 1, :, :length].detach().clone(),
                   v[row:row + 1, :, :length].detach().clone(), i)
    return out


class CleanGuard(LogitsProcessor):
    """Sample the story (row 0) only among tokens its shadow (row 1) allows.

    Allowed: shadow probability >= ``alpha`` x the shadow's top probability. A
    violation is a step whose story top token is not allowed; ``on_violation``
    (input_ids, token) is then called and returns whether it corrected anything.
    With no shadow row the scores pass through untouched.
    """

    def __init__(self, alpha: float,
                 on_violation: Optional[Callable[[torch.Tensor, int], bool]] = None):
        self.alpha = float(alpha)
        self.on_violation = on_violation
        self.steps = 0
        self.violations = 0
        self.corrections = 0

    def __call__(self, input_ids, scores):
        if scores.dim() != 2 or scores.shape[0] < 2:
            return scores
        with torch.no_grad():
            clean = torch.softmax(scores[1].float(), dim=-1)
            allowed = clean >= self.alpha * clean.max()
            top = int(scores[0].argmax())
        self.steps += 1
        if not bool(allowed[top]):
            self.violations += 1
            if self.on_violation is not None and self.on_violation(input_ids, top):
                self.corrections += 1
        out = scores.clone()
        out[0] = scores[0].masked_fill(~allowed.to(scores.device), float("-inf"))
        return out

    def summary(self) -> Dict[str, float]:
        return {"steps": float(self.steps), "violations": float(self.violations),
                "corrections": float(self.corrections)}


class ConfidentAnchor(LogitsProcessor):
    """Where the noise-free shadow is near-certain, the story takes its word.

    At a step whose shadow (row 1) puts at least ``threshold`` on its top token,
    the story's scores (row 0) are replaced by the shadow's, so grammar,
    capitals after a full stop and the ends of set phrases come from the clean
    model; everywhere else the story samples from its own perturbed scores.
    The perturbation itself is untouched, so the story's state keeps diverging.
    """

    def __init__(self, threshold: float):
        self.threshold = float(threshold)
        self.steps = 0
        self.anchored = 0

    def __call__(self, input_ids, scores):
        if scores.dim() != 2 or scores.shape[0] < 2:
            return scores
        with torch.no_grad():
            p1 = float(torch.softmax(scores[1].float(), dim=-1).max())
        self.steps += 1
        if p1 < self.threshold:
            return scores
        self.anchored += 1
        out = scores.clone()
        out[0] = scores[1]
        return out

    def summary(self) -> Dict[str, float]:
        return {"steps": float(self.steps), "anchored": float(self.anchored)}


_QUOTES = ('"', "\u201c", "\u201d")


def debt_token_sets(tokenizer) -> Dict[str, List[int]]:
    """Token ids for each owed rule's words: speech marks, she/her, he/him, like/as."""
    words = {"she": {"she", "her", "hers", "herself"},
             "he": {"he", "him", "his", "himself"},
             "simile": {"like", "as"}}
    out: Dict[str, List[int]] = {"dialogue": [], "she": [], "he": [], "simile": []}
    for i in range(int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))):
        try:
            txt = tokenizer.decode([i])
        except Exception:
            continue
        if any(q in txt for q in _QUOTES):
            out["dialogue"].append(i)
            continue
        w = txt.strip().lower()
        if not w or not (txt[:1].isspace() or txt[:1].isupper()):
            continue
        for k, ws in words.items():
            if w in ws:
                out[k].append(i)
    return out


def owed_rules(text: str) -> List[str]:
    """The content rules the text so far has not paid."""
    from .constraint_metrics_en import _HE, _SHE, has_simile
    owed = []
    if not any(q in text for q in _QUOTES):
        owed.append("dialogue")
    if not _SHE.search(text):
        owed.append("she")
    if not _HE.search(text):
        owed.append("he")
    if not has_simile(text):
        owed.append("simile")
    return owed


class RuleDebt(LogitsProcessor):
    """Correct the noise where it lowers the words of a rule the story still owes.

    At each step with ``active()`` true, for every owed rule whose words the
    shadow (row 1) gives at least ``min_mass`` of its probability, the gap
    log P_story(words) - log P_shadow(words) is read; the rule with the largest
    shortfall beyond ``tau`` nats is passed to ``on_suppressed(input_ids,
    token_ids)``, which returns whether it corrected anything. Scores pass
    through unchanged.
    """

    def __init__(self, tokenizer, token_sets: Dict[str, List[int]], tau: float,
                 on_suppressed: Optional[Callable[[torch.Tensor, List[int]], bool]] = None,
                 active: Optional[Callable[[], bool]] = None, min_mass: float = 0.02):
        self.tokenizer = tokenizer
        self.sets = {k: torch.tensor(v, dtype=torch.long) for k, v in token_sets.items() if v}
        self.tau = float(tau)
        self.min_mass = float(min_mass)
        self.on_suppressed = on_suppressed
        self.active = active
        self.prompt_len: Optional[int] = None
        self.steps = 0
        self.triggers: Dict[str, int] = {}
        self.corrections = 0

    def __call__(self, input_ids, scores):
        if self.prompt_len is None:
            self.prompt_len = int(input_ids.shape[-1])
        if scores.dim() != 2 or scores.shape[0] < 2:
            return scores
        if self.active is not None and not self.active():
            return scores
        self.steps += 1
        text = self.tokenizer.decode(input_ids[0, self.prompt_len:], skip_special_tokens=True)
        owed = [r for r in owed_rules(text) if r in self.sets]
        if not owed:
            return scores
        with torch.no_grad():
            ls = torch.log_softmax(scores[:2].float(), dim=-1)
            best, worst = None, self.tau
            for r in owed:
                ids = self.sets[r].to(ls.device)
                clean = float(torch.logsumexp(ls[1, ids], dim=0))
                if clean < math.log(self.min_mass):
                    continue
                short = clean - float(torch.logsumexp(ls[0, ids], dim=0))
                if short > worst:
                    best, worst = r, short
        if best is None:
            return scores
        self.triggers[best] = self.triggers.get(best, 0) + 1
        if self.on_suppressed is not None and self.on_suppressed(input_ids, self.sets[best].tolist()):
            self.corrections += 1
        return scores

    def summary(self) -> Dict[str, float]:
        out = {"steps": float(self.steps), "corrections": float(self.corrections)}
        out.update({f"trig_{k}": float(v) for k, v in self.triggers.items()})
        return out


# How a reply that is not a story begins: a heading or title, or an answer to
# the reader ("Sure", "Here is", "Certainly", "Okay", "Of course").
_SLIP_STARTS = ("**", "#", "title", "sure", "here", "certainly", "okay", "of course", "aim ")


def slip_token_ids(tokenizer) -> List[int]:
    """Token ids a non-story opening starts with (heading marks, title, reply words)."""
    out = []
    for i in range(int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))):
        try:
            w = tokenizer.decode([i]).strip().lower()
        except Exception:
            continue
        if w and (w.startswith(_SLIP_STARTS[:2]) or any(
                w == x.strip() or w.startswith(x) for x in _SLIP_STARTS[2:])):
            out.append(i)
    return out


def _off_protected(vec: torch.Tensor, protect: Optional[torch.Tensor]) -> torch.Tensor:
    if protect is None:
        return vec
    p = protect.to(device=vec.device, dtype=vec.dtype)
    return vec - p @ (p.t() @ vec)


def correct_offsets(plan, grads: Dict[int, Optional[torch.Tensor]], eta: float, t: int) -> bool:
    """Remove ``eta`` of each layer's noise component along its token gradient.

    Only where the noise pushes the offending token up (component > 0). With a
    drifting direction the component is removed from the directions the drift
    mixes, so every later step is corrected too; the length is kept.
    """
    changed = False
    orth = getattr(plan, "offset_mode", "orth") == "orth"
    for layer, g in grads.items():
        lp = plan.layer_plans.get(layer)
        if lp is None or lp.offset is None or g is None:
            continue
        g = g.to(device=lp.offset.device, dtype=torch.float32)
        if orth:
            g = _off_protected(g, lp.protect)
        n = float(g.norm())
        if not math.isfinite(n) or n <= 1e-12:
            continue
        gh = g / n
        if lp.offset_traj is not None and lp.offset_basis is not None:
            cur = lp.offset_at(t).float()
            if float(cur @ gh) <= 0:
                continue
            basis = lp.offset_basis.float()
            basis = basis - eta * gh.unsqueeze(1) * (gh @ basis).unsqueeze(0)
            lp.offset_basis = basis.to(lp.offset_basis.dtype)
        else:
            v = lp.offset.float()
            c = float(v @ gh)
            if c <= 0:
                continue
            length = float(v.norm())
            nv = v - eta * c * gh
            lp.offset = (nv * (length / float(nv.norm().clamp_min(1e-12)))).to(lp.offset.dtype)
        changed = True
    return changed


def rotate_away(plan, grads: Dict[int, Optional[torch.Tensor]], eta: float) -> float:
    """Turn each layer's noise ``eta`` radians toward ``-grads[layer]``.

    The turn is toward the part of ``-g`` orthogonal to the current direction,
    projected off the protected subspace for an ``orth`` plan; the length is
    kept. Returns the mean cosine between each layer's old and new direction.
    """
    orth = getattr(plan, "offset_mode", "orth") == "orth"
    cosines = []
    for layer, g in grads.items():
        lp = plan.layer_plans.get(layer)
        if lp is None or lp.offset is None or g is None:
            continue
        v = lp.offset.float()
        length = float(v.norm())
        if length <= 1e-12:
            continue
        u = v / length
        d = -g.to(device=u.device, dtype=torch.float32)
        if orth:
            d = _off_protected(d, lp.protect)
        r = d - (d @ u) * u
        rn = float(r.norm())
        if not math.isfinite(rn) or rn <= 1e-12:
            continue
        new = math.cos(eta) * u + math.sin(eta) * (r / rn)
        if orth:
            new = _off_protected(new, lp.protect)
        new = new / new.norm().clamp_min(1e-12)
        cosines.append(float(new @ u))
        lp.offset = (new * length).to(lp.offset.dtype)
    return sum(cosines) / len(cosines) if cosines else 1.0


class DefaultAvoid(LogitsProcessor):
    """Where the shadow is choosing, turn the noise away from its default word.

    At a step with ``active()`` true whose shadow (row 1) gives its top word
    less than ``p1_max``, ``on_step(input_ids, token)`` is called with that
    word; it returns the mean cosine of the turn, or None if it did not turn.
    Scores pass through unchanged.
    """

    def __init__(self, p1_max: float, on_step: Callable[[torch.Tensor, int], Optional[float]],
                 active: Optional[Callable[[], bool]] = None,
                 after: Optional[Callable[[], None]] = None,
                 cohere_alpha: float = 0.0):
        # With cohere_alpha > 0 the trigger is instead the story's own top word
        # falling outside the shadow's support (shadow probability below alpha
        # times the shadow's top): the story is breaking away from anything the
        # clean model would write.
        self.cohere_alpha = float(cohere_alpha)
        self.p1_max = float(p1_max)
        self.on_step = on_step
        self.active = active
        self.after = after
        self.steps = 0
        self.turns = 0
        self.cos_sum = 0.0

    def __call__(self, input_ids, scores):
        if scores.dim() != 2 or scores.shape[0] < 2:
            return scores
        if self.active is not None and not self.active():
            return scores
        self.steps += 1
        with torch.no_grad():
            p = torch.softmax(scores[1].float(), dim=-1)
            p1, top = p.max(dim=-1)
            if self.cohere_alpha > 0:
                own = int(scores[0].argmax())
                fire = float(p[own]) < self.cohere_alpha * float(p1)
            else:
                fire = float(p1) < self.p1_max
        if fire:
            cos = self.on_step(input_ids, int(top))
            if cos is not None:
                self.turns += 1
                self.cos_sum += float(cos)
        if self.after is not None:
            self.after()
        return scores

    def summary(self) -> Dict[str, float]:
        return {"steps": float(self.steps), "turns": float(self.turns),
                "step_cos": self.cos_sum / max(self.turns, 1)}


_SENTENCE_END = (".", "!", "?", "\n", ".\"", "!\"", "?\"", ".\u201d", "!\u201d", "?\u201d")


class PulseTrigger(LogitsProcessor):
    """Start a new segment of noise at the first sentence end after ``every`` steps.

    At each step it checks whether at least ``every`` steps have passed since the
    current segment began and the story's last word ends a sentence; if so it
    calls ``on_pulse(input_ids)`` (which draws a new direction and reads the
    context again) and returns. Scores pass through unchanged.
    """

    def __init__(self, tokenizer, every: int, on_pulse: Callable[[torch.Tensor], None],
                 start: Callable[[], int], step: Callable[[], int], max_pulses: int = 8):
        self.tokenizer = tokenizer
        self.every = int(every)
        self.on_pulse = on_pulse
        self.start = start
        self.step = step
        self.max_pulses = int(max_pulses)
        self.pulses = 0
        self.at: List[int] = []

    def __call__(self, input_ids, scores):
        t = int(self.step())
        if self.pulses >= self.max_pulses or t - int(self.start()) < self.every:
            return scores
        last = self.tokenizer.decode(input_ids[0, -1:], skip_special_tokens=True).rstrip(" ")
        if not last.endswith(_SENTENCE_END):
            return scores
        self.on_pulse(input_ids)
        self.pulses += 1
        self.at.append(t)
        return scores

    def summary(self) -> Dict[str, float]:
        return {"pulses": float(self.pulses), "at": list(self.at)}


def feedback_update(plan, delta: torch.Tensor, mode: str, eta: float) -> float:
    """Turn every layer's per-story noise by the downstream displacement ``delta``.

    ``toward``/``away``: rotate by ``eta`` radians toward/away from the part of
    ``delta`` orthogonal to the noise. ``cancel``: the new direction is
    ``(1 - eta) u - eta * delta_hat`` (``eta`` = 1 replaces it by the opposite
    of the displacement). The length is kept; with an ``orth`` plan the result is
    projected off the protected subspace. Returns the mean cosine between each
    layer's old and new direction.
    """
    d = delta.float()
    dn = float(d.norm())
    if not math.isfinite(dn) or dn <= 1e-12:
        return 1.0
    orth = getattr(plan, "offset_mode", "orth") == "orth"
    cosines = []
    for lp in plan.layer_plans.values():
        if lp.offset is None:
            continue
        v = lp.offset.float()
        length = float(v.norm())
        if length <= 1e-12:
            continue
        u = v / length
        dd = d.to(u.device)
        if mode in ("toward", "away"):
            r = dd - (dd @ u) * u
            rn = float(r.norm())
            if rn <= 1e-9 * dn:
                continue
            s = 1.0 if mode == "toward" else -1.0
            new = math.cos(eta) * u + s * math.sin(eta) * (r / rn)
        elif mode == "cancel":
            new = (1.0 - eta) * u - eta * (dd / dn)
        else:
            raise ValueError(f"unknown feedback mode {mode!r}")
        if orth:
            new = _off_protected(new, lp.protect)
        nn = float(new.norm())
        if nn <= 1e-12:
            continue
        new = new / nn
        cosines.append(float(new @ u))
        lp.offset = (new * length).to(lp.offset.dtype)
    return sum(cosines) / len(cosines) if cosines else 1.0
