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
