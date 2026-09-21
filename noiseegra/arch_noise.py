"""Random noise placed at different parts of the transformer, sized alike.

The per-story offset adds a vector to the residual stream. That is one place in
the architecture and one kind of noise. This module puts genuinely random noise
at other places -- nothing here is learned from the model's outputs or aimed at
anything -- so they can be compared with each other and with the offset:

``rotate``   The residual stream, rotated rather than pushed. RMSNorm discards a
             hidden state's length, so an added vector spends part of itself
             changing a length the next layer throws away. Here the part of the
             state outside the protected rule directions is turned along the
             sphere toward a random direction by an angle, its length kept and
             the rule components left exactly as they were.
``headtemp`` Attention, per head. Each head's queries are scaled by a random
             factor, which sharpens or flattens where that head looks. Applied
             after Qwen3's per-head query norm, which would otherwise undo it.
``rope``     Position. Each rotary frequency is rescaled by a random factor, so
             the model's sense of how far back a word is shifts. Noise on the
             circle, not in the stream: no vector is added anywhere.
``value``    The prompt's cached values. Noise on the value vectors at the
             prompt positions only, so each story reads the instruction slightly
             differently while nothing the story writes is touched directly.
``mlpmask``  The MLP's own features. A random fraction of each steered layer's
             MLP units is silenced, the rest rescaled -- noise in the model's own
             coordinates rather than in an arbitrary basis.

Every draw is fixed for a story: a generator seeded from the story's seed, so
each forward pass during sizing and generation sees the same noise. The size is
set per story the same way the offset's is, by bisection on the mechanism's one
knob until the next-token distribution moves a set Fisher-Rao distance along
the reference passage, in units of how far nucleus sampling moves it. So every
mechanism, and the offset, is compared at the same change in the model's
predictions, and they differ only in where and how that change is made.
"""

from __future__ import annotations

import math
import zlib
from typing import Dict, List, Optional

import torch

from .fisher import fisher_rao_distance

MECHANISMS = ("rotate", "headtemp", "rope", "value", "mlpmask")
# The largest value each knob may take in the search.
_KNOB_MAX = {"rotate": math.pi / 2, "headtemp": 2.0, "rope": 0.5,
             "value": 4.0, "mlpmask": 0.6}


def _layers(egra, plan) -> List[int]:
    blocks = egra._get_transformer_blocks()
    return sorted({egra._normalize_layer_index(int(i), len(blocks))
                   for i in plan.layers})


def _rotary(egra):
    """The rotary module: shared across the model, or one per attention block."""
    inner = getattr(egra.model, "model", None)
    if inner is not None and hasattr(inner, "rotary_emb"):
        return [inner.rotary_emb]
    return [b.self_attn.rotary_emb for b in egra._get_transformer_blocks()
            if hasattr(b.self_attn, "rotary_emb")]


class ArchNoise:
    """One story's noise at one place in the architecture."""

    def __init__(self, egra, plan, mechanism: str, seed: int, n_prompt: int):
        if mechanism not in MECHANISMS:
            raise ValueError(f"unknown mechanism {mechanism!r}; one of {MECHANISMS}")
        self.egra, self.plan, self.mech = egra, plan, mechanism
        self.seed = int(seed) & 0x7FFFFFFF
        self.n_prompt = int(n_prompt)
        self.knob = 0.0
        self.handles: List = []
        self._units: Dict = {}

    # -- fixed random draws ------------------------------------------------ #
    def _draw(self, key, shape, device, kind="normal"):
        """The same random tensor every time it is asked for, per story and key."""
        k = (key, tuple(shape), str(device), kind)
        if k not in self._units:
            # crc32 rather than hash(): Python salts string hashes per process,
            # which would give a story different noise on every run.
            g = torch.Generator(device="cpu").manual_seed(
                (self.seed * 1000003 + zlib.crc32(repr(key).encode())) & 0x7FFFFFFF)
            t = (torch.randn(shape, generator=g) if kind == "normal"
                 else torch.rand(shape, generator=g))
            self._units[k] = t.to(device)
        return self._units[k]

    # -- hooks --------------------------------------------------------------- #
    def attach(self) -> None:
        self.detach()
        blocks = self.egra._get_transformer_blocks()
        layers = _layers(self.egra, self.plan)
        m = self.mech
        if m == "rotate":
            for li in layers:
                self.handles.append(blocks[li].register_forward_hook(self._rotate_hook(li)))
        elif m == "headtemp":
            for li in layers:
                att = blocks[li].self_attn
                site = getattr(att, "q_norm", None) or att.q_proj
                self.handles.append(site.register_forward_hook(self._headtemp_hook(li, att)))
        elif m == "rope":
            for rot in _rotary(self.egra):
                self.handles.append(rot.register_forward_hook(self._rope_hook()))
        elif m == "value":
            for li in layers:
                att = blocks[li].self_attn
                self.handles.append(att.v_proj.register_forward_hook(self._value_hook(li)))
        elif m == "mlpmask":
            for li in layers:
                down = blocks[li].mlp.down_proj
                self.handles.append(down.register_forward_pre_hook(self._mask_hook(li)))

    def detach(self) -> None:
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []

    def _rotate_hook(self, li):
        def hook(module, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if not isinstance(t, torch.Tensor) or t.dim() != 3 or self.knob <= 0:
                return None
            with torch.no_grad():
                x = t.float()
                prot = self.plan.layer_plans[li].protect
                u = self._draw(("rot", li), (x.shape[-1],), x.device)
                if prot is not None:
                    p = prot.to(x.device, torch.float32)
                    u = u - p @ (p.t() @ u)
                    inside = (x @ p) @ p.t()
                else:
                    inside = torch.zeros_like(x)
                rest = x - inside
                n = rest.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                r = rest / n
                perp = u.view(1, 1, -1) - (r * u.view(1, 1, -1)).sum(-1, keepdim=True) * r
                perp = perp / perp.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                turned = n * (math.cos(self.knob) * r + math.sin(self.knob) * perp)
                t.copy_((inside + turned).to(t.dtype))
            return None
        return hook

    def _headtemp_hook(self, li, att):
        heads = int(getattr(att, "num_heads", 0) or getattr(
            getattr(att, "config", None), "num_attention_heads", 0) or 0)

        def hook(module, inp, out):
            if not isinstance(out, torch.Tensor) or self.knob <= 0:
                return None
            with torch.no_grad():
                if out.dim() == 4:                   # (b, seq, heads, head_dim)
                    h = out.shape[2]
                    z = self._draw(("head", li), (h,), out.device)
                    out.mul_(torch.exp(self.knob * z).to(out.dtype).view(1, 1, h, 1))
                elif out.dim() == 3 and heads:       # (b, seq, heads * head_dim)
                    z = self._draw(("head", li), (heads,), out.device)
                    s = torch.exp(self.knob * z).to(out.dtype)
                    b, n, d = out.shape
                    out.view(b, n, heads, d // heads).mul_(s.view(1, 1, heads, 1))
            return None
        return hook

    def _rope_hook(self):
        def hook(module, inp, out):
            if self.knob <= 0 or not isinstance(out, (tuple, list)) or len(out) != 2:
                return None
            x, position_ids = inp[0], inp[1] if len(inp) > 1 else None
            inv = getattr(module, "inv_freq", None)
            if position_ids is None or inv is None:
                return None
            z = self._draw(("rope",), tuple(inv.shape), inv.device)
            scaled = inv.float() * (1.0 + self.knob * z).clamp_min(0.05)
            freqs = position_ids[:, :, None].float() * scaled[None, None, :]
            emb = torch.cat((freqs, freqs), dim=-1)
            k = float(getattr(module, "attention_scaling", 1.0) or 1.0)
            cos, sin = out
            return (emb.cos() * k).to(cos.dtype), (emb.sin() * k).to(sin.dtype)
        return hook

    def _value_hook(self, li):
        def hook(module, inp, out):
            if not isinstance(out, torch.Tensor) or out.dim() != 3 or self.knob <= 0:
                return None
            n = out.shape[1]
            if n <= 1:                                # a decode step: not the prompt
                return None
            k = min(self.n_prompt, n)
            with torch.no_grad():
                v = out[:, :k, :]
                rms = v.float().pow(2).mean().sqrt()
                xi = self._draw(("val", li, k), (k, out.shape[-1]), out.device)
                v.add_((self.knob * rms * xi).to(out.dtype).unsqueeze(0))
            return None
        return hook

    def _mask_hook(self, li):
        def hook(module, inp):
            x = inp[0]
            if not isinstance(x, torch.Tensor) or self.knob <= 0:
                return None
            u = self._draw(("mask", li), (x.shape[-1],), x.device, kind="uniform")
            keep = (u >= self.knob).to(x.dtype) / max(1.0 - self.knob, 1e-3)
            return (x * keep,) + tuple(inp[1:])
        return hook


# --------------------------------------------------------------------------- #
#  Sizing: the same Fisher-Rao unit as the offset                             #
# --------------------------------------------------------------------------- #
def calibrate(noise: ArchNoise, prompt_ids: torch.Tensor, target: float, *,
              iters: int = 12, cache: Optional[Dict] = None) -> Dict[str, float]:
    """Set the mechanism's knob so it moves the predictions by ``target``.

    ``target`` is in the unit of fisher_calibration.nucleus_unit, measured along
    the same reference passage, so a mechanism at 1.0 changes the predictions
    as much as the offset at 1.0 and as nucleus sampling does.
    """
    from .fisher_calibration import _logits, _probs, nucleus_unit, reference_passage

    egra, plan = noise.egra, noise.plan
    key = ("arch", tuple(int(i) for i in prompt_ids.view(-1).tolist()))
    n_prompt = int(prompt_ids.shape[-1])
    noise.detach()
    if cache is not None and key in cache:
        passage, unit, ref = cache[key]
    else:
        passage = reference_passage(egra, plan, prompt_ids)
        unit = nucleus_unit(egra, plan, passage, n_prompt)
        ref = _probs(_logits(egra, plan, passage, n_prompt, with_offset=False), n_prompt)
        if cache is not None:
            cache[key] = (passage, unit, ref)
    goal = float(target) * unit

    def distance(knob: float) -> float:
        noise.knob = knob
        noise.attach()
        try:
            moved = _probs(_logits(egra, plan, passage, n_prompt, with_offset=False),
                           n_prompt)
        finally:
            noise.detach()
        return float(fisher_rao_distance(ref, moved).mean())

    top_knob = _KNOB_MAX[noise.mech]
    top = distance(top_knob)
    if top < goal:
        noise.knob = top_knob
        return {"knob": top_knob, "distance": top / unit, "unit": unit, "reached": 0.0}
    lo, hi = math.log(top_knob / 1000.0), math.log(top_knob)
    best = (top_knob, top)
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        d = distance(math.exp(mid))
        if abs(d - goal) < abs(best[1] - goal):
            best = (math.exp(mid), d)
        if d < goal:
            lo = mid
        else:
            hi = mid
    noise.knob = best[0]
    return {"knob": best[0], "distance": best[1] / unit, "unit": unit, "reached": 1.0}
