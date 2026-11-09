"""Per-story random noise on attention's own vectors: queries and keys.

Attention writes softmax(Q K^T / sqrt(d)) V into the stream. A vector added to
the stream can leave the set of states real text produces, and the layers after
it correct it away or break. Noise on Q or K cannot: whatever it does to the
scores, the output is still a weighted mix of the same genuine value vectors.
It changes what the model looks at, not what is there to be read.

``q``  One random direction per head and layer, fixed for the story, added to
       the queries of the words being written (and of the prompt's last
       position, which writes the first word): the story keeps looking at its
       context with a consistent, story-specific slant. Full for ``fade``
       written tokens, then down to nothing over ``fade`` more on a cosine.
``k``  Independent noise on the keys of each prompt position, stored with the
       prompt: every later word finds some parts of the instruction easier and
       others harder to attend to, the same way for the whole story.
``v``  Independent noise on the prompt's stored values (as arch_noise's
       ``value``), for comparison: what is read changes, not where from.

Sizes are multiples of the vector's own RMS at that layer. Set by the runner
(--attn-noise q:1 k:0.5 --attn-fade 32); None = off.
"""
from __future__ import annotations

import math
import zlib
from typing import Dict, List, Optional

import torch

# {"q": size, "k": size, "v": size}; missing or 0 = off.
ATTN_NOISE: Optional[Dict[str, float]] = None
ATTN_FADE: int = 32


def _site(att, which: str):
    """The module whose output is the Q / K / V vectors (after Qwen3's per-head
    norm on Q and K, which would otherwise rescale the noise away)."""
    norm = getattr(att, f"{which}_norm", None)
    return norm if norm is not None else getattr(att, f"{which}_proj")


def _unit(seed: int, key, shape, device):
    g = torch.Generator(device="cpu").manual_seed(
        (int(seed) * 1000003 + zlib.crc32(repr(key).encode())) & 0x7FFFFFFF)
    return torch.randn(shape, generator=g).to(device)


def attach(egra, layers: List[int], n_prompt: int, seed: int,
           spec: Optional[Dict[str, float]] = None, fade: Optional[int] = None):
    """Hooks for one story. Returns (handles, state); state["log"] collects the
    relative sizes actually applied."""
    spec = dict(spec if spec is not None else (ATTN_NOISE or {}))
    fade = int(ATTN_FADE if fade is None else fade)
    blocks = egra._get_transformer_blocks()
    handles, state = [], {"t": {}, "log": {}}

    def env(step: int) -> float:
        if fade <= 0:
            return 1.0
        late = min(max((step - fade) / fade, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * late))

    def q_hook(li, size):
        def hook(module, inp, out):
            if not isinstance(out, torch.Tensor) or out.dim() < 3:
                return None
            n = out.shape[1]
            with torch.no_grad():
                tail = out.shape[2:]
                u = _unit(seed, ("q", li), tuple(tail), out.device)
                # each head's direction at unit RMS
                u = u / u.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
                if n > 1:                             # the prompt: its last position only
                    g, pos = 1.0, slice(n - 1, n)
                    state["t"][li] = 0
                else:
                    step = state["t"].get(li, 0)
                    state["t"][li] = step + 1
                    g, pos = env(step), slice(0, 1)
                if g <= 0:
                    return None
                x = out[:1, pos]
                rms = x.float().pow(2).mean().sqrt()
                x.add_((g * size * rms * u).to(out.dtype))
            return None
        return hook

    def kv_hook(li, size, which):
        def hook(module, inp, out):
            if not isinstance(out, torch.Tensor) or out.dim() < 3 or out.shape[1] <= 1:
                return None                           # a decode step: not the prompt
            k = min(int(n_prompt), out.shape[1])
            with torch.no_grad():
                x = out[:1, 1:k]                      # the first token is left alone
                rms = x.float().pow(2).mean().sqrt()
                xi = _unit(seed, (which, li), tuple(x.shape[1:]), out.device)
                x.add_((size * rms * xi).to(out.dtype))
            return None
        return hook

    for li in layers:
        att = blocks[int(li)].self_attn
        if float(spec.get("q", 0) or 0) > 0:
            handles.append(_site(att, "q").register_forward_hook(q_hook(int(li), float(spec["q"]))))
        for which in ("k", "v"):
            if float(spec.get(which, 0) or 0) > 0:
                handles.append(_site(att, which).register_forward_hook(
                    kv_hook(int(li), float(spec[which]), which)))
    return handles, state


def tag() -> str:
    if not ATTN_NOISE:
        return ""
    def f(x):
        return (f"{x:g}").replace(".", "p")
    return "__att" + "".join(f"{w}{f(ATTN_NOISE[w])}" for w in ("q", "k", "v")
                             if float(ATTN_NOISE.get(w, 0) or 0) > 0) + f"f{int(ATTN_FADE)}"
