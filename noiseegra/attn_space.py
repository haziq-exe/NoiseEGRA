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


# --------------------------------------------------------------------------- #
#  Rule steering in attention's own spaces                                     #
# --------------------------------------------------------------------------- #

# Set by the runner: the contrast-pair file and the rule names steered.
STEER_PAIRS: Optional[str] = None
STEER_NAMES: List[str] = []


@torch.no_grad()
def contrast_directions(egra, prompt_ids: torch.Tensor, layers: List[int], which: str):
    """One direction per layer in Q (or V) space for following the rules.

    The same contrast pairs the residual steering uses, read where attention
    computes its queries (or values): for each rule, the mean over its pairs of
    the Q vectors while the model reads the rule-following sentence minus the
    rule-breaking one, after the story prompt, over equally many positions.
    Each rule's direction is set to unit RMS, the rules are summed and the sum
    set to unit RMS again, so one size steers all rules together.
    """
    key = (which, tuple(STEER_NAMES), int(prompt_ids.shape[-1]), tuple(layers))
    cache = getattr(egra, "_attn_dirs", None) or {}
    if key in cache:
        return cache[key]
    from .steering_vectors import load_pairs
    cons = load_pairs(STEER_PAIRS)
    cons = cons.get("constraints", cons)
    blocks = egra._get_transformer_blocks()
    tok = egra.tokenizer
    ctx = prompt_ids[0].tolist()
    box: Dict[int, torch.Tensor] = {}
    hs = [(_site(blocks[int(li)].self_attn, which).register_forward_hook(
        (lambda li: (lambda m, i, o: box.__setitem__(li, o[0].detach().float())))(int(li))))
        for li in layers]

    def read(ids, start, count):
        t = torch.tensor([ids], dtype=torch.long, device=prompt_ids.device)
        egra.model(input_ids=t, use_cache=False)
        return {li: box[li][start:start + count].mean(0).cpu() for li in box}

    total: Dict[int, torch.Tensor] = {}
    try:
        for name in STEER_NAMES:
            if name not in cons:
                continue
            acc: Dict[int, torch.Tensor] = {}
            for item in cons[name]["pairs"]:
                pre = tok(item.get("prefix", ""), add_special_tokens=False)["input_ids"]
                p = tok(item["positive"], add_special_tokens=False)["input_ids"]
                n = tok(item["negative"], add_special_tokens=False)["input_ids"]
                c = min(len(p), len(n)); start = len(ctx) + len(pre)
                a = read(ctx + pre + p, start, c); b = read(ctx + pre + n, start, c)
                for li in a:
                    acc[li] = acc.get(li, 0) + (a[li] - b[li])
            for li, d in acc.items():
                d = d / d.pow(2).mean().sqrt().clamp_min(1e-8)
                total[li] = total.get(li, 0) + d
    finally:
        for h in hs:
            h.remove()
    dirs = {li: d / d.pow(2).mean().sqrt().clamp_min(1e-8) for li, d in total.items()}
    cache[key] = dirs
    egra._attn_dirs = cache
    return dirs


class instruction_boost:
    """While the story is written, every head's attention scores toward the
    instruction's tokens are raised by ``beta`` (its attention weight there is
    multiplied by e^beta before renormalising): the rules are consulted more,
    nothing is added to any vector. Done by wrapping the attention function for
    the length of one generation."""

    def __init__(self, beta: float, span, n_prompt: int):
        self.beta, self.span, self.n_prompt, self.calls = float(beta), span, int(n_prompt), 0

    def __enter__(self):
        import torch.nn.functional as F
        self.F, self.orig = F, F.scaled_dot_product_attention

        def wrapped(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                    scale=None, **kw):
            L, S = query.shape[-2], key.shape[-2]
            if L != 1 or S < self.n_prompt:           # only the words being written
                return self.orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                                 is_causal=is_causal, scale=scale, **kw)
            bias = torch.zeros(S, dtype=query.dtype, device=query.device)
            bias[self.span[0]:self.span[1]] = self.beta
            bias = bias.view(*([1] * (query.dim() - 1)), S)
            if attn_mask is None:
                m = bias
            elif attn_mask.dtype == torch.bool:
                m = torch.zeros_like(attn_mask, dtype=query.dtype).masked_fill(
                    ~attn_mask, float("-inf")) + bias
            else:
                m = attn_mask.to(query.dtype) + bias
            self.calls += 1
            return self.orig(query, key, value, attn_mask=m, dropout_p=dropout_p,
                             is_causal=False, scale=scale, **kw)

        F.scaled_dot_product_attention = wrapped
        return self

    def __exit__(self, *exc):
        self.F.scaled_dot_product_attention = self.orig
        return False


def _instruction_span(egra, prompt, chat: str, n_prompt: int):
    """Token positions of the user's message inside the prompt."""
    try:
        content = prompt[-1]["content"]
        a = chat.find(content)
        enc = egra.tokenizer(chat, return_offsets_mapping=True)
        off = enc["offset_mapping"]
        idx = [i for i, (s, e) in enumerate(off) if e > a and s < a + len(content)]
        if a >= 0 and idx:
            return (idx[0], idx[-1] + 1)
    except Exception:
        pass
    return (1, max(2, n_prompt - 1))


def generate_with_attn_steer(model, prompt, seed: int, kind: str, layers, *,
                             max_new_tokens: int = 500, max_words=None,
                             temperature: float = 1.0, story_index=None) -> str:
    """An otherwise untouched story with the rules steered inside attention.

    ``qsteer:B``  B x RMS(Q) of the rule direction in query space added to the
                  queries of every word written: the story looks at its context
                  the way rule-following text does.
    ``vsteer:B``  the same in value space, on every position's values: what
                  attention reads out is shifted toward rule-following text.
    ``boost:B``   attention to the instruction raised by B (instruction_boost).
    """
    which, val = kind.split(":")
    beta = float(val)
    chat = model.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    ids = model.tokenizer(chat, return_tensors="pt")["input_ids"].to(model._input_device())
    n_prompt = int(ids.shape[-1])
    blocks = model._get_transformer_blocks()
    layers = [int(l) for l in layers]
    handles, ctx, note = [], None, ""
    if which in ("qsteer", "vsteer"):
        w = which[0]
        dirs = contrast_directions(model, ids, layers, w)

        def hook_for(li):
            d = dirs[li]

            def hook(module, inp, out):
                if not isinstance(out, torch.Tensor) or out.dim() < 3:
                    return None
                n = out.shape[1]
                with torch.no_grad():
                    if w == "q":
                        pos = slice(n - 1, n)          # the word about to be written
                    else:
                        pos = slice(1, n) if n > 1 else slice(0, 1)
                    x = out[:, pos]
                    rms = x.float().pow(2).mean().sqrt()
                    x.add_((beta * rms * d.to(out.device).view(*x.shape[2:])).to(out.dtype))
                return None
            return hook

        handles = [_site(blocks[li].self_attn, w).register_forward_hook(hook_for(li))
                   for li in layers if li in dirs]
    elif which == "boost":
        span = _instruction_span(model, prompt, chat, n_prompt)
        ctx = instruction_boost(beta, span, n_prompt)
        note = f", instruction tokens {span[0]}-{span[1]} of {n_prompt}"
    else:
        raise ValueError(f"unknown attention steering {kind!r}")
    try:
        if ctx is not None:
            with ctx:
                out = model.generate(prompt, max_new_tokens=max_new_tokens, do_sample=True,
                                     temperature=temperature, seed=seed, max_words=max_words)
            note += f", applied on {ctx.calls} attention calls"
        else:
            out = model.generate(prompt, max_new_tokens=max_new_tokens, do_sample=True,
                                 temperature=temperature, seed=seed, max_words=max_words)
    finally:
        for h in handles:
            h.remove()
    if story_index is not None and int(story_index) < 3:
        print(f"  [attention steering {kind}] story {story_index}: layers {layers[0]}-{layers[-1]}"
              f"{note}", flush=True)
    return out
