"""Hidden context at the start of the reply: a control for the hidden daydream.

The daydream writes a short opening under strong noise and hides it. These
controls hide something that involved no noise at all -- random words, or a
random coherent sentence -- before an untouched story, to tell whether the
daydream's effect needs the noise or only some unrelated text in front of the
story.
"""
import math
from pathlib import Path

import torch
from transformers import LogitsProcessor

SENTENCES = Path(__file__).resolve().parents[1] / "eval" / "random_sentences.txt"
KINDS = ("words", "sentence")


def word_ids(model) -> list:
    """Vocabulary entries that are a whole alphabetic word (word-initial marker,
    then letters only), so a random draw reads as random English words rather
    than code, other scripts or word pieces."""
    cached = getattr(model, "_hidden_word_ids", None)
    if cached is not None:
        return cached
    tok = model.tokenizer
    try:
        n = len(tok)
    except TypeError:
        n = int(model.model.get_input_embeddings().weight.shape[0])
    toks = (tok.convert_ids_to_tokens(list(range(n))) if hasattr(tok, "convert_ids_to_tokens")
            else [tok.decode([i]) for i in range(n)])
    ids = [i for i, s in enumerate(toks)
           if isinstance(s, str) and len(s) >= 2 and s[0] in "Ġ▁ "
           and s[1:].isalpha() and s[1:].isascii()]
    if not ids:
        raise ValueError("no whole-word entries in the vocabulary")
    model._hidden_word_ids = ids
    return ids


def hidden_text(model, kind: str, seed: int, n_tokens: int = 32) -> str:
    """This story's hidden context, drawn from its own seed (a private
    generator, so the story's sampling sees the same random stream as the
    untouched story with that seed)."""
    g = torch.Generator().manual_seed(int(seed) & 0x7FFFFFFF)
    if kind == "words":
        pool = word_ids(model)
        pick = torch.randint(len(pool), (int(n_tokens),), generator=g).tolist()
        return model.tokenizer.decode([pool[i] for i in pick]).strip()
    if kind == "sentence":
        lines = [s for s in SENTENCES.read_text().splitlines() if s.strip()]
        return lines[int(torch.randint(len(lines), (1,), generator=g))]
    raise ValueError(f"hidden context must be one of {KINDS}, got {kind!r}")


def _sentence_end(tok, ids) -> bool:
    s = tok.decode(ids)
    return "\n" in s[-3:] or s.rstrip().rstrip('"”’\'').endswith((".", "!", "?"))


def _turning_push(model, seed: int, sigma: float, n_steps: int, layers):
    """A per-story push of ``sigma`` times the hidden state's own norm at each
    layer in ``layers``, along a direction drawn from ``seed`` that turns by a
    quarter circle over ``n_steps`` steps. Returns (state, handles); the push
    acts only while ``state["on"]``, at step ``state["t"]``."""
    W = model.model.get_input_embeddings().weight
    g = torch.Generator().manual_seed(int(seed) & 0x7FFFFFFF)
    a = torch.randn(W.shape[1], generator=g)
    b = torch.randn(W.shape[1], generator=g)
    b = b - (b @ a) / (a @ a) * a
    a, b = a / a.norm(), b / b.norm()
    state = {"on": False, "t": 0}

    def push(mod, inp, out):
        if not state["on"]:
            return None
        state["fired"] = state.get("fired", 0) + 1
        h = out[0] if isinstance(out, (tuple, list)) else out
        th = 0.5 * math.pi * min(state["t"], n_steps - 1) / max(1, int(n_steps) - 1)
        # The layers of a model split over two GPUs live on different devices.
        u = (math.cos(th) * a + math.sin(th) * b).to(device=h.device, dtype=h.dtype)
        h2 = h + sigma * h.norm(dim=-1, keepdim=True) * u
        return (h2, *out[1:]) if isinstance(out, (tuple, list)) else h2

    blocks = model._get_transformer_blocks()
    return state, [blocks[int(l)].register_forward_hook(push) for l in layers]


def generate_after_hidden(model, prompt, kind: str, seed: int, n_tokens: int = 32,
                          layers=(), max_new_tokens: int = 500, max_words=None,
                          temperature: float = 1.0, extra: int = 24, story_index=None) -> str:
    """An untouched story after a hidden opening that is generated, not given.

    ``self``: the untouched model writes the opening itself (``n_tokens``, run on
    to the next sentence end, at most ``extra`` more), with no noise -- whether
    the daydream needs its noise at all.

    ``latent:SIGMA``: a latent daydream. For ``n_tokens`` steps the model's input
    is its own expected next-token embedding (the top-50 probabilities times
    their input embeddings), never a token, while every layer in ``layers``
    adds a push of SIGMA times the hidden state's own norm, along a direction
    drawn for this story that turns slowly across the daydream. The story is
    then written without noise, attending to that trajectory of embeddings
    (recomputed noise-free) and a paragraph break. Nothing of the daydream is
    ever decoded, so a push strong enough to garble text cannot write garble
    into the context.
    """
    tok, m = model.tokenizer, model.model
    dev = model._input_device()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    chat = model.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    ids = tok(chat, return_tensors="pt")["input_ids"].to(dev)
    brk = torch.tensor([tok("\n\n", add_special_tokens=False)["input_ids"]], device=dev)
    sampling = model._sampling_kwargs(do_sample=True, temperature=temperature, top_p=None,
                                      top_k=None)
    pad = {"pad_token_id": tok.pad_token_id if tok.pad_token_id is not None
           else tok.eos_token_id}
    E = m.get_input_embeddings()

    premise = kind.startswith("premise:")
    if premise:
        # A one-sentence story idea, written after a fixed cue at the start of
        # the reply and hidden with it: the plot is decided in a few committed
        # words, which is where the push acts (sigma 0: the model's own idea).
        cue = torch.tensor([tok("Story idea:", add_special_tokens=False)["input_ids"]], device=dev)
        ids = torch.cat([ids, cue], dim=-1)
    if kind == "self" or kind.startswith("noisy:") or premise:
        handles, state = [], None
        if (kind.startswith("noisy:") or premise) and float(kind.split(":", 1)[1]) > 0:
            # The opening is sampled as usual, with the push on top of it; the
            # prompt is read and the story written without it.
            state, handles = _turning_push(model, seed, float(kind.split(":", 1)[1]),
                                           int(n_tokens), layers)

            class _Step(LogitsProcessor):
                def __call__(self_, input_ids, scores):
                    state["t"] += 1
                    return scores

            state["on"] = False
        try:
            if state is not None:
                # On for the opening's own steps only: the prompt's forward
                # pass is the first call, so switch on after it.
                from transformers import LogitsProcessorList
                first = {"done": False}

                def on_first(mod, args, kwargs):
                    if first["done"]:
                        state["on"] = True
                    first["done"] = True
                    return None

                handles.append(m.register_forward_pre_hook(on_first, with_kwargs=True))
                extra_kw = {"logits_processor": LogitsProcessorList([_Step()])}
            else:
                extra_kw = {}
            out = m.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                             max_new_tokens=int(n_tokens) + int(extra), **sampling, **pad,
                             **extra_kw)
        finally:
            if state is not None:
                state["on"] = False
            for hd in handles:
                hd.remove()
        hid = out[0, ids.shape[-1]:]
        if premise:
            n_tokens = int(cue.shape[-1]) + 6
            ids, hid = ids[:, :-cue.shape[-1]], torch.cat([cue[0], hid])
        eos = tok.eos_token_id
        if eos is not None and (hid == eos).any():
            hid = hid[:int((hid == eos).nonzero()[0])]
        cut = len(hid)
        for j in range(min(int(n_tokens), len(hid)), len(hid) + 1):
            if j > 0 and _sentence_end(tok, hid[:j].tolist()):
                cut = j
                break
        hid = hid[:cut]
        model.last_hidden = tok.decode(hid, skip_special_tokens=True).strip()
        if story_index is not None and int(story_index) < 3:
            print(f"  [hidden {kind}] story {story_index}: "
                  f"{' '.join(tok.decode(hid).split())[:240]}", flush=True)
        full = torch.cat([ids, hid[None].to(dev), brk], dim=-1)
        stopper = model._word_budget_stopper(full.shape[-1], max_words)
        out = m.generate(input_ids=full, attention_mask=torch.ones_like(full),
                         max_new_tokens=max_new_tokens, **sampling, **pad,
                         **({"stopping_criteria": stopper} if stopper is not None else {}))
        return tok.decode(out[0, full.shape[-1]:], skip_special_tokens=True).strip()

    if not kind.startswith("latent:"):
        raise ValueError(f"unknown generated hidden context {kind!r}")
    W = E.weight
    state, handles = _turning_push(model, seed, float(kind.split(":", 1)[1]), int(n_tokens),
                                   layers)
    softs = []
    try:
        with torch.no_grad():
            o = m(input_ids=ids, use_cache=True)
            past, logits = o.past_key_values, o.logits[:, -1].float()
            state["on"] = True
            for t in range(int(n_tokens)):
                state["t"] = t
                p = torch.softmax(logits / float(temperature), dim=-1)
                v, ix = p.topk(50, dim=-1)
                v = (v / v.sum(-1, keepdim=True)).to(device=W.device, dtype=W.dtype)
                e = (v[..., None] * W[ix.to(W.device)]).sum(1)
                softs.append(e)
                o = m(inputs_embeds=e[:, None, :], past_key_values=past, use_cache=True)
                past, logits = o.past_key_values, o.logits[:, -1].float()
    finally:
        state["on"] = False
        for hd in handles:
            hd.remove()
    if story_index is not None and int(story_index) < 3:
        near = [tok.decode([int((W @ s[0].to(W.device)).argmax())]) for s in softs]
        print(f"  [hidden latent] story {story_index}, nearest words: {''.join(near)[:240]!r}",
              flush=True)
    with torch.no_grad():
        emb = torch.cat([E(ids.to(W.device)), torch.stack(softs, dim=1).to(W.device),
                         E(brk.to(W.device))], dim=1)
    stopper = model._word_budget_stopper(0, max_words)
    out = m.generate(inputs_embeds=emb, attention_mask=torch.ones(emb.shape[:2], device=emb.device,
                                                                    dtype=torch.long),
                     max_new_tokens=max_new_tokens, **sampling, **pad,
                     **({"stopping_criteria": stopper} if stopper is not None else {}))
    return tok.decode(out[0], skip_special_tokens=True).strip()


def generate_selected_opening(model, prompt, seed: int, k: int, sigma: float, layers,
                              n_tokens: int = 32, max_new_tokens: int = 500, max_words=None,
                              temperature: float = 1.0, slack: float = 1.0,
                              story_index=None) -> str:
    """A visible opening chosen from ``k`` noise paths by where they took the model.

    The untouched model writes its default opening; ``k`` noise paths (the turning
    push of ``sigma`` x the hidden state's norm at ``layers``, a different
    direction each) write alternatives. Each opening is read once without noise:
    its plan state is the mean hidden state over the opening at a layer two
    thirds of the way up, its fluency the mean log-probability of its tokens.
    Among openings no more than ``slack`` nats per token less fluent than the
    default, the one whose plan state is farthest (cosine) from the default's is
    kept, and the story continues from it without noise. The opening is part of
    the returned story.
    """
    tok, m = model.tokenizer, model.model
    dev = model._input_device()
    chat = model.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    ids = tok(chat, return_tensors="pt")["input_ids"].to(dev)
    n_in = int(ids.shape[-1])
    sampling = model._sampling_kwargs(do_sample=True, temperature=temperature, top_p=None,
                                      top_k=None)
    pad = {"pad_token_id": tok.pad_token_id if tok.pad_token_id is not None
           else tok.eos_token_id}
    blocks = model._get_transformer_blocks()
    plan_layer = max(1, int(round(2 * len(blocks) / 3)))

    def opening(s, push):
        torch.manual_seed(int(s))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(s))
        handles, state, extra = [], None, {}
        if push:
            state, handles = _turning_push(model, s, sigma, int(n_tokens), layers)
            from transformers import LogitsProcessorList

            class _Step(LogitsProcessor):
                def __call__(self_, input_ids, scores):
                    state["t"] += 1
                    return scores

            first = {"done": False}

            def on_first(mod, args, kwargs):
                if first["done"]:
                    state["on"] = True
                first["done"] = True
                return None

            handles.append(m.register_forward_pre_hook(on_first, with_kwargs=True))
            extra = {"logits_processor": LogitsProcessorList([_Step()])}
        try:
            out = m.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                             max_new_tokens=int(n_tokens), **sampling, **pad, **extra)
        finally:
            if state is not None:
                state["on"] = False
            for hd in handles:
                hd.remove()
        return out[0, n_in:]

    def read(op):
        full = torch.cat([ids[0], op.to(dev)])[None]
        with torch.no_grad():
            o = m(input_ids=full, output_hidden_states=True)
        lp = torch.log_softmax(o.logits[0, n_in - 1:-1].float(), dim=-1)
        flu = float(lp.gather(-1, full[0, n_in:, None].to(lp.device)).mean())
        h = o.hidden_states[plan_layer][0, n_in:].float().mean(0)
        return h, flu

    default = opening(seed, push=False)
    h0, f0 = read(default)
    best, best_d, rows = default, -1.0, []
    for j in range(int(k)):
        op = opening(int(seed) * 1009 + j + 1, push=True)
        h, f = read(op)
        d = float(1 - torch.nn.functional.cosine_similarity(h, h0.to(h.device), dim=0))
        rows.append((round(d, 3), round(f - f0, 2)))
        if f >= f0 - slack and d > best_d:
            best, best_d = op, d
    if story_index is not None and int(story_index) < 3:
        print(f"  [selected opening] story {story_index}: distance, fluency vs default "
              f"{rows}; kept {'the default' if best_d < 0 else f'distance {best_d:.3f}'}: "
              f"{' '.join(tok.decode(best, skip_special_tokens=True).split())[:200]}",
              flush=True)
    full = torch.cat([ids[0], best.to(dev)])[None]
    stopper = model._word_budget_stopper(n_in, max_words)
    torch.manual_seed(int(seed))
    out = m.generate(input_ids=full, attention_mask=torch.ones_like(full),
                     max_new_tokens=max(1, int(max_new_tokens) - int(best.numel())), **sampling,
                     **pad, **({"stopping_criteria": stopper} if stopper is not None else {}))
    return tok.decode(out[0, n_in:], skip_special_tokens=True).strip()


_WEIGHT_BACKUP: dict = {}


def _down_projections(model, layers):
    """The linear maps in ``layers`` that write back into the residual stream from
    a wider space (the MLP output projection); any residual-width output if none."""
    d = int(model.model.config.hidden_size)
    blocks = model._get_transformer_blocks()
    out = []
    for l in layers:
        lins = [m for m in blocks[int(l)].modules() if isinstance(m, torch.nn.Linear)
                and m.out_features == d]
        wide = [m for m in lins if m.in_features > d]
        out += wide or lins
    return out


class weight_noise:
    """Context manager: the MLP output projections in ``layers`` perturbed by
    Gaussian noise of norm ``rho`` x the weight's own (drawn from ``seed``), and
    restored exactly on exit from a copy kept on the CPU."""

    def __init__(self, model, seed: int, rho: float, layers):
        self.mats = _down_projections(model, layers)
        self.seed, self.rho = int(seed), float(rho)

    def __enter__(self):
        for lin in self.mats:
            if id(lin) not in _WEIGHT_BACKUP:
                _WEIGHT_BACKUP[id(lin)] = lin.weight.detach().to("cpu", copy=True)
        with torch.no_grad():
            for i, lin in enumerate(self.mats):
                W = lin.weight
                g = torch.Generator(device=W.device).manual_seed(
                    (self.seed * 7919 + i) & 0x7FFFFFFF)
                noise = torch.randn(W.shape, generator=g, device=W.device, dtype=torch.float32)
                noise.mul_(self.rho * float(W.float().norm()) / float(noise.norm()))
                W.add_(noise.to(W.dtype))
                del noise
        return self

    def __exit__(self, *exc):
        with torch.no_grad():
            for lin in self.mats:
                lin.weight.copy_(_WEIGHT_BACKUP[id(lin)].to(lin.weight.device))
        return False


def generate_with_weight_noise(model, prompt, seed: int, rho: float, layers,
                               max_new_tokens: int = 500, max_words=None,
                               temperature: float = 1.0) -> str:
    """An untouched story written by a per-story perturbed copy of the model.

    Each MLP output projection in ``layers`` gets Gaussian noise drawn from this
    story's seed, scaled so its norm is ``rho`` times the weight's own. A push
    added to the hidden state shifts every next-word score by about the same
    amount; a change to the weights changes how the model responds to context,
    so what the model itself finds fluent moves with it. The original weights
    are restored exactly (from a copy kept on the CPU) after the story.
    """
    mats = _down_projections(model, layers)
    for i, lin in enumerate(mats):
        key = id(lin)
        if key not in _WEIGHT_BACKUP:
            _WEIGHT_BACKUP[key] = lin.weight.detach().to("cpu", copy=True)
    try:
        with torch.no_grad():
            for i, lin in enumerate(mats):
                W = lin.weight
                g = torch.Generator(device=W.device).manual_seed((int(seed) * 7919 + i) & 0x7FFFFFFF)
                noise = torch.randn(W.shape, generator=g, device=W.device, dtype=torch.float32)
                noise.mul_(float(rho) * float(W.float().norm()) / float(noise.norm()))
                W.add_(noise.to(W.dtype))
                del noise
        return model.generate(prompt, max_new_tokens=max_new_tokens, do_sample=True,
                              temperature=temperature, seed=seed, max_words=max_words)
    finally:
        with torch.no_grad():
            for lin in mats:
                lin.weight.copy_(_WEIGHT_BACKUP[id(lin)].to(lin.weight.device))


# --------------------------------------------------------------------------- #
#  A random context's meaning, placed where the reply is planned               #
# --------------------------------------------------------------------------- #

# How many sentences of the pool make the average state taken off every draw.
_MEAN_POOL = 64


@torch.no_grad()
def content_state(model, text: str, layer: int) -> torch.Tensor:
    """The hidden state after block ``layer`` while the model reads ``text``
    as plain text, averaged over its tokens (the first left out: it carries the
    huge attention-sink activations, not the text's meaning)."""
    blocks = model._get_transformer_blocks()
    box = {}

    def grab(module, inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        box["h"] = t[0].detach().float()

    ids = model.tokenizer(text, return_tensors="pt")["input_ids"].to(model._input_device())
    h = blocks[int(layer)].register_forward_hook(grab)
    try:
        model.model(input_ids=ids, use_cache=False)
    finally:
        h.remove()
    s = box["h"]
    return (s[1:] if s.shape[0] > 1 else s).mean(0)


def _mean_state(model, layer: int) -> torch.Tensor:
    cache = getattr(model, "_transplant_mean", None) or {}
    if layer not in cache:
        lines = [s for s in SENTENCES.read_text().splitlines() if s.strip()][:_MEAN_POOL]
        cache[layer] = torch.stack([content_state(model, s, layer) for s in lines]).mean(0)
        model._transplant_mean = cache
    return cache[layer]


def transplant_delta(model, seed: int, layer: int, alpha: float, iso: bool = False):
    """This story's push: ``alpha`` times how far a random sentence's state at
    ``layer`` sits from the average sentence's -- a direction the model reads
    as meaning, because real text put it there. With ``iso``, a random
    direction of the same length instead (the control)."""
    text = hidden_text(model, "sentence", seed)
    v = content_state(model, text, layer) - _mean_state(model, layer)
    if iso:
        g = torch.Generator().manual_seed((int(seed) * 104729 + 17) & 0x7FFFFFFF)
        r = torch.randn(v.shape, generator=g).to(v.device)
        v = r * (v.norm() / r.norm())
    return float(alpha) * v, text


def generate_with_transplant(model, prompt, seed: int, alpha: float, layers, *,
                             n_positions: int = 8, iso: bool = False,
                             max_new_tokens: int = 500, max_words=None,
                             temperature: float = 1.0, story_index=None) -> str:
    """An untouched story whose plan is formed from a shifted reading of the request.

    The push is added once, while the prompt is read, after one middle block
    (the middle of ``layers``), at the last ``n_positions`` prompt positions --
    where models settle what the whole reply will be (Dong et al., ICML 2025) --
    and nowhere else. Every word is then written by the untouched model,
    attending to that shifted memory of the request. The push is the state a
    random Wikipedia sentence (the same one the hidden-sentence control shows
    the model as text) leaves at that block, minus the average sentence's: a
    random point of the model's own meaning space, never shown as words, so
    there is nothing to copy, continue or write about.
    """
    layers = list(layers)
    layer = int(layers[len(layers) // 2])
    delta, text = transplant_delta(model, seed, layer, alpha, iso=iso)
    blocks = model._get_transformer_blocks()
    seen = {}

    def push(module, inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        if t.dim() != 3 or t.shape[1] <= 1 or seen.get("done"):
            return None
        n = min(int(n_positions), t.shape[1])
        with torch.no_grad():
            seen["ratio"] = float(delta.norm() / t[0, -n:].float().norm(dim=-1).mean())
            t[:, -n:, :] += delta.to(device=t.device, dtype=t.dtype)
        seen["done"] = True
        return None

    h = blocks[layer].register_forward_hook(push)
    try:
        out = model.generate(prompt, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=temperature, seed=seed, max_words=max_words)
    finally:
        h.remove()
    if story_index is not None and int(story_index) < 3:
        print(f"  [transplant{' iso' if iso else ''} {alpha:g}] story {story_index}: layer "
              f"{layer}, last {n_positions} prompt positions, push {seen.get('ratio', 0):.2f} x "
              f"the state there; from: {text[:120]}", flush=True)
    return out


# --------------------------------------------------------------------------- #
#  A random direction grown into one the model amplifies                       #
# --------------------------------------------------------------------------- #

class _Stop(Exception):
    pass


# Set by the runner (--amplify R:STEPS): the steered arms then carry this story's
# amplified direction (radius, steps); None = off.
AMPLIFY = None
# How long that direction lasts while writing (--amplify-fade): None = throughout.
AMPLIFY_FADE = None
# Caps on the grown direction (--amplify-caps MOVE:KL): None = no caps.
AMPLIFY_CAPS = None
# Where and how it is applied (--amplify-where): push only the last K prompt
# positions (0 = all after the first), and rotate each state toward the push
# keeping its length instead of adding (norm).
AMPLIFY_LASTK = 0
AMPLIFY_NORM = False


def apply_push(t: torch.Tensor, v: torch.Tensor, positions, norm: bool) -> torch.Tensor:
    """``t`` (batch, seq, d) with ``v`` applied at ``positions`` (a slice): added,
    or with ``norm`` the state turned toward state + v keeping its own length
    (Angular / Spherical Steering: meaning rides on direction, length changes
    are what break fluency). Returns a new tensor."""
    out = t.clone()
    h = out[:, positions]
    new = h + v
    if norm:
        new = new * (h.norm(dim=-1, keepdim=True) / new.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    out[:, positions] = new.to(out.dtype)
    return out


def amplified_direction(model, ids, seed: int, radius: float, steps: int, src: int, tgt: int,
                        n_last: int = 8, lr: float = 0.5, lock: float = 0.0, caps=None,
                        lastk: int = 0, norm: bool = False):
    """A per-story direction at block ``src`` that later blocks amplify.

    Random noise is corrected away by the layers after it; directions toward
    other real states are not (arXiv 2410.12555, 2606.24964). Following MELBO
    (Mack and Turner, 2024): start from a random direction drawn from ``seed``,
    of length ``radius`` x the hidden state's mean size at ``src``, added at every
    prompt position after the first, and take ``steps`` gradient steps on the
    sphere of that radius to make the state at block ``tgt`` over the last
    ``n_last`` prompt positions -- where the reply is planned -- move as far as
    possible from where it is without the push. Only the prompt is read; nothing
    is generated and no example text is used. ``steps`` 0 is the random start
    itself (the control). With ``lock`` > 0 each step also pays ``lock`` x the
    change (KL, nats) in the model's chances for the first word of its reply,
    so the direction cannot grow by switching what kind of reply it gives (an
    essay, a refusal) -- each kind starts differently. Returns (direction, how
    far it moved the planning state at the start and at the end, relative to
    that state's size); with a lock the first-word change at the end is kept
    in ``model._amplify_kl``.

    ``caps`` = (most planning movement, most first-word change in nats): a
    step that would leave either cap is refused and the step halved, and a
    direction still outside them at the end (a random start already beyond
    them) is shortened until it is inside -- some random starts otherwise run
    away into text the model cannot write ("TheTheThe...", word salad).
    """
    blocks = model._get_transformer_blocks()
    for p in model.model.parameters():
        p.requires_grad_(False)
    box = {}

    def grab_src(module, inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        box["src"] = t[0, 1:].detach().float()

    def grab_tgt(module, inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        box["tgt"] = t[0, -int(n_last):]
        if not lock:
            raise _Stop

    def run(theta=None):
        hs = [blocks[int(tgt)].register_forward_hook(grab_tgt)]
        if theta is None:
            hs.append(blocks[int(src)].register_forward_hook(grab_src))
        else:
            def add(module, inp, out):
                t = out[0] if isinstance(out, (tuple, list)) else out
                pos = slice(-int(lastk), None) if lastk else slice(1, None)
                t2 = apply_push(t, theta.to(device=t.device, dtype=t.dtype), pos, norm)
                return (t2, *out[1:]) if isinstance(out, (tuple, list)) else t2
            hs.append(blocks[int(src)].register_forward_hook(add))
        try:
            o = model.model(input_ids=ids, use_cache=False)
            if lock:
                box["first"] = torch.log_softmax(o.logits[0, -1].float(), dim=-1)
        except _Stop:
            pass
        finally:
            for h in hs:
                h.remove()
        return box["tgt"]

    with torch.no_grad():
        clean = run().float().detach()
        first_clean = box["first"].detach() if lock else None

    def kl_first():
        lp = box["first"]
        return (first_clean.exp() * (first_clean - lp)).sum()
    R = float(radius) * float(box["src"].norm(dim=-1).mean())
    scale = float(clean.norm(dim=-1).mean())
    g = torch.Generator().manual_seed((int(seed) * 15485863 + 3) & 0x7FFFFFFF)
    theta = torch.randn(box["src"].shape[-1], generator=g).to(box["src"].device)
    theta = theta * (R / theta.norm())

    def moved(th):
        with torch.no_grad():
            return float((run(th).float() - clean).norm(dim=-1).mean()) / scale

    first = moved(theta)
    if lock:
        with torch.no_grad():
            run(theta)
            model._amplify_kl = [float(kl_first())]

    def inside(th):
        """Planning movement and first-word change of ``th``, and whether both
        are within the caps."""
        mv = moved(th)
        kl = float(kl_first()) if lock else 0.0
        return mv, kl, (mv <= caps[0] and kl <= caps[1])

    step = float(lr)
    for _ in range(int(steps)):
        th = theta.clone().requires_grad_(True)
        with torch.enable_grad():
            loss = (run(th).float() - clean).pow(2).sum(-1).mean() / scale ** 2
            if lock:
                # The first word's chances come off the last layer, which
                # can sit on another GPU than the planning layer.
                loss = loss - float(lock) * kl_first().to(loss.device)
            loss.backward()
        grad = th.grad
        if grad is None or not torch.isfinite(grad).all() or float(grad.norm()) == 0:
            break
        grad = grad - (grad @ theta) / (theta @ theta) * theta
        cand = theta + step * R * grad / grad.norm()
        cand = cand * (R / cand.norm())
        if caps is not None:
            _, _, ok = inside(cand)
            if not ok:
                step *= 0.5
                continue
        theta = cand
    if caps is not None:
        mv, kl, ok = inside(theta)
        shrink = 1.0
        while not ok and shrink > 1 / 64:
            shrink *= 0.7
            mv, kl, ok = inside(theta * shrink)
        theta = theta * shrink
        model._amplify_shrink = shrink
    if lock:
        with torch.no_grad():
            run(theta)
            model._amplify_kl.append(float(kl_first()))
    return theta.detach(), first, (moved(theta) if steps else first)


def generate_with_amplified(model, prompt, seed: int, radius: float, steps: int, layers, *,
                            max_new_tokens: int = 500, max_words=None,
                            temperature: float = 1.0, story_index=None) -> str:
    """An untouched story written with this story's amplified direction added at
    the first steering layer, at every position after the first (prompt and
    story), the way MELBO's vectors are applied. The direction is grown toward
    the planning positions at two thirds of the depth."""
    layers = list(layers)
    src = int(layers[0])
    tgt = max(src + 1, int(round(2 * len(model._get_transformer_blocks()) / 3)))
    chat = model.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    ids = model.tokenizer(chat, return_tensors="pt")["input_ids"].to(model._input_device())
    theta, first, last = amplified_direction(model, ids, seed, radius, steps, src, tgt)
    blocks = model._get_transformer_blocks()
    state = {"prefill": True}

    def add(module, inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        with torch.no_grad():
            v = theta.to(device=t.device, dtype=t.dtype)
            if state["prefill"]:
                t[:, 1:] += v
                state["prefill"] = False
            else:
                t += v
        return None

    h = blocks[src].register_forward_hook(add)
    try:
        out = model.generate(prompt, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=temperature, seed=seed, max_words=max_words)
    finally:
        h.remove()
    if story_index is not None and int(story_index) < 3:
        print(f"  [amplified {radius:g}x, {steps} steps] story {story_index}: layer {src} -> "
              f"{tgt}; moves the planning state {first:.3f} -> {last:.3f} of its size",
              flush=True)
    return out
