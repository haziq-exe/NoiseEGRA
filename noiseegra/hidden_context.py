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

    if kind == "self":
        out = m.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                         max_new_tokens=int(n_tokens) + int(extra), **sampling, **pad)
        hid = out[0, ids.shape[-1]:]
        eos = tok.eos_token_id
        if eos is not None and (hid == eos).any():
            hid = hid[:int((hid == eos).nonzero()[0])]
        cut = len(hid)
        for j in range(min(int(n_tokens), len(hid)), len(hid) + 1):
            if j > 0 and _sentence_end(tok, hid[:j].tolist()):
                cut = j
                break
        hid = hid[:cut]
        if story_index is not None and int(story_index) < 3:
            print(f"  [hidden self] story {story_index}: {' '.join(tok.decode(hid).split())[:240]}",
                  flush=True)
        full = torch.cat([ids, hid[None].to(dev), brk], dim=-1)
        stopper = model._word_budget_stopper(full.shape[-1], max_words)
        out = m.generate(input_ids=full, attention_mask=torch.ones_like(full),
                         max_new_tokens=max_new_tokens, **sampling, **pad,
                         **({"stopping_criteria": stopper} if stopper is not None else {}))
        return tok.decode(out[0, full.shape[-1]:], skip_special_tokens=True).strip()

    if not kind.startswith("latent:"):
        raise ValueError(f"unknown generated hidden context {kind!r}")
    sigma = float(kind.split(":", 1)[1])
    W = E.weight
    g = torch.Generator().manual_seed(int(seed) & 0x7FFFFFFF)
    a = torch.randn(W.shape[1], generator=g)
    b = torch.randn(W.shape[1], generator=g)
    b = b - (b @ a) / (a @ a) * a
    a, b = (a / a.norm()).to(dev, W.dtype), (b / b.norm()).to(dev, W.dtype)
    state = {"on": False, "t": 0}

    def push(mod, inp, out):
        if not state["on"]:
            return None
        h = out[0] if isinstance(out, (tuple, list)) else out
        th = 0.5 * math.pi * state["t"] / max(1, int(n_tokens) - 1)
        # The layers of a model split over two GPUs live on different devices.
        u = (math.cos(th) * a + math.sin(th) * b).to(device=h.device, dtype=h.dtype)
        h2 = h + sigma * h.norm(dim=-1, keepdim=True) * u
        return (h2, *out[1:]) if isinstance(out, (tuple, list)) else h2

    blocks = model._get_transformer_blocks()
    handles = [blocks[int(l)].register_forward_hook(push) for l in layers]
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
