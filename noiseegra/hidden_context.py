"""Hidden context at the start of the reply: a control for the hidden daydream.

The daydream writes a short opening under strong noise and hides it. These
controls hide something that involved no noise at all -- random words, or a
random coherent sentence -- before an untouched story, to tell whether the
daydream's effect needs the noise or only some unrelated text in front of the
story.
"""
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
