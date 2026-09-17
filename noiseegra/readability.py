"""Readability measured by something other than sentence length.

The Flesch-Kincaid grade level this project reports has exactly two inputs --
words per sentence and syllables per word -- and nothing else. That makes it
blind in ways that matter here:

* **It rewards chopping.** The same sentences cut into fragments score two full
  grades easier: "Mia walks to the park with her mother" (grade 0.2) becomes
  "Mia walks. To the park. With mother." (grade -2.3). Fragmentation is this
  method's characteristic failure, so the metric moves the wrong way on it.
* **It cannot see whether a word is known.** "The firm must pay its debt this
  week; the bank will seize the funds" scores -0.8, easier than a children's
  story, because its words are short.
* **It punishes long easy words.** "Grandmother celebrated my birthday" scores
  24.3, nominally post-doctoral, for a sentence a six-year-old understands.

And in this project it is close to circular: three of the fifteen requirements
(grade at most 2.5, sentences at most eight words, words at most two syllables)
push directly on its two inputs, so a method that satisfies the requirements
drives the metric down almost mechanically.

What it misses is vocabulary: whether the words are ones a young reader has met.
``uncommon_word_share`` measures that, as the share of a story's ordinary words
that are not among the commonest few thousand in English. It is the idea behind
the Dale-Chall formula's familiar-word list, built here from a corpus that can be
redistributed -- the list lives in ``data/common_words_en.json`` and its
provenance is recorded inside that file.

Read the two together. Sentence length and syllable count say how heavy the
surface is; the uncommon-word share says whether the words themselves are known.
A story that is easy on both is easy; one that is easy on the first alone may
just be chopped up.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import List, Sequence

_DATA = Path(__file__).resolve().parent / "data" / "common_words_en.json"


@lru_cache(maxsize=4)
def common_words(top_k: int = 3000) -> frozenset:
    """The ``top_k`` commonest English words, from the committed list."""
    with _DATA.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    return frozenset(payload["words"][:top_k])


def uncommon_word_share(
    texts: Sequence[str], top_k: int = 3000, skip_names: bool = True
) -> List[float]:
    """Share of each story's ordinary words that are not among the commonest.

    Proper nouns are skipped by default: a character called Mira is not a
    vocabulary burden, and the same choice is made by the structural diversity
    measure, so the two agree about what counts as content. Stories with no
    countable words return 0.0.

    Needs spaCy for the proper-noun test; without it, every alphabetic token is
    counted and the number is a little higher across the board, which is
    harmless as long as conditions are compared with the same setting.
    """
    from .structure import _Nlp

    vocab = common_words(top_k)
    nlp = _Nlp.get() if skip_names else None
    out: List[float] = []
    if nlp is None:
        for t in texts:
            words = [w.strip(".,!?\"';:()").lower() for w in t.split()]
            words = [w for w in words if w.isalpha()]
            out.append(sum(1 for w in words if w not in vocab) / len(words)
                       if words else 0.0)
        return out
    for doc in nlp.pipe(list(texts), batch_size=64):
        words = [t.text.lower() for t in doc
                 if t.is_alpha and not (skip_names and t.pos_ == "PROPN")]
        out.append(sum(1 for w in words if w not in vocab) / len(words)
                   if words else 0.0)
    return out
