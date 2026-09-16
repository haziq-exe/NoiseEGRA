"""Structural and syntactic diversity, computed from the words alone.

One embedding-based Vendi score cannot say *what kind* of variety it is seeing:
two stories can differ because different things happen in them, or because the
same things are phrased differently. Temperature mostly buys the second kind;
the point of a prompt-level perturbation is the first. These two scores separate
them, and both run locally from saved stories -- spaCy only, no embedding model,
no GPU.

**Structural diversity** represents each story as the *set* of content lemmas in
it -- what is mentioned and done, order and phrasing removed -- and takes the
Vendi score over those set vectors. Two stories about a cat chasing a ball land
together however differently they are worded; a story about a sandcastle washed
away by the tide lands apart. Proper nouns are excluded on purpose: "Mia runs to
the park" and "Tom runs to the park" are the same story with the name swapped,
and a name swap is not structural variety. Presence, not counts: a story that
says "cat" twenty times is no further from one that says it once.

**Syntactic diversity** represents each story by its distribution of
part-of-speech trigrams -- the shapes of its sentences, the words removed. This
is the surface-level complement: raised temperature can move it while leaving
the structural score alone, and a phrasing-only intervention should move nothing
else.

Both scores are as length-sensitive as any diversity measure, so they take the
same word truncation as the embedding Vendi and must only be compared at
matched truncation, on groups of matched size. Degenerate text inflates these
scores less than it inflates the embedding Vendi (a loop has few lemmas), but
filtering first is still the honest order: score what a filter keeps, and report
the drop rate beside the score.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .diversity import truncate_words, vendi_from_embeddings

# Which parts of speech carry the story. PROPN is deliberately absent (see the
# module docstring); pronouns, determiners and auxiliaries carry no content.
_CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}


class _Nlp:
    """One spaCy pipeline with the lemmatizer on, shared by both scores.

    Separate from the one in ``constraint_metrics_en``, which disables the
    lemmatizer; loading a second pipeline once is cheaper than lemmatizing by
    hand badly.
    """

    _nlp = None
    _failed = False

    @classmethod
    def get(cls):
        if cls._nlp is not None:
            return cls._nlp
        if cls._failed:
            return None
        try:
            import spacy
        except Exception:
            cls._failed = True
            return None
        for name in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"):
            try:
                cls._nlp = spacy.load(name, disable=["parser", "ner"])
                return cls._nlp
            except Exception:
                continue
        cls._failed = True
        return None


def _docs(texts: Sequence[str], truncate: Optional[int]):
    nlp = _Nlp.get()
    if nlp is None:
        raise RuntimeError(
            "structural diversity needs spaCy with an English model; "
            "run `python -m spacy download en_core_web_sm`."
        )
    cut = [truncate_words(t, truncate) for t in texts]
    return list(nlp.pipe(cut, batch_size=64))


def content_lemma_sets(
    texts: Sequence[str], truncate: Optional[int] = 40
) -> List[frozenset]:
    """The set of content lemmas per story, lowercased, stopwords removed."""
    out = []
    for doc in _docs(texts, truncate):
        lemmas = {
            t.lemma_.lower()
            for t in doc
            if t.pos_ in _CONTENT_POS and t.is_alpha and not t.is_stop
        }
        out.append(frozenset(lemmas))
    return out


def _set_vectors(sets: Sequence[frozenset]) -> np.ndarray:
    """Binary presence vectors over the union vocabulary, L2-normalised.

    A story with no content words at all gets its own reserved dimension rather
    than a zero vector, so the kernel stays valid; such a story is broken and
    should have been filtered before scoring, but a metric must not crash on it.
    """
    vocab = sorted(set().union(*sets)) if sets else []
    index = {w: i for i, w in enumerate(vocab)}
    dim = len(vocab) + 1
    m = np.zeros((len(sets), dim), dtype=np.float64)
    for row, s in enumerate(sets):
        if not s:
            m[row, -1] = 1.0
            continue
        for w in s:
            m[row, index[w]] = 1.0
        m[row] /= np.linalg.norm(m[row])
    return m


def pos_trigram_vectors(
    texts: Sequence[str], truncate: Optional[int] = 40
) -> np.ndarray:
    """L2-normalised part-of-speech trigram frequency vectors, one per story."""
    grams_per_text: List[dict] = []
    vocab = {}
    for doc in _docs(texts, truncate):
        tags = [t.pos_ for t in doc if not t.is_space]
        counts: dict = {}
        for i in range(len(tags) - 2):
            g = (tags[i], tags[i + 1], tags[i + 2])
            counts[g] = counts.get(g, 0) + 1
            if g not in vocab:
                vocab[g] = len(vocab)
        grams_per_text.append(counts)
    m = np.zeros((len(texts), len(vocab) + 1), dtype=np.float64)
    for row, counts in enumerate(grams_per_text):
        if not counts:
            m[row, -1] = 1.0
            continue
        for g, c in counts.items():
            m[row, vocab[g]] = float(c)
        m[row] /= np.linalg.norm(m[row])
    return m


def structural_vendi(
    texts: Sequence[str], truncate: Optional[int] = 40, q: float = 1.0
) -> float:
    """Effective number of structurally distinct stories (content-lemma sets)."""
    return vendi_from_embeddings(_set_vectors(content_lemma_sets(texts, truncate)), q=q)


def syntactic_vendi(
    texts: Sequence[str], truncate: Optional[int] = 40, q: float = 1.0
) -> float:
    """Effective number of syntactically distinct stories (POS-trigram profiles)."""
    return vendi_from_embeddings(pos_trigram_vectors(texts, truncate), q=q)


def rarefied_vendi(
    vectors: np.ndarray,
    group_size: int,
    *,
    draws: int = 12,
    q: float = 1.0,
    seed: int = 0,
) -> Tuple[float, float]:
    """Mean and spread of the Vendi score over random subsets of fixed size.

    Vendi grows with group size, so conditions that keep different numbers of
    stories after filtering cannot be compared raw. Every condition is resampled
    to the same ``group_size`` and the score averaged over ``draws`` subsets --
    the same correction ``distinct@m`` uses. Returns ``(mean, std)``.
    """
    n = vectors.shape[0]
    if n < 2 or group_size < 2:
        return 1.0, 0.0
    if group_size >= n:
        return vendi_from_embeddings(vectors, q=q), 0.0
    rng = np.random.default_rng(seed)
    scores = [
        vendi_from_embeddings(vectors[rng.choice(n, size=group_size, replace=False)], q=q)
        for _ in range(draws)
    ]
    return float(np.mean(scores)), float(np.std(scores))
