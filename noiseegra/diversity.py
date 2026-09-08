"""Length-matched Vendi and distinct-k.

Two problems with a plain Vendi score over generated stories:

1. **Length.** Longer stories cover more ground, so their embeddings sit further
   apart, so a set of long stories scores as more diverse than a set of short
   ones even when nothing about the writing changed. In our own alpha sweep,
   mean word count and Vendi correlated at r = +0.94, and once word count was
   held fixed nothing else predicted Vendi at all. ``truncate_words`` cuts every
   story to the same number of words before embedding, which removes the effect.

2. **Degradation reads as diversity.** Cosine spread cannot tell "ten different
   stories" from "ten broken outputs". ``distinct_k`` asks the different question:
   how many genuinely distinct stories are in this set of k? Ten retellings of
   one plot score 1. It is bounded by k, so it cannot run away as text degrades.

``distinct_k`` follows NoveltyBench (Zhang et al., 2025) in counting equivalence
classes rather than measuring spread, but approximates their trained partitioner
with threshold clustering over embeddings -- we have no human partition
annotations to train on. The threshold is calibrated from the data rather than
picked by hand: see ``calibrate_threshold``.
"""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_WORD = re.compile(r"\S+")


# --------------------------------------------------------------------------- #
#  Text handling                                                              #
# --------------------------------------------------------------------------- #

def word_count(text: str) -> int:
    return len(_WORD.findall(text or ""))


def truncate_words(text: str, n: Optional[int]) -> str:
    """First ``n`` whitespace-separated words of ``text`` (all of it if ``n`` is None)."""
    if n is None or n <= 0:
        return text
    parts = _WORD.findall(text or "")
    return " ".join(parts[:n])


def read_run_csv(path: Path | str) -> Tuple[List[str], List[int]]:
    """Return ``(stories, prompt_indices)`` from a run CSV.

    Handles the English runner's ``prompt_index,story_index,story`` format and the
    older headerless one-story-per-row files under ``experiment_results/``. Files
    with no prompt column are treated as a single group, which is correct for the
    Arabic runs: every story there came from the same prompt.
    """
    path = Path(path)
    stories: List[str] = []
    prompts: List[int] = []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames and "story" in reader.fieldnames:
            has_idx = "prompt_index" in reader.fieldnames
            for row in reader:
                text = (row.get("story") or "").strip()
                if not text:
                    continue
                stories.append(text)
                prompts.append(int(row["prompt_index"]) if has_idx else 0)
            return stories, prompts
    with path.open(encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if row and row[0].strip():
                stories.append(row[0].strip())
                prompts.append(0)
    return stories, prompts


def group_by_prompt(
    stories: Sequence[str], prompt_idx: Sequence[int], min_group: int = 2
) -> List[List[int]]:
    """Positions of the stories for each prompt, keeping groups of at least ``min_group``."""
    groups: Dict[int, List[int]] = defaultdict(list)
    for i, p in enumerate(prompt_idx):
        groups[p].append(i)
    return [ix for _, ix in sorted(groups.items()) if len(ix) >= min_group]


# --------------------------------------------------------------------------- #
#  Metrics over an embedding matrix                                            #
# --------------------------------------------------------------------------- #

def vendi_from_embeddings(emb: np.ndarray) -> float:
    """Vendi score of L2-normalised row vectors, under the cosine kernel.

    The exponential of the Shannon entropy of the eigenvalues of the similarity
    matrix, i.e. the effective number of distinct items. Ranges from 1 (all
    identical) to n (all mutually orthogonal).
    """
    n = emb.shape[0]
    if n < 2:
        return 1.0
    kernel = emb @ emb.T
    vals = np.linalg.eigvalsh(kernel / n)
    vals = np.clip(vals.real, 0.0, None)
    vals = vals[vals > 1e-12]
    if vals.size == 0:
        return 1.0
    entropy = float(-(vals * np.log(vals)).sum())
    return float(math.exp(entropy))


def _average_linkage_labels(sim: np.ndarray, threshold: float) -> np.ndarray:
    """Agglomerative average-linkage clustering; merge while similarity >= threshold.

    Groups are at most a few hundred stories, so the naive O(n^3) loop is fine and
    saves a scikit-learn dependency.
    """
    n = sim.shape[0]
    labels = np.arange(n)
    members: Dict[int, List[int]] = {i: [i] for i in range(n)}
    work = sim.astype(float).copy()
    np.fill_diagonal(work, -np.inf)
    sizes = {i: 1 for i in range(n)}
    active = set(range(n))

    while len(active) > 1:
        idx = list(active)
        sub = work[np.ix_(idx, idx)]
        flat = int(np.argmax(sub))
        a, b = idx[flat // len(idx)], idx[flat % len(idx)]
        if sub.max() < threshold:
            break
        # average linkage: the merged cluster's similarity to c is the
        # size-weighted mean of a's and b's
        na, nb = sizes[a], sizes[b]
        for c in active:
            if c in (a, b):
                continue
            merged = (na * work[a, c] + nb * work[b, c]) / (na + nb)
            work[a, c] = work[c, a] = merged
        members[a].extend(members.pop(b))
        sizes[a] = na + nb
        active.discard(b)
        work[b, :] = -np.inf
        work[:, b] = -np.inf

    for new_label, root in enumerate(sorted(active)):
        for i in members[root]:
            labels[i] = new_label
    return labels


def distinct_k(emb: np.ndarray, threshold: float) -> int:
    """Number of equivalence classes among ``emb`` rows at cosine similarity ``threshold``."""
    if emb.shape[0] == 0:
        return 0
    if emb.shape[0] == 1:
        return 1
    sim = emb @ emb.T
    return int(_average_linkage_labels(sim, threshold).max() + 1)


def calibrate_threshold(
    emb: np.ndarray, prompt_idx: Sequence[int], percentile: float = 99.0
) -> float:
    """Pick the "same story" similarity cut-off from known-different pairs.

    Two stories written from *different* WritingPrompts prompts are different
    stories by construction. Their similarities give a reference distribution for
    "distinct". The cut-off is the ``percentile``-th of that distribution: a
    within-prompt pair only counts as the same story if it is more similar than
    almost every known-distinct pair.

    Returns ``nan`` when there is only one prompt, in which case the caller must
    supply a threshold. Calibrate once over all conditions and reuse the number,
    otherwise the conditions are not being compared on the same scale.
    """
    p = np.asarray(prompt_idx)
    if np.unique(p).size < 2:
        return float("nan")
    sim = emb @ emb.T
    cross = sim[p[:, None] != p[None, :]]
    if cross.size == 0:
        return float("nan")
    return float(np.percentile(cross, percentile))


# --------------------------------------------------------------------------- #
#  Scorer                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class ConditionDiversity:
    n: int
    n_groups: int
    mean_words: float
    vendi_raw: float
    vendi_matched: float
    distinct_mean: float
    distinct_frac: float
    group_size: float
    plot_vendi: float = float("nan")
    plot_distinct: float = float("nan")
    extra: Dict[str, float] = field(default_factory=dict)


class DiversityScorer:
    """Embeds once, then reports raw Vendi, length-matched Vendi and distinct-k.

    ``truncate_to`` is the word budget every story is cut to before embedding for
    the length-matched score. Set it from the shortest condition in the
    comparison (``scripts/score_diversity.py --truncate-words auto`` does this).
    """

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        *,
        truncate_to: Optional[int] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        model=None,
    ):
        from .embeddings import DEFAULT_EMBEDDING_MODEL, resolve_embedding_model

        self.embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL
        self.hf_id = resolve_embedding_model(self.embedding_model)
        self.truncate_to = truncate_to
        self.batch_size = batch_size
        self._model = model
        self._device = device
        self._cache: Dict[Tuple[str, Optional[int]], np.ndarray] = {}

    @property
    def model(self):
        if self._model is None:
            from .embeddings import load_embedder

            self._model = load_embedder(self.embedding_model, device=self._device)
        return self._model

    def encode(self, texts: Sequence[str], truncate: Optional[int] = None) -> np.ndarray:
        """L2-normalised embeddings, optionally of the first ``truncate`` words only."""
        prepared = [truncate_words(t, truncate) for t in texts]
        missing = [t for t in prepared if (t, truncate) not in self._cache]
        if missing:
            uniq = list(dict.fromkeys(missing))
            vecs = self.model.encode(
                uniq,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            for t, v in zip(uniq, vecs):
                self._cache[(t, truncate)] = v
        return np.stack([self._cache[(t, truncate)] for t in prepared])

    def score(
        self,
        stories: Sequence[str],
        prompt_idx: Sequence[int],
        *,
        threshold: float,
        min_group: int = 2,
        truncate_to: Optional[int] = "unset",  # type: ignore[assignment]
    ) -> ConditionDiversity:
        """Per-prompt diversity for one condition, averaged over prompt groups.

        Diversity is computed **within** a prompt and then averaged. Stories from
        different prompts are trivially dissimilar, so pooling them would measure
        the prompt set rather than the model.
        """
        if truncate_to == "unset":
            truncate_to = self.truncate_to
        groups = group_by_prompt(stories, prompt_idx, min_group)
        mean_words = float(np.mean([word_count(s) for s in stories])) if stories else float("nan")
        if not groups:
            return ConditionDiversity(
                n=len(stories), n_groups=0, mean_words=mean_words,
                vendi_raw=float("nan"), vendi_matched=float("nan"),
                distinct_mean=float("nan"), distinct_frac=float("nan"), group_size=float("nan"),
            )

        emb_raw = self.encode(stories, truncate=None)
        emb_cut = self.encode(stories, truncate=truncate_to) if truncate_to else emb_raw

        raws, matched, dists, fracs, sizes = [], [], [], [], []
        for ix in groups:
            raws.append(vendi_from_embeddings(emb_raw[ix]))
            matched.append(vendi_from_embeddings(emb_cut[ix]))
            if not math.isnan(threshold):
                d = distinct_k(emb_cut[ix], threshold)
                dists.append(d)
                fracs.append(d / len(ix))
            sizes.append(len(ix))

        return ConditionDiversity(
            n=len(stories),
            n_groups=len(groups),
            mean_words=mean_words,
            vendi_raw=float(np.mean(raws)),
            vendi_matched=float(np.mean(matched)),
            distinct_mean=float(np.mean(dists)) if dists else float("nan"),
            distinct_frac=float(np.mean(fracs)) if fracs else float("nan"),
            group_size=float(np.mean(sizes)),
        )
