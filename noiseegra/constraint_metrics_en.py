"""Exact, judge-free checks for the four steered English constraints.

    length           word count <= N                        (exact)
    present_tense    fraction of finite verbs in the         (POS-tagged)
                     present >= threshold
    simple_register  Flesch-Kincaid grade level <= N         (exact formula)
    dialogue         contains at least one quoted utterance  (exact)

Three of the four mirror the Arabic constraints from the original study, so the
cross-lingual claim is about the same constraint types. ``dialogue`` is added
because it is an *inclusion* constraint -- it asks for something to be present
rather than limiting something -- which tests whether steering works in both
directions.

Tense backends
--------------
``spacy``  part-of-speech tagging (``pip install spacy`` and
           ``python -m spacy download en_core_web_sm``). This is the backend to
           use for anything you intend to publish.
``regex``  a suffix-and-irregular-list fallback that needs no model. English
           irregular verbs make this genuinely approximate; it is here so the
           module always runs, not because it is good. It prints a warning.
"""

from __future__ import annotations

import csv
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

DEFAULT_MAX_WORDS = 150
DEFAULT_PRESENT_RATIO = 0.8
DEFAULT_MAX_GRADE = 6.0

CONSTRAINT_NAMES = ("length", "present_tense", "simple_register", "dialogue")

_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SENT_SPLIT = re.compile(r"(?<=[.!?])[\s\"”’)]+|\n+")
# A quoted span of at least a couple of words, straight or curly quotes.
_QUOTED = re.compile(r"[\"“][^\"“”]*?\b\w+\b[^\"“”]*?[\"”]")

_VOWEL_GROUP = re.compile(r"[aeiouy]+")

_IRREGULAR_PAST = frozenset("""
was were had did said went took saw came got made knew thought found told became
left felt put brought began kept held wrote stood heard let meant set met ran paid
sat spoke lay led grew lost fell sent built understood drew broke spent rose drove
bought wore chose ate gave took slept swam sang rang drank shook threw flew forgot
hid rode wrote won taught caught bit blew froze stole tore woke bore dug hung struck
""".split())
_PRESENT_AUX = frozenset("is are am has have does do can will shall may must".split())


def tokenize_words(text: str) -> List[str]:
    return _WORD.findall(text)


def split_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text)]
    return [p for p in parts if p and _WORD.search(p)]


def count_syllables(word: str) -> int:
    """Standard vowel-group heuristic with silent-e correction."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = _VOWEL_GROUP.findall(w)
    n = len(groups)
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and n > 1:
        n -= 1
    if w.endswith("le") and len(w) > 2 and w[-3] not in "aeiouy":
        pass  # 'table', 'little' keep the final syllable
    return max(n, 1)


def flesch_kincaid_grade(words: Sequence[str], n_sentences: int) -> float:
    """0.39*(words/sentence) + 11.8*(syllables/word) - 15.59."""
    if not words or n_sentences <= 0:
        return 0.0
    syllables = sum(count_syllables(w) for w in words)
    return (
        0.39 * (len(words) / n_sentences)
        + 11.8 * (syllables / len(words))
        - 15.59
    )


class _SpacyTense:
    _nlp = None
    _failed = False

    @classmethod
    def available(cls) -> bool:
        if cls._failed:
            return False
        if cls._nlp is not None:
            return True
        try:
            import spacy
        except Exception:
            cls._failed = True
            return False
        for name in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"):
            try:
                cls._nlp = spacy.load(name, disable=["ner", "lemmatizer"])
                return True
            except Exception:
                continue
        print(
            "[constraint_metrics_en] spaCy is installed but no English model was "
            "found; run `python -m spacy download en_core_web_sm`. Falling back to "
            "the approximate regex tense backend."
        )
        cls._failed = True
        return False

    @classmethod
    def counts(cls, text: str) -> Dict[str, int]:
        present = past = 0
        for tok in cls._nlp(text):
            if tok.pos_ not in ("VERB", "AUX"):
                continue
            tag = tok.tag_
            if tag in ("VBP", "VBZ"):
                present += 1
            elif tag == "VBD":
                past += 1
            elif tag == "VBN":
                # A past participle is only past tense when its auxiliary is;
                # "has walked" is present perfect, "had walked" is past perfect.
                head = tok.head
                if head is not tok and head.tag_ == "VBD":
                    past += 1
                elif head is not tok and head.tag_ in ("VBP", "VBZ"):
                    present += 1
        return {"present": present, "past": past}


def _regex_tense_counts(words: Sequence[str]) -> Dict[str, int]:
    present = past = 0
    for raw in words:
        w = raw.lower()
        if w in _IRREGULAR_PAST:
            past += 1
        elif w in _PRESENT_AUX:
            present += 1
        elif w.endswith("ed") and len(w) > 4:
            past += 1
    return {"present": present, "past": past}


@dataclass
class StoryMetrics:
    story_index: int
    prompt_index: int
    word_count: int
    n_sentences: int
    mean_sentence_words: float
    grade_level: float
    present_verbs: int
    past_verbs: int
    present_ratio: Optional[float]
    n_quoted_spans: int
    type_token_ratio: float
    length_ok: bool
    present_tense_ok: Optional[bool]
    simple_register_ok: bool
    dialogue_ok: bool
    violations: int


class EnglishConstraintChecker:
    """Deterministic scorer for the four steered English constraints."""

    def __init__(
        self,
        *,
        max_words: int = DEFAULT_MAX_WORDS,
        present_ratio_threshold: float = DEFAULT_PRESENT_RATIO,
        max_grade_level: float = DEFAULT_MAX_GRADE,
        backend: str = "auto",
        constraints: Sequence[str] = CONSTRAINT_NAMES,
    ):
        if backend not in ("auto", "spacy", "regex"):
            raise ValueError("backend must be 'auto', 'spacy' or 'regex'.")
        unknown = [c for c in constraints if c not in CONSTRAINT_NAMES]
        if unknown:
            raise ValueError(f"unknown constraints {unknown}; have {CONSTRAINT_NAMES}")

        self.max_words = int(max_words)
        self.present_ratio_threshold = float(present_ratio_threshold)
        self.max_grade_level = float(max_grade_level)
        self.constraints = tuple(constraints)

        if backend == "spacy":
            if not _SpacyTense.available():
                raise RuntimeError(
                    "backend='spacy' requested but unavailable. "
                    "pip install spacy && python -m spacy download en_core_web_sm"
                )
            self.backend = "spacy"
        elif backend == "regex":
            self.backend = "regex"
        else:
            self.backend = "spacy" if _SpacyTense.available() else "regex"
            if self.backend == "regex" and "present_tense" in self.constraints:
                print(
                    "[constraint_metrics_en] using the approximate regex tense backend. "
                    "Install spaCy for publishable numbers: "
                    "pip install spacy && python -m spacy download en_core_web_sm"
                )

    def _tense_counts(self, text: str, words: Sequence[str]) -> Dict[str, int]:
        if self.backend == "spacy":
            return _SpacyTense.counts(text)
        return _regex_tense_counts(words)

    def evaluate(self, story: str, story_index: int = 0, prompt_index: int = 0) -> StoryMetrics:
        text = story.strip()
        words = tokenize_words(text)
        sentences = split_sentences(text)
        n_sent = max(len(sentences), 1)

        tense = self._tense_counts(text, words)
        finite = tense["present"] + tense["past"]
        if finite > 0:
            ratio: Optional[float] = tense["present"] / finite
            tense_ok: Optional[bool] = ratio >= self.present_ratio_threshold
        else:
            ratio, tense_ok = None, None

        grade = flesch_kincaid_grade(words, n_sent)
        n_quotes = len(_QUOTED.findall(text))

        length_ok = len(words) <= self.max_words
        register_ok = grade <= self.max_grade_level
        dialogue_ok = n_quotes >= 1

        checks = {
            "length": not length_ok,
            "present_tense": tense_ok is False,
            "simple_register": not register_ok,
            "dialogue": not dialogue_ok,
        }
        violations = sum(int(checks[c]) for c in self.constraints)

        return StoryMetrics(
            story_index=story_index,
            prompt_index=prompt_index,
            word_count=len(words),
            n_sentences=len(sentences),
            mean_sentence_words=round(len(words) / n_sent, 3),
            grade_level=round(grade, 3),
            present_verbs=tense["present"],
            past_verbs=tense["past"],
            present_ratio=None if ratio is None else round(ratio, 4),
            n_quoted_spans=n_quotes,
            type_token_ratio=round(len(set(w.lower() for w in words)) / max(len(words), 1), 4),
            length_ok=length_ok,
            present_tense_ok=tense_ok,
            simple_register_ok=register_ok,
            dialogue_ok=dialogue_ok,
            violations=violations,
        )

    def evaluate_all(
        self,
        stories: Sequence[str],
        prompt_indices: Optional[Sequence[int]] = None,
    ) -> Dict[str, object]:
        idx = prompt_indices or [0] * len(stories)
        rows = [self.evaluate(s, i, int(p)) for i, (s, p) in enumerate(zip(stories, idx))]
        n = max(len(rows), 1)
        scored_tense = [r for r in rows if r.present_tense_ok is not None]

        pass_rate = {
            "length": sum(r.length_ok for r in rows) / n,
            "present_tense": (
                sum(1 for r in scored_tense if r.present_tense_ok) / len(scored_tense)
                if scored_tense else float("nan")
            ),
            "simple_register": sum(r.simple_register_ok for r in rows) / n,
            "dialogue": sum(r.dialogue_ok for r in rows) / n,
        }

        return {
            "n_stories": len(rows),
            "backend": self.backend,
            "constraints": list(self.constraints),
            "thresholds": {
                "max_words": self.max_words,
                "present_ratio_threshold": self.present_ratio_threshold,
                "max_grade_level": self.max_grade_level,
            },
            "pass_rate": pass_rate,
            "tense_coverage": len(scored_tense) / n,
            "mean_violations": sum(r.violations for r in rows) / n,
            "mean_word_count": statistics.mean([r.word_count for r in rows]) if rows else 0.0,
            "median_word_count": statistics.median([r.word_count for r in rows]) if rows else 0.0,
            "mean_present_ratio": (
                statistics.mean([r.present_ratio for r in scored_tense]) if scored_tense else float("nan")
            ),
            "mean_grade_level": statistics.mean([r.grade_level for r in rows]) if rows else 0.0,
            "stories": rows,
        }

    def to_csv(self, stories: Sequence[str], path: str | Path,
               prompt_indices: Optional[Sequence[int]] = None) -> Path:
        idx = prompt_indices or [0] * len(stories)
        rows = [self.evaluate(s, i, int(p)) for i, (s, p) in enumerate(zip(stories, idx))]
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
            w.writeheader()
            for r in rows:
                w.writerow(asdict(r))
        return p

    def print_report(self, stories: Sequence[str], label: str = "",
                     prompt_indices: Optional[Sequence[int]] = None) -> Dict[str, object]:
        res = self.evaluate_all(stories, prompt_indices)
        print(f"=== Exact English constraints{(' — ' + label) if label else ''} ===")
        t = res["thresholds"]
        print(f"stories={res['n_stories']}  tense backend={res['backend']}  "
              f"coverage={res['tense_coverage']:.0%}")
        print(f"thresholds: words<={t['max_words']}, present ratio>="
              f"{t['present_ratio_threshold']}, grade level<={t['max_grade_level']}")
        pr = res["pass_rate"]
        print(f"  length          pass {pr['length']:6.1%}  (mean {res['mean_word_count']:.0f} words)")
        print(f"  present_tense   pass {pr['present_tense']:6.1%}  (mean ratio {res['mean_present_ratio']:.3f})")
        print(f"  simple_register pass {pr['simple_register']:6.1%}  (mean grade {res['mean_grade_level']:.2f})")
        print(f"  dialogue        pass {pr['dialogue']:6.1%}")
        print(f"  mean violations / story: {res['mean_violations']:.3f} of {len(self.constraints)}")
        return res
