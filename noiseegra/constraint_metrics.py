"""Exact, LLM-free checks for the three steered EGRA constraints.

Every check here is deterministic and requires no judge model, so the ablation can
be run and re-run without an API bill and without the LLM-judge biases documented
in the paper's Limitations section.

    length           word count <= 60                       (exact)
    present_tense    fraction of finite verbs in the        (morphological)
                     imperfect >= threshold
    simple_register  mean and max words per sentence        (exact)
                     within early-grade bounds

Word tokenisation, diacritic stripping and sentence splitting are delegated to
:class:`~noiseegra.egra_constraint_checker.EGRAConstraintChecker` so the word
counts reported here are identical to the ones in the published pipeline.

Tense backends
--------------
``camel``  CAMeL Tools morphological disambiguation (``pip install camel-tools``
           and ``camel_data -i disambig-mle-calima-msa-r13``). This is the
           rigorous backend and the one to use for a journal submission.
``regex``  A morphology-only fallback with no external data. It is a *proxy*: it
           recognises the 3rd-person imperfect prefixes (ya-/ta-) and the perfect
           suffix conjugations, filters out the noun families that collide with
           them (definite article, ta-marbuta and sound-feminine-plural endings)
           and resolves the ta-...-t ambiguity in favour of the perfect, which is
           correct in Arabic. It agrees with CAMeL on the common narrative verbs
           but will miss unsuffixed 3rd-person-masculine perfects (e.g. ``لعب``),
           so it under-counts past tense and is therefore a *conservative* test of
           a present-tense intervention.

``backend="auto"`` uses CAMeL when importable and falls back to regex otherwise.
"""

from __future__ import annotations

import csv
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .egra_constraint_checker import EGRAConstraintChecker

_BASE = EGRAConstraintChecker()

DEFAULT_MAX_WORDS = 60
DEFAULT_PRESENT_RATIO = 0.8
DEFAULT_MEAN_SENTENCE_WORDS = 12.0
DEFAULT_MAX_SENTENCE_WORDS = 18

CONSTRAINT_NAMES = ("length", "present_tense", "simple_register")

_AR = r"[ء-ي]"
_ARABIC_WORD = re.compile(rf"^{_AR}+$")
# 3rd-person imperfect: ya-/ta- prefix (+ optional mood/number suffix). The 1st
# person 'a-'/'na-' prefixes are excluded: they are rare in EGRA narration and
# collide with very common nouns (أرنب, نافذة, نهر).
_IMPERFECT = re.compile(rf"^[يت]{_AR}{{2,}}(?:ون|ين|ان|وا|ن)?$")
# Perfect with an explicit suffix. Stem >= 3 characters so bare 3-letter nouns
# ending in ta (بيت, بنت, وقت, صوت) cannot match.
_PERFECT = re.compile(rf"^{_AR}{{3,}}(?:ت|نا|تم|تن|وا)$")
# Nouns/adjectives that would otherwise trip the verb patterns.
_DEFINITE = re.compile(rf"^(?:ال|وال|بال|كال|فال|لل){_AR}+$")
_NOMINAL_SUFFIX = re.compile(rf"^{_AR}+(?:ة|ات)$")
_NON_VERB_STOPLIST = frozenset({
    "يوم", "يوما", "يوميا", "تحت", "تلك", "تين", "يمين", "يسار", "تاج", "تراب",
    "تفاح", "توت", "يد", "يقين", "تمثال", "يمنى", "يسرى", "تحية", "تاريخ", "توقيت",
    "بصوت", "بوقت", "بيت", "بنت", "وقت", "صوت", "زيت", "موت", "ليت", "سبت",
    "بيوت", "نحت", "بحت", "صمت", "سمت", "نبت", "بنات", "بيتنا",
})


def tokenize_words(text: str) -> List[str]:
    """Same tokenisation the published constraint checker uses."""
    return _BASE._tokenize_words(text)


def split_sentences(text: str) -> List[str]:
    return _BASE._split_sentences(text)


def normalize(text: str) -> str:
    return _BASE._normalize(text)


def _regex_tense_counts(words: Sequence[str]) -> Dict[str, int]:
    present = past = 0
    for w in words:
        if not _ARABIC_WORD.match(w):
            continue
        if w in _NON_VERB_STOPLIST or _DEFINITE.match(w) or _NOMINAL_SUFFIX.match(w):
            continue
        # Perfect wins over imperfect: ta-...-t is a perfect, never an imperfect.
        if _PERFECT.match(w):
            past += 1
        elif _IMPERFECT.match(w):
            present += 1
    return {"present": present, "past": past}


class _CamelTense:
    """Lazy singleton around CAMeL Tools' MLE disambiguator."""

    _mle = None
    _tokenize = None
    _failed = False

    @classmethod
    def available(cls) -> bool:
        if cls._failed:
            return False
        if cls._mle is not None:
            return True
        try:
            from camel_tools.disambig.mle import MLEDisambiguator
            from camel_tools.tokenizers.word import simple_word_tokenize
        except Exception:
            cls._failed = True
            return False
        try:
            cls._mle = MLEDisambiguator.pretrained()
        except Exception as exc:  # model data not downloaded
            print(
                "[constraint_metrics] camel-tools is installed but its model data is "
                f"missing ({exc}); run `camel_data -i disambig-mle-calima-msa-r13`. "
                "Falling back to the regex backend."
            )
            cls._failed = True
            return False
        cls._tokenize = simple_word_tokenize
        return True

    @classmethod
    def counts(cls, text: str) -> Dict[str, int]:
        present = past = 0
        for d in cls._mle.disambiguate(cls._tokenize(text)):
            if not d.analyses:
                continue
            analysis = d.analyses[0].analysis
            if analysis.get("pos") != "verb":
                continue
            asp = analysis.get("asp")
            if asp == "i":
                present += 1
            elif asp == "p":
                past += 1
        return {"present": present, "past": past}


@dataclass
class StoryMetrics:
    story_index: int
    word_count: int
    n_sentences: int
    mean_sentence_words: float
    max_sentence_words: int
    present_verbs: int
    past_verbs: int
    present_ratio: Optional[float]
    type_token_ratio: float
    mean_word_length: float
    length_ok: bool
    present_tense_ok: Optional[bool]
    simple_register_ok: bool
    violations: int


class ExactConstraintChecker:
    """Deterministic scorer for the three steered constraints."""

    def __init__(
        self,
        *,
        max_words: int = DEFAULT_MAX_WORDS,
        present_ratio_threshold: float = DEFAULT_PRESENT_RATIO,
        mean_sentence_words: float = DEFAULT_MEAN_SENTENCE_WORDS,
        max_sentence_words: int = DEFAULT_MAX_SENTENCE_WORDS,
        min_verbs_for_tense: int = 1,
        backend: str = "auto",
    ):
        if backend not in ("auto", "camel", "regex"):
            raise ValueError("backend must be 'auto', 'camel' or 'regex'.")
        self.max_words = int(max_words)
        self.present_ratio_threshold = float(present_ratio_threshold)
        self.mean_sentence_words = float(mean_sentence_words)
        self.max_sentence_words = int(max_sentence_words)
        self.min_verbs_for_tense = int(min_verbs_for_tense)

        if backend == "camel":
            if not _CamelTense.available():
                raise RuntimeError(
                    "backend='camel' requested but CAMeL Tools is unavailable. "
                    "pip install camel-tools && camel_data -i disambig-mle-calima-msa-r13"
                )
            self.backend = "camel"
        elif backend == "regex":
            self.backend = "regex"
        else:
            self.backend = "camel" if _CamelTense.available() else "regex"

    # ---------------------------------------------------------------- #

    def _tense_counts(self, text: str, words: Sequence[str]) -> Dict[str, int]:
        if self.backend == "camel":
            return _CamelTense.counts(text)
        return _regex_tense_counts(words)

    def evaluate(self, story: str, story_index: int = 0) -> StoryMetrics:
        text = normalize(story)
        words = tokenize_words(text)
        sentences = [tokenize_words(s) for s in split_sentences(text)]
        sentences = [s for s in sentences if s]

        sent_lens = [len(s) for s in sentences] or [len(words)]
        mean_sent = float(statistics.mean(sent_lens))
        max_sent = int(max(sent_lens))

        tense = self._tense_counts(text, words)
        finite = tense["present"] + tense["past"]
        if finite >= self.min_verbs_for_tense and finite > 0:
            ratio: Optional[float] = tense["present"] / finite
            tense_ok: Optional[bool] = ratio >= self.present_ratio_threshold
        else:
            ratio, tense_ok = None, None

        length_ok = len(words) <= self.max_words
        register_ok = (mean_sent <= self.mean_sentence_words) and (
            max_sent <= self.max_sentence_words
        )

        violations = int(not length_ok) + int(not register_ok) + int(tense_ok is False)

        return StoryMetrics(
            story_index=story_index,
            word_count=len(words),
            n_sentences=len(sentences),
            mean_sentence_words=round(mean_sent, 3),
            max_sentence_words=max_sent,
            present_verbs=tense["present"],
            past_verbs=tense["past"],
            present_ratio=None if ratio is None else round(ratio, 4),
            type_token_ratio=round(len(set(words)) / max(len(words), 1), 4),
            mean_word_length=round(
                statistics.mean([len(w) for w in words]) if words else 0.0, 3
            ),
            length_ok=length_ok,
            present_tense_ok=tense_ok,
            simple_register_ok=register_ok,
            violations=violations,
        )

    def evaluate_all(self, stories: Sequence[str]) -> Dict[str, object]:
        rows = [self.evaluate(s, i) for i, s in enumerate(stories)]
        n = max(len(rows), 1)
        scored_tense = [r for r in rows if r.present_tense_ok is not None]

        def rate(pred) -> float:
            return sum(1 for r in rows if pred(r)) / n

        return {
            "n_stories": len(rows),
            "backend": self.backend,
            "thresholds": {
                "max_words": self.max_words,
                "present_ratio_threshold": self.present_ratio_threshold,
                "mean_sentence_words": self.mean_sentence_words,
                "max_sentence_words": self.max_sentence_words,
            },
            "pass_rate": {
                "length": rate(lambda r: r.length_ok),
                "present_tense": (
                    sum(1 for r in scored_tense if r.present_tense_ok) / len(scored_tense)
                    if scored_tense
                    else float("nan")
                ),
                "simple_register": rate(lambda r: r.simple_register_ok),
            },
            "tense_coverage": len(scored_tense) / n,
            "mean_violations": sum(r.violations for r in rows) / n,
            "mean_word_count": statistics.mean([r.word_count for r in rows]) if rows else 0.0,
            "median_word_count": statistics.median([r.word_count for r in rows]) if rows else 0.0,
            "mean_present_ratio": (
                statistics.mean([r.present_ratio for r in scored_tense]) if scored_tense else float("nan")
            ),
            "mean_sentence_words": (
                statistics.mean([r.mean_sentence_words for r in rows]) if rows else 0.0
            ),
            "stories": rows,
        }

    # ---------------------------------------------------------------- #

    def to_csv(self, stories: Sequence[str], path: str | Path) -> Path:
        rows = [self.evaluate(s, i) for i, s in enumerate(stories)]
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
            writer.writeheader()
            for r in rows:
                writer.writerow(asdict(r))
        return p

    def print_report(self, stories: Sequence[str], label: str = "") -> Dict[str, object]:
        res = self.evaluate_all(stories)
        head = f"=== Exact EGRA Constraints{(' — ' + label) if label else ''} ==="
        print(head)
        print(f"stories={res['n_stories']}  tense backend={res['backend']}  "
              f"tense coverage={res['tense_coverage']:.0%}")
        t = res["thresholds"]
        print(f"thresholds: words<={t['max_words']}, present_ratio>="
              f"{t['present_ratio_threshold']}, mean_sent<={t['mean_sentence_words']}, "
              f"max_sent<={t['max_sentence_words']}")
        pr = res["pass_rate"]
        print(f"  length          pass {pr['length']:6.1%}   "
              f"(mean {res['mean_word_count']:.1f} words, median {res['median_word_count']:.0f})")
        print(f"  present_tense   pass {pr['present_tense']:6.1%}   "
              f"(mean present ratio {res['mean_present_ratio']:.3f})")
        print(f"  simple_register pass {pr['simple_register']:6.1%}   "
              f"(mean {res['mean_sentence_words']:.1f} words/sentence)")
        print(f"  mean violations / story: {res['mean_violations']:.3f} of 3")
        return res
