"""Exact, judge-free checks for the English constrained-writing task.

Every check here is deterministic: two people running it on the same story get
the same answer, and no language model is asked for an opinion. That is the point
-- an LLM judge would make the constraint scores a function of the judge.

    length             word count <= N
    present_tense      finite verbs in the present >= a fraction        (POS-tagged)
    simple_register    Flesch-Kincaid grade level <= N                  (exact formula)
    dialogue           contains at least one quoted utterance
    easy_opening       the first sentence is at most N words
    short_sentences    no sentence longer than N words
    sentence_count     between N and M sentences
    short_words        no word longer than N syllables
    no_digits          numbers are written as words, not digits
    one_name           exactly one proper name, used at least twice     (NER)
    varied_openers     no two sentences begin with the same word
    single_paragraph   one paragraph, no title, heading or list markers

The first four mirror the Arabic constraints from the original study, so the
cross-lingual claim is about the same constraint types. The rest reproduce
requirements the original EGRA prompt made in prose but never scored: a very easy
first sentence, one proper name, varied sentence structure rather than a list of
weakly linked sentences, vocabulary within reach of the age group. Having twelve
rather than four matters because the pressure a long constraint list puts on the
model is what drives it toward one modal story, which is the thing the method is
supposed to relieve.

Two checks need a model and degrade gracefully without one:

``present_tense``  spaCy part-of-speech tags, else a suffix-and-irregular-list
                   regex fallback that is genuinely approximate.
``one_name``       spaCy named-entity recognition, else a capitalisation
                   heuristic that skips sentence-initial words.

Install spaCy for anything you intend to publish:
``pip install spacy && python -m spacy download en_core_web_sm``.
"""

from __future__ import annotations

import csv
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
#  Defaults                                                                    #
# --------------------------------------------------------------------------- #

DEFAULT_MAX_WORDS = 60
DEFAULT_PRESENT_RATIO = 0.8
DEFAULT_MAX_GRADE = 3.0
DEFAULT_MAX_OPENING_WORDS = 8
DEFAULT_MAX_SENTENCE_WORDS = 15
DEFAULT_SENTENCE_RANGE = (5, 9)
DEFAULT_MAX_SYLLABLES = 3
DEFAULT_MIN_NAME_USES = 2

CONSTRAINT_NAMES = (
    "length", "present_tense", "simple_register", "dialogue",
    "easy_opening", "short_sentences", "sentence_count", "short_words",
    "no_digits", "one_name", "varied_openers", "single_paragraph",
)

# Short column headers for wide tables, and the full text for the legend.
CONSTRAINT_SHORT = {
    "length": "len", "present_tense": "tense", "simple_register": "easy",
    "dialogue": "quote", "easy_opening": "open", "short_sentences": "short",
    "sentence_count": "count", "short_words": "syll", "no_digits": "digit",
    "one_name": "name", "varied_openers": "varied", "single_paragraph": "format",
}

_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SENT_SPLIT = re.compile(r"(?<=[.!?])[\s\"”’)]+|\n+")
# A quoted span of at least a couple of words, straight or curly quotes.
_QUOTED = re.compile(r"[\"“][^\"“”]*?\b\w+\b[^\"“”]*?[\"”]")
_VOWEL_GROUP = re.compile(r"[aeiouy]+")
_DIGIT = re.compile(r"\d")
_LIST_OR_HEADING = re.compile(r"^\s*(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|\*\*|Title\s*:)", re.I | re.M)
_BLANK_LINE = re.compile(r"\n\s*\n")

_IRREGULAR_PAST = frozenset("""
was were had did said went took saw came got made knew thought found told became
left felt put brought began kept held wrote stood heard let meant set met ran paid
sat spoke lay led grew lost fell sent built understood drew broke spent rose drove
bought wore chose ate gave took slept swam sang rang drank shook threw flew forgot
hid rode wrote won taught caught bit blew froze stole tore woke bore dug hung struck
""".split())
_PRESENT_AUX = frozenset("is are am has have does do can will shall may must".split())

# Common words that are not proper names, for the no-spaCy fallback. A story
# often uses its character's name only at the start of sentences, so the fallback
# cannot simply skip sentence-initial capitals -- it has to decide whether a
# capitalised opener is a name or an ordinary word, and this is the list that
# decides. It covers function words plus the words stories actually open with.
_NOT_A_NAME = frozenset("""
i a an the and or but if of to in on at by for with from as into onto over under
than then so is are am was were be been being do does did done have has had will
would shall should can could may might must not no nor yes there here where when
while because about above across after again against along around before behind
below beneath beside between beyond during except inside near off out outside
since through throughout till toward towards until up upon within without
all also always any anyone anything both each either enough even ever every
everyone everything few many more most much neither never nobody none nothing
once only other others own same several some somebody someone something sometimes
soon still such today tonight tomorrow yesterday too very what whatever whenever
whether which while who whom whose why how just like now well yet
he she it they we you him her them us his their our your my its me
this that these those one two three four five six seven eight nine ten
finally first second next last later meanwhile suddenly slowly quickly quietly
together perhaps maybe instead however although though because unless
mr mrs ms dr sir madam
come look wait stop go help hello hi thanks please oh okay ok sorry careful listen
watch run hurry let lets dont don't cant can't im i'm its it's thats that's hey ouch
wow yay shh good great nice sure fine right left home away back down
monday tuesday wednesday thursday friday saturday sunday
january february march april may june july august september october november december
""".split())


# --------------------------------------------------------------------------- #
#  Text helpers                                                                #
# --------------------------------------------------------------------------- #

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
    n = len(_VOWEL_GROUP.findall(w))
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and n > 1:
        n -= 1
    return max(n, 1)


def flesch_kincaid_grade(words: Sequence[str], n_sentences: int) -> float:
    """0.39*(words/sentence) + 11.8*(syllables/word) - 15.59."""
    if not words or n_sentences <= 0:
        return 0.0
    syllables = sum(count_syllables(w) for w in words)
    return 0.39 * (len(words) / n_sentences) + 11.8 * (syllables / len(words)) - 15.59


# --------------------------------------------------------------------------- #
#  spaCy backends                                                              #
# --------------------------------------------------------------------------- #

class _Spacy:
    """One loaded pipeline, shared by the tense and name checks."""

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
                cls._nlp = spacy.load(name, disable=["lemmatizer"])
                return True
            except Exception:
                continue
        print("[constraint_metrics_en] spaCy is installed but no English model was "
              "found; run `python -m spacy download en_core_web_sm`. Falling back to "
              "the approximate backends.")
        cls._failed = True
        return False

    @classmethod
    def doc(cls, text: str):
        return cls._nlp(text)

    @classmethod
    def tense_counts(cls, text: str) -> Dict[str, int]:
        present = past = 0
        for tok in cls.doc(text):
            if tok.pos_ not in ("VERB", "AUX"):
                continue
            tag = tok.tag_
            if tag in ("VBP", "VBZ"):
                present += 1
            elif tag == "VBD":
                past += 1
            elif tag == "VBN":
                # A past participle is only past tense when its auxiliary is:
                # "has walked" is present perfect, "had walked" is past perfect.
                head = tok.head
                if head is not tok and head.tag_ == "VBD":
                    past += 1
                elif head is not tok and head.tag_ in ("VBP", "VBZ"):
                    present += 1
        return {"present": present, "past": past}

    @classmethod
    def name_counts(cls, text: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for ent in cls.doc(text).ents:
            if ent.label_ == "PERSON":
                key = ent.text.strip().split()[0].lower()
                counts[key] = counts.get(key, 0) + 1
        return counts


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


def _heuristic_name_counts(text: str) -> Dict[str, int]:
    """Best-effort proper names without a NER model.

    A capitalised token counts as a name when it is not in ``_NOT_A_NAME`` and the
    same token never appears in lower case elsewhere in the story. That second
    condition is what separates "Mira" from "Soon": a story that opens a sentence
    with an ordinary word almost always uses that word in lower case somewhere
    too. Sentence-initial occurrences are counted, because a character's name is
    frequently used only there.

    Approximate by construction. Install spaCy for numbers you intend to publish.
    """
    words = tokenize_words(text)
    lowercased = {w for w in words if w[0].islower()}
    counts: Dict[str, int] = {}
    for w in words:
        if not w[0].isupper():
            continue
        key = w.lower()
        if key in _NOT_A_NAME or key in lowercased:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
#  Per-story result                                                            #
# --------------------------------------------------------------------------- #

@dataclass
class StoryMetrics:
    story_index: int
    prompt_index: int
    word_count: int
    n_sentences: int
    mean_sentence_words: float
    max_sentence_words: int
    opening_words: int
    grade_level: float
    present_verbs: int
    past_verbs: int
    present_ratio: Optional[float]
    n_quoted_spans: int
    n_long_words: int
    n_digits: int
    n_names: int
    name_uses: int
    n_repeated_openers: int
    type_token_ratio: float
    violations: int
    checks: Dict[str, Optional[bool]] = field(default_factory=dict)

    def flat(self) -> Dict[str, object]:
        row = {k: v for k, v in asdict(self).items() if k != "checks"}
        for name, ok in self.checks.items():
            row[f"{name}_ok"] = "" if ok is None else int(ok)
        return row


class EnglishConstraintChecker:
    """Deterministic scorer for the English constrained-writing constraints.

    ``constraints`` selects which of ``CONSTRAINT_NAMES`` count toward the
    violation total. Everything is measured either way, so a check can be added
    to the report later without regenerating anything.
    """

    def __init__(
        self,
        *,
        max_words: int = DEFAULT_MAX_WORDS,
        present_ratio_threshold: float = DEFAULT_PRESENT_RATIO,
        max_grade_level: float = DEFAULT_MAX_GRADE,
        max_opening_words: int = DEFAULT_MAX_OPENING_WORDS,
        max_sentence_words: int = DEFAULT_MAX_SENTENCE_WORDS,
        sentence_range: Tuple[int, int] = DEFAULT_SENTENCE_RANGE,
        max_syllables: int = DEFAULT_MAX_SYLLABLES,
        min_name_uses: int = DEFAULT_MIN_NAME_USES,
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
        self.max_opening_words = int(max_opening_words)
        self.max_sentence_words = int(max_sentence_words)
        self.sentence_range = (int(sentence_range[0]), int(sentence_range[1]))
        self.max_syllables = int(max_syllables)
        self.min_name_uses = int(min_name_uses)
        self.constraints = tuple(constraints)

        if backend == "spacy":
            if not _Spacy.available():
                raise RuntimeError(
                    "backend='spacy' requested but unavailable. "
                    "pip install spacy && python -m spacy download en_core_web_sm"
                )
            self.backend = "spacy"
        elif backend == "regex":
            self.backend = "regex"
        else:
            self.backend = "spacy" if _Spacy.available() else "regex"
            if self.backend == "regex" and (
                "present_tense" in self.constraints or "one_name" in self.constraints
            ):
                print("[constraint_metrics_en] using the approximate regex backends for "
                      "tense and names. Install spaCy for publishable numbers: "
                      "pip install spacy && python -m spacy download en_core_web_sm")

    # -- description of the task, for prompts and legends -------------------- #

    def requirements(self) -> Dict[str, str]:
        lo, hi = self.sentence_range
        return {
            "length": f"the whole story is at most {self.max_words} words long",
            "present_tense": "it is written in the present tense throughout (at least "
                             f"{self.present_ratio_threshold:.0%} of its verbs)",
            "simple_register": "the language is simple enough for a grade-"
                               f"{self.max_grade_level:g} reader: short common words in "
                               "short sentences (Flesch-Kincaid grade at most "
                               f"{self.max_grade_level:g})",
            "dialogue": "at least one line of speech appears inside quotation marks",
            "easy_opening": f"the first sentence is very easy: at most "
                            f"{self.max_opening_words} words",
            "short_sentences": f"no sentence runs longer than {self.max_sentence_words} "
                               "words",
            "sentence_count": f"the story has between {lo} and {hi} sentences",
            "short_words": f"no word has more than {self.max_syllables} syllables",
            "no_digits": "any number is written out as a word, never as a digit",
            "one_name": "exactly one character is given a name, and that name is used "
                        f"at least {self.min_name_uses} times",
            "varied_openers": "no two sentences begin with the same word",
            "single_paragraph": "it is one paragraph, with no title, heading, bullet or "
                                "numbered list",
        }

    def requirements_short(self) -> Dict[str, str]:
        """The same rules in a few words each, for a table's row labels."""
        lo, hi = self.sentence_range
        return {
            "length": f"at most {self.max_words} words",
            "present_tense": "present tense throughout",
            "simple_register": f"grade {self.max_grade_level:g} or easier",
            "dialogue": "a line of dialogue",
            "easy_opening": f"first sentence at most {self.max_opening_words} words",
            "short_sentences": f"no sentence over {self.max_sentence_words} words",
            "sentence_count": f"{lo} to {hi} sentences",
            "short_words": f"no word over {self.max_syllables} syllables",
            "no_digits": "no digits",
            "one_name": f"one name, used {self.min_name_uses}+ times",
            "varied_openers": "no repeated sentence opener",
            "single_paragraph": "one paragraph, no heading",
        }

    # -- measurement --------------------------------------------------------- #

    def _tense_counts(self, text: str, words: Sequence[str]) -> Dict[str, int]:
        if self.backend == "spacy":
            return _Spacy.tense_counts(text)
        return _regex_tense_counts(words)

    def _name_counts(self, text: str) -> Dict[str, int]:
        if self.backend == "spacy":
            return _Spacy.name_counts(text)
        return _heuristic_name_counts(text)

    def evaluate(self, story: str, story_index: int = 0, prompt_index: int = 0) -> StoryMetrics:
        text = story.strip()
        words = tokenize_words(text)
        sentences = split_sentences(text)
        n_sent = max(len(sentences), 1)
        sent_words = [len(tokenize_words(s)) for s in sentences] or [0]

        tense = self._tense_counts(text, words)
        finite = tense["present"] + tense["past"]
        ratio: Optional[float] = tense["present"] / finite if finite else None
        tense_ok: Optional[bool] = (
            None if ratio is None else ratio >= self.present_ratio_threshold
        )

        grade = flesch_kincaid_grade(words, n_sent)
        n_quotes = len(_QUOTED.findall(text))
        long_words = [w for w in words if count_syllables(w) > self.max_syllables]
        digits = len(_DIGIT.findall(text))

        names = self._name_counts(text)
        n_names = len(names)
        name_uses = max(names.values()) if names else 0

        openers = [tokenize_words(s)[0].lower() for s in sentences if tokenize_words(s)]
        repeated = len(openers) - len(set(openers))

        lo, hi = self.sentence_range
        checks: Dict[str, Optional[bool]] = {
            "length": len(words) <= self.max_words,
            "present_tense": tense_ok,
            "simple_register": grade <= self.max_grade_level,
            "dialogue": n_quotes >= 1,
            "easy_opening": bool(sentences) and sent_words[0] <= self.max_opening_words,
            "short_sentences": max(sent_words) <= self.max_sentence_words,
            "sentence_count": lo <= len(sentences) <= hi,
            "short_words": not long_words,
            "no_digits": digits == 0,
            "one_name": n_names == 1 and name_uses >= self.min_name_uses,
            "varied_openers": repeated == 0,
            "single_paragraph": not _BLANK_LINE.search(text)
                                and not _LIST_OR_HEADING.search(text),
        }
        violations = sum(1 for c in self.constraints if checks[c] is False)

        return StoryMetrics(
            story_index=story_index,
            prompt_index=prompt_index,
            word_count=len(words),
            n_sentences=len(sentences),
            mean_sentence_words=round(len(words) / n_sent, 3),
            max_sentence_words=max(sent_words),
            opening_words=sent_words[0],
            grade_level=round(grade, 3),
            present_verbs=tense["present"],
            past_verbs=tense["past"],
            present_ratio=None if ratio is None else round(ratio, 4),
            n_quoted_spans=n_quotes,
            n_long_words=len(long_words),
            n_digits=digits,
            n_names=n_names,
            name_uses=name_uses,
            n_repeated_openers=repeated,
            type_token_ratio=round(len(set(w.lower() for w in words)) / max(len(words), 1), 4),
            violations=violations,
            checks=checks,
        )

    def evaluate_all(
        self,
        stories: Sequence[str],
        prompt_indices: Optional[Sequence[int]] = None,
    ) -> Dict[str, object]:
        idx = prompt_indices or [0] * len(stories)
        rows = [self.evaluate(s, i, int(p)) for i, (s, p) in enumerate(zip(stories, idx))]
        n = max(len(rows), 1)

        pass_rate: Dict[str, float] = {}
        for name in CONSTRAINT_NAMES:
            scored = [r for r in rows if r.checks.get(name) is not None]
            pass_rate[name] = (sum(1 for r in scored if r.checks[name]) / len(scored)
                               if scored else float("nan"))
        tense_scored = [r for r in rows if r.present_ratio is not None]

        def mean(key):
            return statistics.mean([getattr(r, key) for r in rows]) if rows else 0.0

        return {
            "n_stories": len(rows),
            "backend": self.backend,
            "constraints": list(self.constraints),
            "requirements": self.requirements(),
            "pass_rate": pass_rate,
            "tense_coverage": len(tense_scored) / n,
            "mean_violations": sum(r.violations for r in rows) / n,
            "max_violations": len(self.constraints),
            "mean_word_count": mean("word_count"),
            "median_word_count": statistics.median([r.word_count for r in rows]) if rows else 0.0,
            "mean_sentences": mean("n_sentences"),
            "mean_grade_level": mean("grade_level"),
            "mean_present_ratio": (statistics.mean([r.present_ratio for r in tense_scored])
                                   if tense_scored else float("nan")),
            "stories": rows,
        }

    def to_csv(self, stories: Sequence[str], path: str | Path,
               prompt_indices: Optional[Sequence[int]] = None) -> Path:
        idx = prompt_indices or [0] * len(stories)
        rows = [self.evaluate(s, i, int(p)) for i, (s, p) in enumerate(zip(stories, idx))]
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        flat = [r.flat() for r in rows]
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(flat[0].keys()) if flat else [])
            w.writeheader()
            w.writerows(flat)
        return p

    def print_report(self, stories: Sequence[str], label: str = "",
                     prompt_indices: Optional[Sequence[int]] = None) -> Dict[str, object]:
        res = self.evaluate_all(stories, prompt_indices)
        req = res["requirements"]
        print(f"=== Exact English constraints{(' — ' + label) if label else ''} ===")
        print(f"stories={res['n_stories']}  backend={res['backend']}  "
              f"tense coverage={res['tense_coverage']:.0%}")
        width = max(len(req[c]) for c in self.constraints)
        for c in self.constraints:
            rate = res["pass_rate"][c]
            shown = "  --  " if rate != rate else f"{rate:6.1%}"
            print(f"  {req[c]:<{width}}  pass {shown}")
        print(f"  mean violations / story: {res['mean_violations']:.3f} of "
              f"{len(self.constraints)}")
        return res
