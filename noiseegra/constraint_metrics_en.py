"""Exact, judge-free checks for the English constrained-writing task.

Every check here is deterministic: two people running it on the same story get
the same answer, and no language model is asked for an opinion. That is the point
-- an LLM judge would make the constraint scores a function of the judge.

    length             word count inside a two-sided band
    present_tense      every finite verb is present tense                (POS-tagged)
    simple_register    Flesch-Kincaid grade level <= N                   (exact formula)
    dialogue           exactly N quoted utterances
    easy_opening       the first sentence is at most N words
    sentence_band      every sentence is between N and M words
    sentence_count     between N and M sentences
    short_words        no word longer than N syllables
    one_name           exactly one proper name, used at least N times    (NER)
    varied_openers     no word begins more than N sentences
    plain_punctuation  one paragraph, and only simple punctuation
    spelled_number     a counted quantity appears, written as a word

Every rule is two-sided or near-ceiling on purpose. The first version of this
file asked for one-sided minima -- "at most 60 words", "at least one line of
dialogue", "no digits" -- and Qwen3-8B passed ten of the twelve at 98-100%. A
constraint that is always satisfied measures nothing: it cannot record the cost
of a perturbation, because there is no headroom below it to lose. A constraint
that is never satisfied measures nothing either, for the same reason from the
other side; the old ``varied_openers`` (no repeated sentence opener at all) sat
at 0% and contributed a constant to every condition.

So each rule here is a band the model has to land inside, not a ceiling it has to
stay under: a word count between 50 and 65 rather than under 60, six to eight
sentences rather than five to nine, exactly two lines of dialogue rather than at
least one, every sentence between four and ten words. Landing inside a band
requires the model to plan the whole story, and that is what a perturbation
disturbs.

None of them is specific to a story. They are properties of the text -- counts,
bands, tense, vocabulary, punctuation -- so the same list applies whatever the
model decides to write about, and the diversity measurement is not fighting a
constraint that dictates the content.

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

DEFAULT_WORD_RANGE = (50, 65)
DEFAULT_PRESENT_RATIO = 1.0
DEFAULT_MAX_GRADE = 2.5
DEFAULT_N_QUOTES = 2
DEFAULT_MAX_OPENING_WORDS = 5
DEFAULT_SENTENCE_WORD_RANGE = (4, 10)
DEFAULT_SENTENCE_RANGE = (6, 8)
DEFAULT_MAX_SYLLABLES = 2
DEFAULT_MIN_NAME_USES = 3
DEFAULT_MAX_OPENER_USES = 2
# Share of a story's five-word sequences that are repeats. A perturbation strong
# enough to buy diversity is often strong enough to send the model into a loop,
# and a set of forty differently-broken stories scores as *more* diverse than
# forty good ones, because differently-broken text really is mutually
# dissimilar. Without this in the scored list, an arm can win the headline number
# by producing worse writing: steering at the measured optimum tripled the rate
# of looping stories, from two in twenty-four to six, and no column said so.
DEFAULT_MAX_REPEAT = 0.15

# Thresholds for the monotone requirements. Each was chosen by measuring the
# candidate on forty baseline stories written *without* being asked for it, and
# keeping the level that passed between 5% and 50% of them. A rule the model
# already satisfies unasked measures nothing, and one it never satisfies measures
# nothing either; the levels below leave the prompted rate somewhere in the
# middle, where a perturbation has room to cost something and an intervention has
# room to win something.
DEFAULT_MAX_SENTENCE_WORDS = 8     # unprompted 65%
DEFAULT_MIN_QUOTES = 3             # unprompted 48%
DEFAULT_MAX_ADVERBS = 2            # unprompted 50%
DEFAULT_MIN_SENSORY = 2            # unprompted 18%
DEFAULT_MAX_SUBORDINATE = 1        # unprompted 78%
DEFAULT_MAX_WORD_USES = 3          # unprompted 32%, i.e. no content word four times

# Words a five-year-old would call seeing, hearing, smelling, tasting or
# touching. "Use the senses" is one of the oldest pieces of advice given to
# people writing for children, and unlike most such advice it is countable.
_SENSORY_WORDS = frozenset("""
see sees seeing saw look looks looking watch watches watching peek peeks
hear hears hearing heard listen listens listening sound sounds
smell smells smelling sniff sniffs taste tastes tasting
touch touches touching feel feels feeling felt pat pats
soft warm cold hot bright dark loud quiet sweet shiny wet dry rough smooth
sunny cool fresh crunchy sticky fuzzy crisp salty sour fluffy sparkly
buzz buzzes hum hums crunch crunches splash splashes whisper whispers
shout shouts giggle giggles rustle rustles thump thumps
""".split())

# Adverbs that are not really modifiers of manner and that no writing guide asks
# anyone to remove: the negation, the existential "there", and the "so" that
# starts a sentence.
_FREE_ADVERBS = frozenset(["not", "n't", "there", "so"])

# Kept so callers that still pass the old single-sided thresholds keep working.
DEFAULT_MAX_WORDS = DEFAULT_WORD_RANGE[1]

CONSTRAINT_NAMES = (
    "length", "present_tense", "simple_register", "dialogue",
    "easy_opening", "sentence_band", "sentence_count", "short_words",
    "one_name", "varied_openers", "plain_punctuation", "spelled_number",
    "no_repetition",
    # Monotone requirements: more of the property is never worse. They are
    # additions rather than replacements so that every earlier run re-scores to
    # the same numbers it did when it was produced.
    "short_sentences", "dialogue_min", "plain_words", "sensory",
    "simple_syntax", "fresh_words", "named_character",
)

# What a checker scores unless told otherwise: the original thirteen. The
# monotone requirements are additions, and making them default would silently
# change what every existing caller measures.
DEFAULT_CONSTRAINTS = CONSTRAINT_NAMES[:13]

# The monotone set, which is what the main comparison is run against. A band --
# "between 50 and 65 words", "exactly two quoted lines" -- has no token-level
# direction meaning "stop here", and steering along one reliably makes it worse:
# across thirty steered conditions not one improved a banded requirement over
# leaving the model alone. These thirteen are all one-sided, so a push along the
# direction and the requirement agree about which way is better.
MONOTONE_CONSTRAINTS = (
    "present_tense", "simple_register", "short_words", "easy_opening",
    "short_sentences", "dialogue_min", "varied_openers", "plain_words",
    "sensory", "simple_syntax", "fresh_words", "named_character",
    "no_repetition",
)

# Short column headers for wide tables, and the full text for the legend.
CONSTRAINT_SHORT = {
    "length": "len", "present_tense": "tense", "simple_register": "easy",
    "dialogue": "quote", "easy_opening": "open", "sentence_band": "sband",
    "sentence_count": "count", "short_words": "syll", "one_name": "name",
    "varied_openers": "varied", "plain_punctuation": "punct",
    "spelled_number": "number",
    "short_sentences": "short", "dialogue_min": "speech", "plain_words": "adverb",
    "sensory": "sense", "simple_syntax": "syntax", "fresh_words": "fresh",
    "named_character": "named", "no_repetition": "norep",
}

_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
# A sentence ends at .!? only when what follows starts a new sentence. Without
# that lookahead, `"Look!" she says.` splits into two, which inflates the
# sentence count, shortens the mean sentence, and so depresses the
# Flesch-Kincaid grade -- every story with a line of dialogue was mis-scored.
_SENT_SPLIT = re.compile(
    r"(?<=[.!?])[\"”’')\]]*\s+(?=[\"“‘'(\[]*[A-Z0-9])"
    r"|\n+"
)
# A quoted span of at least a couple of words, straight or curly quotes.
_QUOTED = re.compile(r"[\"“][^\"“”]*?\b\w+\b[^\"“”]*?[\"”]")
_VOWEL_GROUP = re.compile(r"[aeiouy]+")
_DIGIT = re.compile(r"\d")
_LIST_OR_HEADING = re.compile(r"^\s*(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|\*\*|Title\s*:)", re.I | re.M)
_BLANK_LINE = re.compile(r"\n\s*\n")
# Punctuation a grade-2 reader has not met yet. Straight and curly quotes,
# apostrophes, commas, full stops, question and exclamation marks are allowed;
# everything below is not. The em dash and the semicolon are the two the model
# reaches for constantly, which is what makes this rule bite.
_HARD_PUNCT = re.compile(r"[;:–—()\[\]{}…*#/\\~<>|_+=]|--")

_IRREGULAR_PAST = frozenset("""
was were had did said went took saw came got made knew thought found told became
left felt put brought began kept held wrote stood heard let meant set met ran paid
sat spoke lay led grew lost fell sent built understood drew broke spent rose drove
bought wore chose ate gave took slept swam sang rang drank shook threw flew forgot
hid rode wrote won taught caught bit blew froze stole tore woke bore dug hung struck
""".split())
_PRESENT_AUX = frozenset("is are am has have does do can will shall may must".split())

# A counted quantity, written out. "one" is deliberately absent: it is the most
# common word in the list by an order of magnitude and is usually an article or a
# pronoun ("one day", "the one who"), so counting it would hand the model the
# requirement for free and the check would measure nothing.
_NUMBER_WORDS = frozenset("""
two three four five six seven eight nine ten eleven twelve thirteen fourteen
fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty
seventy eighty ninety hundred thousand
""".split())

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


def repeated_ngram_share(words: Sequence[str], n: int = 5) -> float:
    """Share of a story's ``n``-word sequences that are not unique.

    Zero for ordinary prose and close to one for a story that loops. Five words
    is long enough that a repeated phrase is deliberate rather than incidental --
    "she runs to the" recurring twice is not a degenerate story, and "I am going
    to the store" eight times is.
    """
    w = [x.lower() for x in words]
    if len(w) < 2 * n:
        return 0.0
    grams = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


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

    _cache_key = None
    _cache_doc = None

    @classmethod
    def doc(cls, text: str):
        """One parse per story, shared by every check that needs one.

        Four checks now want a parsed document. Without this each of them parsed
        the story again, and parsing is the slowest thing the scorer does.
        """
        if text != cls._cache_key:
            cls._cache_key, cls._cache_doc = text, cls._nlp(text)
        return cls._cache_doc

    @classmethod
    def adverb_count(cls, text: str) -> int:
        return sum(1 for t in cls.doc(text)
                   if t.pos_ == "ADV" and t.text.lower() not in _FREE_ADVERBS)

    @classmethod
    def subordinate_count(cls, text: str) -> int:
        """Clauses hanging off another clause: relative, adverbial, complement.

        "The dog runs" is one clause. "The dog that barks runs" and "She says
        that the dog runs" are two, and the second clause is the thing a
        grade-two reader has to hold in mind while finishing the first.

        Quoted speech is removed before parsing. ``"Come here," Mira says`` makes
        the quoted words a complement clause of "says", so every line of dialogue
        scored as a subordinate clause -- which put this requirement in direct
        contradiction with the one asking for three lines of speech, and no story
        could satisfy both. Reported speech is not the syntactic load this rule is
        about; a child reads the quoted sentence as its own sentence.
        """
        return sum(1 for t in cls.doc(_QUOTED.sub(" it ", text))
                   if t.dep_ in ("advcl", "relcl", "ccomp", "acl"))

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


_SUBORDINATORS = frozenset("""
because when while although though since unless until after before whenever
wherever that which who whom whose if as
""".split())


def _regex_adverb_count(words: Sequence[str]) -> int:
    """Words ending in -ly, which is most manner adverbs and a few adjectives.

    "Lonely" and "friendly" are counted wrongly here. Install spaCy.
    """
    return sum(1 for w in words if len(w) > 4 and w.lower().endswith("ly"))


def _regex_subordinate_count(words: Sequence[str]) -> int:
    return sum(1 for w in words if w.lower() in _SUBORDINATORS)


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
    min_sentence_words: int
    opening_words: int
    grade_level: float
    present_verbs: int
    past_verbs: int
    present_ratio: Optional[float]
    n_quoted_spans: int
    n_long_words: int
    n_digits: int
    n_hard_punct: int
    n_number_words: int
    repeat_share: float
    n_adverbs: int
    n_sensory: int
    n_subordinate: int
    max_word_uses: int
    n_names: int
    name_uses: int
    max_opener_uses: int
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
        word_range: Tuple[int, int] = DEFAULT_WORD_RANGE,
        max_words: Optional[int] = None,
        min_words: Optional[int] = None,
        present_ratio_threshold: float = DEFAULT_PRESENT_RATIO,
        max_grade_level: float = DEFAULT_MAX_GRADE,
        n_quotes: int = DEFAULT_N_QUOTES,
        max_opening_words: int = DEFAULT_MAX_OPENING_WORDS,
        sentence_word_range: Tuple[int, int] = DEFAULT_SENTENCE_WORD_RANGE,
        sentence_range: Tuple[int, int] = DEFAULT_SENTENCE_RANGE,
        max_syllables: int = DEFAULT_MAX_SYLLABLES,
        min_name_uses: int = DEFAULT_MIN_NAME_USES,
        max_opener_uses: int = DEFAULT_MAX_OPENER_USES,
        max_repeat: float = DEFAULT_MAX_REPEAT,
        max_sentence_words: int = DEFAULT_MAX_SENTENCE_WORDS,
        min_quotes: int = DEFAULT_MIN_QUOTES,
        max_adverbs: int = DEFAULT_MAX_ADVERBS,
        min_sensory: int = DEFAULT_MIN_SENSORY,
        max_subordinate: int = DEFAULT_MAX_SUBORDINATE,
        max_word_uses: int = DEFAULT_MAX_WORD_USES,
        backend: str = "auto",
        constraints: Sequence[str] = DEFAULT_CONSTRAINTS,
    ):
        if backend not in ("auto", "spacy", "regex"):
            raise ValueError("backend must be 'auto', 'spacy' or 'regex'.")
        unknown = [c for c in constraints if c not in CONSTRAINT_NAMES]
        if unknown:
            raise ValueError(f"unknown constraints {unknown}; have {CONSTRAINT_NAMES}")

        lo_w, hi_w = int(word_range[0]), int(word_range[1])
        if min_words is not None:
            lo_w = int(min_words)
        if max_words is not None:
            hi_w = int(max_words)
        if lo_w > hi_w:
            raise ValueError(f"word_range is empty: {lo_w} > {hi_w}")
        self.word_range = (lo_w, hi_w)
        self.min_words, self.max_words = lo_w, hi_w

        self.present_ratio_threshold = float(present_ratio_threshold)
        self.max_grade_level = float(max_grade_level)
        self.n_quotes = int(n_quotes)
        self.max_opening_words = int(max_opening_words)
        self.sentence_word_range = (int(sentence_word_range[0]), int(sentence_word_range[1]))
        self.sentence_range = (int(sentence_range[0]), int(sentence_range[1]))
        self.max_syllables = int(max_syllables)
        self.min_name_uses = int(min_name_uses)
        self.max_opener_uses = int(max_opener_uses)
        self.max_repeat = float(max_repeat)
        self.max_sentence_words = int(max_sentence_words)
        self.min_quotes = int(min_quotes)
        self.max_adverbs = int(max_adverbs)
        self.min_sensory = int(min_sensory)
        self.max_subordinate = int(max_subordinate)
        self.max_word_uses = int(max_word_uses)
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
        wlo, whi = self.word_range
        slo, shi = self.sentence_word_range
        tense = ("every verb in it is in the present tense"
                 if self.present_ratio_threshold >= 1.0 else
                 "it is written in the present tense throughout (at least "
                 f"{self.present_ratio_threshold:.0%} of its verbs)")
        return {
            "length": f"the whole story is between {wlo} and {whi} words long",
            "present_tense": tense,
            "simple_register": "the language is simple enough for a grade-"
                               f"{self.max_grade_level:g} reader: short common words in "
                               "short sentences (Flesch-Kincaid grade at most "
                               f"{self.max_grade_level:g})",
            "dialogue": f"exactly {self.n_quotes} lines of speech appear inside "
                        "quotation marks, no more and no fewer",
            "easy_opening": f"the first sentence is very easy: at most "
                            f"{self.max_opening_words} words",
            "sentence_band": f"every sentence is between {slo} and {shi} words long",
            "sentence_count": f"the story has between {lo} and {hi} sentences",
            "short_words": f"no word has more than {self.max_syllables} syllables",
            "one_name": "exactly one character is given a name, and that name is used "
                        f"at least {self.min_name_uses} times",
            "varied_openers": "sentence openings are varied: no word begins more than "
                              f"{self.max_opener_uses} sentences",
            "plain_punctuation": "it is one paragraph with no title, heading or list, and "
                                 "uses only full stops, commas, question marks, "
                                 "exclamation marks, apostrophes and quotation marks",
            "spelled_number": "the story counts something: a number of two or more "
                              "appears, written as a word and never as a digit",
            "no_repetition": "it does not repeat itself: no run of five words appears "
                             "twice",
            "short_sentences": f"every sentence is short: at most {self.max_sentence_words} words",
            "dialogue_min": f"at least {self.min_quotes} lines of speech appear inside "
                            "quotation marks",
            "plain_words": "it tells the story with verbs rather than adverbs: at most "
                           f"{self.max_adverbs} adverb{'s' if self.max_adverbs != 1 else ''} "
                           "in the whole story",
            "sensory": f"at least {self.min_sensory} words say how something looks, "
                       "sounds, feels, smells or tastes",
            "simple_syntax": "the sentences are simple: at most "
                             f"{self.max_subordinate} of them joins a second clause on "
                             "with a word like because, when, that or which",
            "fresh_words": "it does not lean on one word: no word of four letters or "
                           f"more is used more than {self.max_word_uses} times",
            "named_character": "a character is given a name, and that name is used at "
                               f"least {self.min_name_uses} times",
        }

    def requirements_short(self) -> Dict[str, str]:
        """The same rules in a few words each, for a table's row labels."""
        lo, hi = self.sentence_range
        wlo, whi = self.word_range
        slo, shi = self.sentence_word_range
        return {
            "length": f"{wlo} to {whi} words",
            "present_tense": ("every verb present tense"
                              if self.present_ratio_threshold >= 1.0
                              else "present tense throughout"),
            "simple_register": f"grade {self.max_grade_level:g} or easier",
            "dialogue": f"exactly {self.n_quotes} quoted lines",
            "easy_opening": f"first sentence at most {self.max_opening_words} words",
            "sentence_band": f"every sentence {slo} to {shi} words",
            "sentence_count": f"{lo} to {hi} sentences",
            "short_words": f"no word over {self.max_syllables} syllables",
            "one_name": f"one name, used {self.min_name_uses}+ times",
            "varied_openers": f"no opener used over {self.max_opener_uses} times",
            "plain_punctuation": "one paragraph, simple punctuation",
            "spelled_number": "a number word, no digits",
            "no_repetition": "no repeated five-word run",
            "short_sentences": f"every sentence at most {self.max_sentence_words} words",
            "dialogue_min": f"{self.min_quotes}+ quoted lines",
            "plain_words": f"at most {self.max_adverbs} adverbs",
            "sensory": f"{self.min_sensory}+ sensory words",
            "simple_syntax": f"at most {self.max_subordinate} subordinate clauses",
            "fresh_words": f"no word used over {self.max_word_uses} times",
            "named_character": f"a name, used {self.min_name_uses}+ times",
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
        hard_punct = len(_HARD_PUNCT.findall(text))
        number_words = [w for w in words if w.lower() in _NUMBER_WORDS]

        repeat_share = repeated_ngram_share(words)

        if self.backend == "spacy":
            n_adverbs = _Spacy.adverb_count(text)
            n_subordinate = _Spacy.subordinate_count(text)
        else:
            n_adverbs = _regex_adverb_count(words)
            n_subordinate = _regex_subordinate_count(words)
        n_sensory = sum(1 for w in words if w.lower() in _SENSORY_WORDS)

        names = self._name_counts(text)
        # Four letters or more, so "the", "a", "and" and "was" are not what fails
        # the rule, and not the character's name, which another requirement asks
        # to be used at least three times -- between them those two would leave
        # only the single value three, which is a band and not what this is.
        content = [w.lower() for w in words
                   if len(w) >= 4 and w.lower() not in names]
        max_word_uses = max((content.count(w) for w in set(content)), default=0)
        n_names = len(names)
        name_uses = max(names.values()) if names else 0

        openers = [tokenize_words(s)[0].lower() for s in sentences if tokenize_words(s)]
        opener_uses = max((openers.count(o) for o in set(openers)), default=0)

        lo, hi = self.sentence_range
        wlo, whi = self.word_range
        slo, shi = self.sentence_word_range
        checks: Dict[str, Optional[bool]] = {
            "length": wlo <= len(words) <= whi,
            "present_tense": tense_ok,
            "simple_register": grade <= self.max_grade_level,
            "dialogue": n_quotes == self.n_quotes,
            "easy_opening": bool(sentences) and sent_words[0] <= self.max_opening_words,
            "sentence_band": bool(sentences) and slo <= min(sent_words)
                             and max(sent_words) <= shi,
            "sentence_count": lo <= len(sentences) <= hi,
            "short_words": not long_words,
            "one_name": n_names == 1 and name_uses >= self.min_name_uses,
            "varied_openers": opener_uses <= self.max_opener_uses,
            "plain_punctuation": not _BLANK_LINE.search(text)
                                 and not _LIST_OR_HEADING.search(text)
                                 and hard_punct == 0,
            "spelled_number": bool(number_words) and digits == 0,
            "no_repetition": repeat_share <= self.max_repeat,
            "short_sentences": bool(sentences) and max(sent_words) <= self.max_sentence_words,
            "dialogue_min": n_quotes >= self.min_quotes,
            "plain_words": n_adverbs <= self.max_adverbs,
            "sensory": n_sensory >= self.min_sensory,
            "simple_syntax": n_subordinate <= self.max_subordinate,
            "fresh_words": max_word_uses <= self.max_word_uses,
            "named_character": n_names >= 1 and name_uses >= self.min_name_uses,
        }
        violations = sum(1 for c in self.constraints if checks[c] is False)

        return StoryMetrics(
            story_index=story_index,
            prompt_index=prompt_index,
            word_count=len(words),
            n_sentences=len(sentences),
            mean_sentence_words=round(len(words) / n_sent, 3),
            max_sentence_words=max(sent_words),
            min_sentence_words=min(sent_words),
            opening_words=sent_words[0],
            grade_level=round(grade, 3),
            present_verbs=tense["present"],
            past_verbs=tense["past"],
            present_ratio=None if ratio is None else round(ratio, 4),
            n_quoted_spans=n_quotes,
            n_long_words=len(long_words),
            n_digits=digits,
            n_hard_punct=hard_punct,
            n_number_words=len(number_words),
            repeat_share=round(repeat_share, 4),
            n_adverbs=n_adverbs,
            n_sensory=n_sensory,
            n_subordinate=n_subordinate,
            max_word_uses=max_word_uses,
            n_names=n_names,
            name_uses=name_uses,
            max_opener_uses=opener_uses,
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
