"""Steering whose strength is set by the constraint error of the text so far.

Steering more than doubles compliance on the *monotone* requirements -- present
tense, simple vocabulary, varied sentence openings, where more of the property is
always better -- and cuts compliance on the *counting* ones from 23% to 6%. Word
count, sentence count and exactly-two-quoted-lines are bands, not directions: a
constant push moves the whole distribution of the property and has no way to say
"stop, that is enough". Pushing "more dialogue" takes the exactly-two rule from
4% to 0%, which is the mechanism working perfectly and the requirement failing
anyway.

Feedback on activations does not fix this. A correction that closes the gap
between where the story sits on a constraint axis and where the positive contrast
examples sit saturates at "this text is quote-like enough", because that is what
the examples encode. It never saturates at "I have written two quotes". The
direction is a local property of text; the requirement is a global count.

What can count is the generation loop. At decode step t the tokens produced so
far are available, and the same deterministic checks that score the finished
story can be run over them: how many words, how many sentences, how many quoted
spans, how long the sentence in progress is. That gives a signed error per
requirement -- below the band, inside it, or above it -- and the steering
coefficient becomes a function of that error rather than a constant.

Inside the band the coefficient is zero and the model writes unsteered. Outside
it, the push is proportional to how far outside, and in the direction that brings
the text back. A requirement already satisfied costs nothing, which is what the
constant version cannot do.

The closed-loop steering work this resembles regulates activations: the error
signal is a feature magnitude or a token probability, and the loop controls how
much of the intervention survives rather than what the output actually does. The
error here is measured on the decoded text by the scorer itself, so the quantity
being controlled and the quantity being reported are the same one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_QUOTED = re.compile(r"[\"“][^\"“”]*?\b\w+\b[^\"“”]*?[\"”]")
_SENT_END = re.compile(r"[.!?][\"”’')\]]*(?:\s|$)")


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return lo if x < lo else (hi if x > hi else x)


@dataclass
class PartialState:
    """What the checks can see part-way through a story."""
    words: int
    sentences: int
    quotes: int
    words_in_current_sentence: int


def read_partial(text: str) -> PartialState:
    words = _WORD.findall(text)
    ends = list(_SENT_END.finditer(text))
    finished = len(ends)
    tail = text[ends[-1].end():] if ends else text
    return PartialState(
        words=len(words),
        sentences=finished,
        quotes=len(_QUOTED.findall(text)),
        words_in_current_sentence=len(_WORD.findall(tail)),
    )


@dataclass
class ConstraintController:
    """Signed error per steered direction, from the text written so far.

    ``+1`` means "more of this direction", ``-1`` means "less", ``0`` means the
    requirement is currently satisfied and the direction should be silent.

    ``word_range``, ``sentence_range``, ``sentence_word_range`` and ``n_quotes``
    mirror the checker's own thresholds, so the controller and the scorer cannot
    disagree about what the target is.
    """

    word_range: Sequence[int] = (50, 65)
    sentence_range: Sequence[int] = (6, 8)
    sentence_word_range: Sequence[int] = (4, 10)
    n_quotes: int = 2
    # How much of the word budget must be spent before the length error is
    # trusted. Every story is under the minimum at token five, and pushing
    # against closure there says nothing about whether it will end short.
    warmup: float = 0.6

    # The requirements this controller can measure on partial text. A direction
    # named here is steered by its error; one that is not has no error signal and
    # is steered by a constant coefficient instead.
    WATCHED = ("closure", "terse", "dialogue")

    def watched(self) -> Sequence[str]:
        return self.WATCHED

    def errors(self, text: str) -> Dict[str, float]:
        st = read_partial(text)
        lo_w, hi_w = self.word_range
        lo_s, hi_s = self.sentence_range
        lo_sw, hi_sw = self.sentence_word_range
        out: Dict[str, float] = {}

        # closure: bring the story to an end. Silent until the story is most of
        # the way through its budget, then rising with how far over it runs.
        if st.words > hi_w:
            out["closure"] = _clamp((st.words - hi_w) / max(hi_w - lo_w, 1))
        elif self.warmup * lo_w <= st.words < lo_w:
            # Short, but far enough in that stopping here would land under the
            # minimum. Earlier than that every story is short and the signal means
            # nothing.
            out["closure"] = -_clamp((lo_w - st.words) / max(hi_w - lo_w, 1))
        else:
            out["closure"] = 0.0

        # terse: end sentences sooner. Driven by the sentence in progress, which
        # is the quantity the rule is actually about, and by how the sentence
        # count is tracking against the word count.
        cur = st.words_in_current_sentence
        if cur > hi_sw:
            out["terse"] = _clamp((cur - hi_sw) / max(hi_sw - lo_sw, 1))
        elif 0 < cur < lo_sw:
            out["terse"] = -_clamp((lo_sw - cur) / max(hi_sw - lo_sw, 1))
        else:
            out["terse"] = 0.0
        # A story on course for too many sentences wants longer ones, whatever
        # the sentence in progress is doing.
        if st.words >= self.warmup * lo_w and st.sentences > hi_s:
            out["terse"] = min(out["terse"], -_clamp((st.sentences - hi_s) / max(hi_s - lo_s, 1)))

        # dialogue: exactly n quoted lines. Push for them until they exist, then
        # push against, which is the half a constant coefficient cannot express.
        if st.quotes < self.n_quotes:
            out["dialogue"] = _clamp((self.n_quotes - st.quotes) / max(self.n_quotes, 1))
        elif st.quotes > self.n_quotes:
            out["dialogue"] = -_clamp((st.quotes - self.n_quotes) / max(self.n_quotes, 1))
        else:
            out["dialogue"] = 0.0
        return out


class ConstraintProbe:
    """Logits processor that decodes the story so far and stores its errors.

    A logits processor is the one place in ``generate`` that sees the tokens
    produced up to now. It runs after the forward pass, so the errors it writes
    are read by the block hooks on the *next* step -- one step of lag, against a
    story of eighty, which is not worth avoiding.

    Only the newly generated tokens are decoded, not the prompt.
    """

    def __init__(self, tokenizer, state: dict, controller: ConstraintController,
                 prompt_len: int, every: int = 4):
        self.tokenizer = tokenizer
        self.state = state
        self.controller = controller
        self.prompt_len = prompt_len
        self.every = max(int(every), 1)
        self.calls = 0
        # Seed every watched requirement at zero error before a token exists, so
        # the steering plan can tell a direction the controller watches (and must
        # not push constantly) from one it has no probe for, from the very first
        # step rather than from the probe's first firing.
        state["errors"] = {name: 0.0 for name in controller.watched()}
        state.setdefault("text", "")

    def __call__(self, input_ids, scores):
        self.calls += 1
        if self.calls % self.every == 0 or self.calls == 1:
            new = input_ids[0][self.prompt_len:]
            if new.numel():
                text = self.tokenizer.decode(new, skip_special_tokens=True)
                self.state["text"] = text
                self.state["errors"] = self.controller.errors(text)
        return scores


# Imported here rather than at the top: the scorer imports this module for its
# own checks, and importing it back at module level closes the loop.
from .constraint_metrics_en import genders_present, has_simile  # noqa: E402

# --------------------------------------------------------------------------- #
#  A controller for rules about the whole story                               #
# --------------------------------------------------------------------------- #

# Past-tense verbs, which are what the present-tense rule is broken by. Common
# irregulars by name, and regular "-ed" forms that are not also adjectives.
_PAST_IRREGULAR = frozenset("""
was were had did said went saw took came got made knew felt ran gave told
found thought left kept held stood heard let began brought sat
""".split())
_PAST_ED = re.compile(r"\b\w{3,}ed\b", re.I)
_NOT_PAST_ED = frozenset("""
red bed fed led wed shed sled bred fled bled need indeed speed breed
tired scared worried excited surprised crowded pointed
""".split())
_CAPITAL = re.compile(r"(?<![.!?\"'“‘]\s)(?<!^)\b([A-Z][a-z]{2,})\b")
_COMMON_CAP = frozenset("""
The A An And But So Then When While Her His She He They It Mr Mrs Ms Dr
Monday Tuesday Wednesday Thursday Friday Saturday Sunday
January February March April May June July August September October November
December I'm I'll I've
""".split())


def _has_past_tense(text: str) -> bool:
    low = text.lower()
    for w in _WORD.findall(low):
        if w in _PAST_IRREGULAR:
            return True
    for m in _PAST_ED.findall(text):
        if m.lower() not in _NOT_PAST_ED:
            return True
    return False


def _names_so_far(text: str) -> int:
    """Distinct capitalised words that look like names. A heuristic: the scorer
    is what decides, this only has to be right often enough to steer on."""
    return len({m for m in _CAPITAL.findall(text) if m not in _COMMON_CAP})


@dataclass
class WholeStoryController(ConstraintController):
    """Error per direction for rules about the whole story, not counts in it.

    Every rule here is satisfied or not, so the error is one-sided: a direction
    pushes while its requirement is unmet and goes silent the moment it is met.
    Nothing has to be known in advance about how often the model breaks it --
    the story being written is the measurement, which is the point. The constant
    coefficients this replaces had to be set from a calibration run, and a
    calibration run measures one model on one prompt.

    Two rules need a little of the story before they mean anything. "No simile
    yet" is true of every story at its third word, so the comparison, the
    speech and the second character are only asked for once the story is far
    enough in that their absence says something. `closure` is the opposite: it
    stays silent until the story runs past the length it was asked for, and
    then rises with the overrun, which is the failure this method has left.
    """

    target_words: int = 150
    hard_words: int = 200
    # How far in before "it has not happened yet" is evidence rather than a
    # statement about the story being short.
    patience: float = 0.45

    WATCHED = ("closure", "present_tense", "simile", "dialogue",
               "both_genders", "named_character", "mature_register")

    def watched(self) -> Sequence[str]:
        return self.WATCHED

    def errors(self, text: str) -> Dict[str, float]:
        st = read_partial(text)
        n = st.words
        out: Dict[str, float] = {k: 0.0 for k in self.WATCHED}
        late = n >= self.patience * self.target_words

        # Bring it to an end, once it has run past what was asked for.
        if n > self.target_words:
            out["closure"] = _clamp(
                (n - self.target_words) / max(self.hard_words - self.target_words, 1))

        # A past-tense verb has been written: the rule is already broken, and
        # pushing harder is the only thing that keeps the rest of it present.
        if _has_past_tense(text):
            out["present_tense"] = 1.0

        # Asked for once the story is far enough in to have had the chance.
        if late:
            if not has_simile(text):
                out["simile"] = 1.0
            if st.quotes < 1:
                out["dialogue"] = 1.0
            if genders_present(text) < 2:
                out["both_genders"] = 1.0

        # Exactly one name: push for one while there are none, push against
        # while there are too many.
        names = _names_so_far(text)
        if names == 0 and n >= 0.25 * self.target_words:
            out["named_character"] = 1.0
        elif names >= 2:
            out["named_character"] = -_clamp((names - 1) / 2.0)

        # The reading floor is about sentence length. Measured on what has been
        # finished, once there is enough of it to average.
        if st.sentences >= 3:
            mean_len = (n - st.words_in_current_sentence) / st.sentences
            if mean_len < 9.0:
                out["mature_register"] = _clamp((9.0 - mean_len) / 4.0)

        return out
