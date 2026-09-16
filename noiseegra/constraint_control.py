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
        state.setdefault("errors", {})
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
