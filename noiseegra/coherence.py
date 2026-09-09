"""Cheap coherence checks, so diversity is not measured over broken text.

Every diversity score we have goes *up* when the text falls apart: a story that
degenerates into repeated fragments or trails off into markup embeds far from its
neighbours, and cosine spread reads that as variety. Perturbation at high
magnitude produces exactly this, so a diversity gain measured over unfiltered
output is not evidence of anything.

Two stages, both cheap:

**Heuristics** (free, no model). Repetition loops, junk characters, non-lexical
tokens, collapsed function-word ratio, run-on text with no sentence boundaries,
and trailing meta-commentary or markup. These catch the obvious failures.

**Perplexity** (optional, a small LM). Mean token negative log-likelihood under a
~0.5B model. Two uses: an overall score, thresholded *relative* to the run's own
distribution rather than an absolute cut-off, and a head-versus-tail comparison
that catches a story that starts fine and turns to garbage. The tail is scored
conditioned on the head, in the same forward pass, so it costs nothing extra.

Two things to keep in mind when using this as a filter.

*Filtering is not free.* Dropping stories changes the set diversity is computed
over, and if one condition loses more stories than another the remaining sets are
no longer comparable. Always report the drop rate per condition alongside the
scores; ``scripts/score_diversity.py`` does.

*Prefer trimming to dropping.* Most "garbage at the end" is trailing markup or
the model talking about the story it just wrote. ``trim_tail`` removes that and
keeps the story, which avoids the selection bias entirely. Only text whose body
is broken should be dropped.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# Function words. Prose sits well above the threshold; degenerate token soup and
# list-like fragments fall below it.
EN_FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at", "by",
    "for", "with", "from", "as", "into", "over", "under", "than", "then", "so",
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did",
    "have", "has", "had", "will", "would", "can", "could", "should", "may",
    "might", "must", "not", "no", "he", "she", "it", "they", "we", "you", "i",
    "him", "her", "them", "us", "me", "his", "their", "our", "your", "my", "its",
    "this", "that", "these", "those", "there", "here", "when", "while", "because",
    "about", "up", "down", "out", "off", "all", "some", "any", "one", "just",
}

AR_FUNCTION_WORDS = {
    "من", "في", "على", "إلى", "عن", "مع", "هذا", "هذه", "ذلك", "التي", "الذي",
    "أن", "إن", "كان", "كانت", "ما", "لا", "لم", "لن", "قد", "ثم", "لكن", "أو",
    "كل", "بعد", "قبل", "عند", "هو", "هي", "هم", "أنا", "نحن", "كما", "حتى",
    "بين", "تحت", "فوق", "بها", "له", "لها", "بعض", "غير", "أي", "كي",
}

TERMINALS = ".!?…۔؟"
_CLOSERS = "\"')]}»”’"
PUNCT = set(".,;:!?…-—–'\"()[]{}«»“”‘’/\\&%$#@+*=_~|`" + "،؛؟٪")

_WORD = re.compile(r"\S+")
_SENT_SPLIT = re.compile(r"[.!?…۔؟]+|\n{2,}")
_ARABIC = re.compile(r"[؀-ۿ]")
_VOWELS = set("aeiouyAEIOUY")
_REPEAT_CHAR = re.compile(r"(.)\1{3,}")
_PUNCT_RUN = re.compile(r"[!?.\-*_=~#·•]{6,}")

# Trailing material that is not part of the story: markup, chat scaffolding, and
# the model commenting on what it just wrote.
_TAIL_META = re.compile(
    r"^\s*(?:"
    r"```|~~~|#{1,6}\s|-{3,}\s*$|\*{3,}\s*$|"
    r"<\|[^|]*\|>|\[/?INST\]|<</?SYS>>|</?s>|"
    r"(?:assistant|user|system)\s*[:：]?\s*$|"
    r"\(?(?:note|word count|words?\s*[:：]|explanation|analysis|disclaimer|"
    r"constraints? (?:met|check)|i hope|hope (?:this|that)|let me know|"
    r"here(?:'s| is) (?:the|a|your)|as requested|feel free)"
    r")",
    re.I,
)


@dataclass
class CoherenceReport:
    """Verdict for one story. ``ok`` is False if any check fired."""

    ok: bool
    reasons: List[str]
    scores: Dict[str, float]
    text: str = ""
    trimmed_words: int = 0

    @property
    def reason(self) -> str:
        return ",".join(self.reasons) if self.reasons else "ok"


@dataclass
class CoherenceThresholds:
    min_words: int = 15
    max_repeat_ratio: float = 0.45      # duplicated 5-grams, as a fraction
    max_char_repeat: int = 4            # 'aaaa' and longer
    max_junk_char_ratio: float = 0.06
    max_nonlexical_ratio: float = 0.12
    # Prose sits far above this; token soup falls below. English only: Arabic
    # attaches its prepositions, conjunctions and articles to the following word
    # (wa-, bi-, li-, al-), so counting whitespace tokens undercounts them badly.
    # Measured on 300 clean Arabic stories the ratio runs 0.00 to 0.30 with a
    # median of 0.164, against 0.09 to 0.21 for a deliberately degenerate
    # temperature-1.8 run -- the distributions overlap, so the check carries no
    # signal there and is disabled rather than shipped as a coin flip.
    min_function_ratio: float = 0.15
    min_function_ratio_arabic: float = 0.0
    max_words_per_sentence: float = 70.0
    max_ppl_z: float = 3.5             # robust z against a reference condition
    # How far *below* the reference a story's sentence-to-sentence similarity may
    # fall before it counts as incoherent. Relative for the same reason as the
    # perplexity threshold: the absolute value depends on the genre and language.
    min_coherence_z: float = -3.5
    max_tail_nll_gap: float = 1.5      # nats; ~4.5x perplexity on the tail
    tail_fraction: float = 0.2
    extra: Dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  Script handling                                                             #
# --------------------------------------------------------------------------- #

def detect_script(text: str) -> str:
    """``"arabic"`` if the text is mostly Arabic letters, else ``"latin"``."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "latin"
    arabic = sum(1 for c in letters if _ARABIC.match(c))
    return "arabic" if arabic > 0.5 * len(letters) else "latin"


def _function_words(script: str) -> set:
    return AR_FUNCTION_WORDS if script == "arabic" else EN_FUNCTION_WORDS


# --------------------------------------------------------------------------- #
#  Individual measurements                                                     #
# --------------------------------------------------------------------------- #

def repeat_ratio(words: Sequence[str], n: int = 5) -> float:
    """Fraction of n-grams that are duplicates. A decode loop drives this to 1."""
    if len(words) < 2 * n:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def junk_char_ratio(text: str) -> float:
    """Fraction of characters that are neither letters, digits, space nor punctuation."""
    if not text:
        return 0.0
    bad = 0
    for c in text:
        if c.isalnum() or c.isspace() or c in PUNCT:
            continue
        cat = unicodedata.category(c)
        # Combining marks are part of the writing system, not noise: Arabic
        # diacritics (fatha, damma, sukun) and Latin accents all land here.
        if cat.startswith("M"):
            continue
        # currency, maths and dingbats are fine in a story; control codes and
        # unassigned code points are not
        if cat in ("Cc", "Cf", "Cn", "Co", "Cs"):
            bad += 1
        elif cat.startswith("S") and c not in "£€$%+×÷":
            bad += 1
        else:
            bad += 1
    return bad / len(text)


def nonlexical_ratio(words: Sequence[str], script: str = "latin") -> float:
    """Fraction of word-like tokens that do not look like words.

    A Latin-script token with four or more letters and no vowel, a run of four
    identical characters, or an absurdly long token. Arabic is written without
    short vowels, so the vowel rule is skipped there.
    """
    stripped = [w.strip("".join(PUNCT)) for w in words]
    cand = [w for w in stripped if w and any(c.isalpha() for c in w)]
    if not cand:
        return 0.0
    bad = 0
    for w in cand:
        if len(w) > 25:
            bad += 1
        elif _REPEAT_CHAR.search(w):
            bad += 1
        elif script != "arabic" and len(w) >= 4 and not (set(w) & _VOWELS):
            bad += 1
    return bad / len(cand)


def function_word_ratio(words: Sequence[str], script: str = "latin") -> float:
    fw = _function_words(script)
    stripped = [w.strip("".join(PUNCT)).lower() for w in words]
    cand = [w for w in stripped if w]
    if not cand:
        return 0.0
    return sum(1 for w in cand if w in fw) / len(cand)


def words_per_sentence(text: str) -> float:
    """Mean words between sentence boundaries. Huge when punctuation disappears."""
    parts = [p for p in _SENT_SPLIT.split(text) if p.strip()]
    n_words = len(_WORD.findall(text))
    if not n_words:
        return 0.0
    return n_words / max(len(parts), 1)


def ends_mid_sentence(text: str) -> bool:
    s = text.rstrip()
    while s and s[-1] in _CLOSERS:
        s = s[:-1].rstrip()
    return bool(s) and s[-1] not in TERMINALS


def split_sentences(text: str) -> List[str]:
    return [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]


def sentence_coherence(
    texts: Sequence[str], encode, min_sentences: int = 3
) -> List[float]:
    """Mean cosine similarity between consecutive sentences, per text.

    A story moves from one sentence to a related one. Word salad -- grammatical
    text whose sentences are about nothing in particular -- does not, and this is
    the cheapest way to see that, because it reuses the embedding model the
    diversity scoring has already loaded rather than adding a second one.

    ``encode`` takes a list of strings and returns L2-normalised row vectors.
    Texts with fewer than ``min_sentences`` sentences get ``nan``: two sentences
    give one comparison, which is too noisy to threshold on.
    """
    import numpy as np

    per_text = [split_sentences(t) for t in texts]
    flat, spans = [], []
    for sents in per_text:
        start = len(flat)
        if len(sents) >= min_sentences:
            flat.extend(sents)
        spans.append((start, len(flat)))
    if not flat:
        return [float("nan")] * len(texts)
    emb = np.asarray(encode(flat))
    out = []
    for lo, hi in spans:
        if hi - lo < min_sentences:
            out.append(float("nan"))
            continue
        block = emb[lo:hi]
        sims = (block[:-1] * block[1:]).sum(axis=1)
        out.append(float(sims.mean()))
    return out


# --------------------------------------------------------------------------- #
#  Tail trimming                                                               #
# --------------------------------------------------------------------------- #

def trim_tail(text: str, max_trim_frac: float = 0.25) -> Tuple[str, int]:
    """Strip trailing markup, chat scaffolding and post-story commentary.

    Then, if what is left ends mid-sentence, cut back to the last sentence
    boundary -- but only when that costs less than ``max_trim_frac`` of the words.
    A story that is mostly one unterminated run is a real failure, not a tail to
    tidy, and is left intact for the checks to reject.

    Returns ``(text, words_removed)``.
    """
    original = len(_WORD.findall(text))
    budget = original * max_trim_frac
    lines = text.rstrip().split("\n")
    removed = 0
    while lines:
        last = lines[-1].strip()
        if not last:
            lines.pop()
            continue
        if not (_TAIL_META.match(last) or _PUNCT_RUN.fullmatch(last)):
            break
        cost = len(_WORD.findall(last))
        # Never eat a quarter of the text as "trailing markup": a story that
        # happens to open its last line with "Here is ..." is a story, and past
        # this budget the text is broken rather than merely untidy.
        if removed + cost > budget:
            break
        removed += cost
        lines.pop()
    out = "\n".join(lines).rstrip()
    out = _PUNCT_RUN.sub("", out).rstrip()

    if out and ends_mid_sentence(out):
        cut = max(out.rfind(c) for c in TERMINALS)
        if cut > 0:
            candidate = out[: cut + 1]
            kept = len(_WORD.findall(candidate))
            if original and (original - kept) / original <= max_trim_frac:
                out = candidate
    return out, original - len(_WORD.findall(out))


# --------------------------------------------------------------------------- #
#  Perplexity                                                                  #
# --------------------------------------------------------------------------- #

class PerplexityScorer:
    """Mean token NLL under a small LM, split into head and tail.

    ``Qwen/Qwen2.5-0.5B`` by default: about 1 GB in float16, multilingual, so the
    same scorer works on the Arabic runs. The tail is scored inside the same
    forward pass, conditioned on the head, which is what makes it a test for
    "started fine, then fell apart" rather than just "the ending is unusual".
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B",
        *,
        batch_size: int = 8,
        max_length: int = 512,
        device: Optional[str] = None,
        dtype: str = "auto",
    ):
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_length = max_length
        self._device = device
        self._dtype = dtype
        self._model = None
        self._tok = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._dtype == "auto":
            cap = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
            torch_dtype = torch.bfloat16 if cap >= 8 else torch.float16
        else:
            torch_dtype = getattr(torch, self._dtype)
        if not torch.cuda.is_available():
            torch_dtype = torch.float32
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._tok.padding_side = "right"
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if self._device:
            self._model = self._model.to(self._device)
        self._model.eval()

    def score(
        self, texts: Sequence[str], tail_fraction: float = 0.2
    ) -> List[Tuple[float, float, float]]:
        """``(overall_nll, head_nll, tail_nll)`` per text, in nats per token."""
        import torch

        self._load()
        out: List[Tuple[float, float, float]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = list(texts[i : i + self.batch_size])
            enc = self._tok(chunk, return_tensors="pt", padding=True, truncation=True,
                            max_length=self.max_length)
            enc = {k: v.to(self._model.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self._model(**enc).logits.float()
            ids, mask = enc["input_ids"], enc["attention_mask"]
            logp = torch.log_softmax(logits[:, :-1], dim=-1)
            nll = -logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            valid = mask[:, 1:].bool()
            for row, keep in zip(nll, valid):
                vals = row[keep]
                n = vals.numel()
                if n == 0:
                    out.append((float("nan"),) * 3)
                    continue
                cut = max(1, int(round(n * (1.0 - tail_fraction))))
                head = vals[:cut]
                tail = vals[cut:] if cut < n else vals[-1:]
                out.append((
                    float(vals.mean()), float(head.mean()), float(tail.mean()),
                ))
        return out


def _median(vals: Sequence[float]) -> float:
    v = sorted(vals)
    m = len(v) // 2
    return v[m] if len(v) % 2 else 0.5 * (v[m - 1] + v[m])


def nll_reference(values: Sequence[float]) -> Tuple[float, float]:
    """``(median, scale)`` for a robust z-score, from median absolute deviation.

    A median and MAD rather than a mean and standard deviation, so a handful of
    broken stories in the reference does not drag the threshold out to meet them.
    ``scale`` of 0 means the reference is degenerate and no z-score is defined.
    """
    vals = [v for v in values if v == v]
    if len(vals) < 3:
        return float("nan"), 0.0
    median = _median(vals)
    mad = _median([abs(v - median) for v in vals])
    return median, 1.4826 * mad  # 1.4826 makes MAD comparable to a std deviation


def robust_z(
    values: Sequence[float], reference: Optional[Tuple[float, float]] = None
) -> List[float]:
    """Robust z-scores of ``values``.

    ``reference`` is a ``(median, scale)`` pair from ``nll_reference``, usually
    computed on a *different*, known-good set. Judging a condition against its own
    median silently fails when most of that condition is broken -- the median is
    then broken too and nothing stands out. Pass a reference built from the
    baseline condition to avoid that.
    """
    median, scale = reference if reference is not None else nll_reference(values)
    if scale <= 1e-9 or median != median:
        return [0.0 for _ in values]
    return [0.0 if v != v else (v - median) / scale for v in values]


# --------------------------------------------------------------------------- #
#  The filter                                                                  #
# --------------------------------------------------------------------------- #

class CoherenceFilter:
    """Heuristic checks, optionally backed by a small LM.

    ``check`` judges one story on its own. ``evaluate`` judges a whole set,
    which is what the perplexity thresholds need: they are relative to the
    distribution of the set, not to an absolute number that would depend on the
    genre, language and length of the stories.
    """

    def __init__(
        self,
        thresholds: Optional[CoherenceThresholds] = None,
        *,
        script: str = "auto",
        trim: bool = True,
        ppl_scorer: Optional[PerplexityScorer] = None,
    ):
        self.t = thresholds or CoherenceThresholds()
        self.script = script
        self.trim = trim
        self.ppl_scorer = ppl_scorer

    def _script_for(self, text: str) -> str:
        return detect_script(text) if self.script == "auto" else self.script

    def check(self, text: str) -> CoherenceReport:
        """Heuristics only. Perplexity needs the whole set; see ``evaluate``."""
        t = self.t
        trimmed_words = 0
        if self.trim:
            text, trimmed_words = trim_tail(text)
        words = _WORD.findall(text)
        script = self._script_for(text)

        scores = {
            "words": float(len(words)),
            "repeat_ratio": repeat_ratio(words),
            "junk_char_ratio": junk_char_ratio(text),
            "nonlexical_ratio": nonlexical_ratio(words, script),
            "function_ratio": function_word_ratio(words, script),
            "words_per_sentence": words_per_sentence(text),
            "trimmed_words": float(trimmed_words),
        }

        reasons = []
        if len(words) < t.min_words:
            reasons.append("too_short")
        if scores["repeat_ratio"] > t.max_repeat_ratio:
            reasons.append("repetition")
        if scores["junk_char_ratio"] > t.max_junk_char_ratio:
            reasons.append("junk_chars")
        if scores["nonlexical_ratio"] > t.max_nonlexical_ratio:
            reasons.append("nonlexical")
        min_fw = (t.min_function_ratio_arabic if script == "arabic"
                  else t.min_function_ratio)
        if len(words) >= t.min_words and min_fw > 0 and scores["function_ratio"] < min_fw:
            reasons.append("no_function_words")
        if scores["words_per_sentence"] > t.max_words_per_sentence:
            reasons.append("run_on")
        if ends_mid_sentence(text) and len(words) >= t.min_words:
            reasons.append("dangling_end")

        return CoherenceReport(not reasons, reasons, scores, text, trimmed_words)

    def evaluate(
        self,
        texts: Sequence[str],
        reference: Optional[Tuple[float, float]] = None,
    ) -> List[CoherenceReport]:
        """Judge a set of stories, adding the perplexity checks if a scorer is set.

        The perplexity threshold is a robust z-score rather than an absolute
        number, so it adapts to the language and genre. ``reference`` is a
        ``(median, scale)`` pair from ``nll_reference``; without it the set is
        judged against its own median, which is only safe when most of the set is
        fine. For a sweep, build the reference once from the baseline condition and
        pass it to every condition -- see ``perplexity_pass``.
        """
        reports = [self.check(t) for t in texts]
        if self.ppl_scorer is None or not reports:
            return reports
        triples = self.ppl_scorer.score([r.text or " " for r in reports],
                                        self.t.tail_fraction)
        self.apply_perplexity(reports, triples, reference)
        return reports

    def apply_perplexity(
        self,
        reports: Sequence[CoherenceReport],
        triples: Sequence[Tuple[float, float, float]],
        reference: Optional[Tuple[float, float]] = None,
    ) -> List[CoherenceReport]:
        """Attach perplexity scores and verdicts to already-checked reports.

        Split out from ``evaluate`` so a caller with several conditions can score
        them all in one pass and judge them against a shared reference.
        """
        zs = robust_z([o for o, _, _ in triples], reference)
        for rep, (overall, head, tail), z in zip(reports, triples, zs):
            rep.scores["nll"] = overall
            rep.scores["nll_z"] = z
            rep.scores["nll_head"] = head
            rep.scores["nll_tail"] = tail
            rep.scores["nll_tail_gap"] = (
                tail - head if not (math.isnan(tail) or math.isnan(head)) else float("nan")
            )
            if z > self.t.max_ppl_z:
                rep.reasons.append("high_perplexity")
            if rep.scores["nll_tail_gap"] > self.t.max_tail_nll_gap:
                rep.reasons.append("garbage_tail")
            rep.ok = not rep.reasons
        return list(reports)


def apply_sentence_coherence(
    reports: Sequence[CoherenceReport],
    sims: Sequence[float],
    thresholds: "CoherenceThresholds",
    reference: Optional[Tuple[float, float]] = None,
) -> List[CoherenceReport]:
    """Attach sentence-coherence scores and flag ``incoherent`` stories."""
    zs = robust_z(sims, reference)
    for rep, sim, z in zip(reports, sims, zs):
        rep.scores["sentence_coherence"] = sim
        rep.scores["sentence_coherence_z"] = z
        if sim == sim and z < thresholds.min_coherence_z:
            rep.reasons.append("incoherent")
        rep.ok = not rep.reasons
    return list(reports)


def summarise(reports: Sequence[CoherenceReport]) -> Dict[str, float]:
    """Pass rate and a count for each reason that fired."""
    out: Dict[str, float] = {
        "n": float(len(reports)),
        "kept": float(sum(1 for r in reports if r.ok)),
    }
    out["pass_rate"] = out["kept"] / out["n"] if reports else float("nan")
    for r in reports:
        for reason in r.reasons:
            out[reason] = out.get(reason, 0.0) + 1.0
    return out
