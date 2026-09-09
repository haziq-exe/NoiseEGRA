"""Coherence checks. No model downloads: the perplexity stage uses a stub."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from noiseegra.coherence import (  # noqa: E402
    CoherenceFilter, CoherenceThresholds, apply_sentence_coherence, detect_script,
    ends_mid_sentence, function_word_ratio, junk_char_ratio, nll_reference,
    nonlexical_ratio, repeat_ratio, robust_z, sentence_coherence, split_sentences,
    summarise, trim_tail, words_per_sentence,
)

failures = []


def check(name, cond, detail=""):
    print(f"{'[ok] ' if cond else '[FAIL] '}{name}" + (f"  {detail}" if detail else ""))
    if not cond:
        failures.append(name)


GOOD = (
    "Mira wakes before the light and goes out to the coop. She counts the hens twice "
    "and the number is wrong. The gate hangs open on one hinge, and the grass beyond "
    "it is flattened in a narrow line. She follows it to the hedge and stops. "
    "\"There you are,\" she says. The hen blinks at her from under the leaves, and "
    "Mira carries it back with both hands."
)

GOOD_AR = (
    "تستيقظ ليلى في الصباح وتذهب إلى الحديقة. تجد الباب مفتوحا ولا ترى القطة الصغيرة. "
    "تبحث بين الأشجار وتنادي عليها بصوت هادئ. تسمع صوتا خافتا تحت الشجرة الكبيرة. "
    "تركض ليلى نحو الصوت وتجد القطة نائمة بين الأوراق. تحملها بين يديها وتعود إلى البيت."
)

# --------------------------------------------------------------------------- #
print("== the good cases pass ==")
filt = CoherenceFilter()
check("a normal English story passes", filt.check(GOOD).ok, filt.check(GOOD).reason)
check("a normal Arabic story passes", filt.check(GOOD_AR).ok, filt.check(GOOD_AR).reason)
check("script detection", detect_script(GOOD) == "latin" and detect_script(GOOD_AR) == "arabic")
check("a good story is not trimmed", filt.check(GOOD).trimmed_words == 0)

# --------------------------------------------------------------------------- #
print("\n== the broken cases fail ==")
CONTROL_JUNK = "\x00\x01\x02\x03\x04\x05\x06\x07\x0b\x0e\x0f\x10\x11\x12\x13\x14"
cases = {
    "too_short": "She runs.",
    "repetition": "The door opens and closes again. " * 20,
    "junk_chars": GOOD[:120] + " " + CONTROL_JUNK * 6 + " " + GOOD[120:],
    "nonlexical": "Brgh vskd tmpl krxz ndhs pltk mrqz wxbn tlkd sprm "
                  "vtsh mlkp brgz tnnn wqrs mnbv lkjh gfds qwrt zxcv bnml. " * 3,
    "no_function_words": "Storm. Harbour. Rope. Lantern. Salt. Gull. Anchor. Fog. Bell. "
                         "Rock. Wave. Net. Boat. Pier. Sail. Wind. Tide. Shell. Sand. "
                         "Star. Moon. Cliff. Cave. Gale. Mast. Deck. Hull. Keel. Oar.",
    "run_on": " ".join(["and then the water rose again over the stones"] * 12),
    "dangling_end": GOOD[:180] + " and then she",
}
for expect, text in cases.items():
    rep = filt.check(text)
    check(f"{expect} is caught", expect in rep.reasons, f"got {rep.reason}")

# --------------------------------------------------------------------------- #
print("\n== individual measurements ==")
check("repeat_ratio near 0 on prose", repeat_ratio(GOOD.split()) < 0.05,
      f"{repeat_ratio(GOOD.split()):.3f}")
check("repeat_ratio high on a loop", repeat_ratio(("a b c d e f " * 20).split()) > 0.8)
check("junk_char_ratio 0 on prose", junk_char_ratio(GOOD) == 0.0)
check("junk_char_ratio catches control codes", junk_char_ratio("ok" + CONTROL_JUNK) > 0.3)
check("nonlexical_ratio 0 on prose", nonlexical_ratio(GOOD.split()) == 0.0,
      f"{nonlexical_ratio(GOOD.split()):.3f}")
check("nonlexical_ratio catches vowel-free tokens",
      nonlexical_ratio("brgh vskd tmpl krxz".split()) == 1.0)
check("the vowel rule is skipped for Arabic",
      nonlexical_ratio(GOOD_AR.split(), "arabic") == 0.0)
check("function_word_ratio healthy on prose", function_word_ratio(GOOD.split()) > 0.3,
      f"{function_word_ratio(GOOD.split()):.3f}")
check("function_word_ratio works on Arabic",
      function_word_ratio(GOOD_AR.split(), "arabic") > 0.05,
      f"{function_word_ratio(GOOD_AR.split(), 'arabic'):.3f}")
check("words_per_sentence sane on prose", words_per_sentence(GOOD) < 30,
      f"{words_per_sentence(GOOD):.1f}")
check("ends_mid_sentence false on prose", not ends_mid_sentence(GOOD))
check("ends_mid_sentence true on a fragment", ends_mid_sentence("She opened the"))
check("a closing quote still counts as terminated",
      not ends_mid_sentence('He shrugged. "Fine."'))

# --------------------------------------------------------------------------- #
print("\n== tail trimming ==")
tails = {
    "code fence": "\n\n```\n",
    "markdown heading": "\n\n### Notes\n",
    "word count note": "\n\nWord count: 58 words.",
    "sign-off": "\n\nI hope you enjoyed this story! Let me know if you want changes.",
    "chat scaffold": "\n\nassistant\n",
    "punctuation run": "\n\n--------------------",
}
for label, tail in tails.items():
    out, removed = trim_tail(GOOD + tail)
    check(f"trims a {label}", out.rstrip() == GOOD.rstrip(),
          f"removed={removed} end={out[-40:]!r}")

out, removed = trim_tail(GOOD + " She reached for the")
check("cuts back to the last full sentence", out.rstrip() == GOOD.rstrip() and removed == 4,
      f"removed={removed}")

fragment = "She walked out into the rain and kept walking until the road ran out and"
out, removed = trim_tail(fragment)
check("a story that is one unterminated run is left for the checks, not trimmed",
      out == fragment and removed == 0, f"removed={removed}")

out, _ = trim_tail(GOOD + "\n\nHere is the story you asked for.")
check("trims post-story commentary", out.rstrip() == GOOD.rstrip())

mostly_meta = "A short line.\n" + "\n".join(["Note: something"] * 20)
out, removed = trim_tail(mostly_meta)
check("refuses to eat most of the text as markup",
      removed <= 0.25 * len(mostly_meta.split()),
      f"removed={removed} of {len(mostly_meta.split())}")

# Junk at the very end is a tail to remove, not a story to throw away: trimming
# cuts back to the last complete sentence and the story survives.
junk_tail = GOOD + " " + CONTROL_JUNK * 6
rep_tail = filt.check(junk_tail)
check("a junk tail is trimmed, not flagged",
      rep_tail.ok and rep_tail.trimmed_words > 0, f"{rep_tail.reason} trimmed={rep_tail.trimmed_words}")
check("trimming a junk tail leaves the story intact", rep_tail.text.rstrip() == GOOD.rstrip())

check("trimming is skipped when asked",
      CoherenceFilter(trim=False).check(GOOD + "\n\n```\n").trimmed_words == 0)

# --------------------------------------------------------------------------- #
print("\n== sentence-to-sentence coherence ==")

check("sentences split on terminals", len(split_sentences(GOOD)) == 6,
      str(len(split_sentences(GOOD))))

TOPICS = {"boat": 0, "sea": 0, "sail": 0, "harbour": 0,
          "moth": 1, "lamp": 1, "wing": 1, "dust": 2, "tax": 3, "kettle": 4}


def topic_encode(sentences):
    """Each sentence maps to the topic of the first keyword it contains, so
    sentences about the same thing embed identically and unrelated ones do not."""
    out = []
    for sent in sentences:
        v = np.zeros(8)
        hit = next((TOPICS[w] for w in sent.lower().split() if w in TOPICS), 7)
        v[hit] = 1.0
        out.append(v)
    return np.stack(out)


on_topic = ("The boat waits at the pier. The sea is flat and grey. "
            "She checks the sail once more. The harbour bell rings twice.")
salad = ("The boat waits at the pier. A moth circles the lamp. "
         "Dust settles on the sill. The tax notice arrives late. The kettle sings.")
sims = sentence_coherence([on_topic, salad, "One sentence only."], topic_encode)
check("a story that stays on topic scores high", sims[0] == 1.0, f"{sims[0]}")
check("word salad scores low", sims[1] == 0.0, f"{sims[1]}")
check("too few sentences gives nan", sims[2] != sims[2])

reps = [CoherenceFilter().check(t) for t in [on_topic, salad]]
# reference from a set of coherent stories: high similarity, a little spread
ref = nll_reference([1.0, 0.98, 1.0, 0.96, 1.0, 0.99])
apply_sentence_coherence(reps, [sims[0], sims[1]], CoherenceThresholds(), ref)
check("the on-topic story is not flagged", "incoherent" not in reps[0].reasons,
      reps[0].reason)
check("the salad is flagged incoherent", "incoherent" in reps[1].reasons,
      reps[1].reason)
check("nan similarity is never flagged",
      "incoherent" not in apply_sentence_coherence(
          [CoherenceFilter().check(GOOD)], [float("nan")],
          CoherenceThresholds(), ref)[0].reasons)


# --------------------------------------------------------------------------- #
print("\n== perplexity stage ==")


class StubPPL:
    """Returns (overall, head, tail) NLL straight from a table keyed by the text."""

    def __init__(self, table):
        self.table = table

    def score(self, texts, tail_fraction=0.2):
        return [self.table.get(t.strip()[-20:], (2.0, 2.0, 2.0)) for t in texts]


normal = [f"{GOOD} Variation number {i}." for i in range(8)]
weird = GOOD + " Odd one out here."
tailbad = GOOD + " Then everything stopped."
table = {t.strip()[-20:]: (2.0 + 0.05 * i, 2.0, 2.0) for i, t in enumerate(normal)}
table[weird.strip()[-20:]] = (9.0, 9.0, 9.0)       # far above the median
table[tailbad.strip()[-20:]] = (2.4, 2.0, 5.0)     # fine, then falls apart

filt_ppl = CoherenceFilter(CoherenceThresholds(), ppl_scorer=StubPPL(table))
reps = filt_ppl.evaluate(normal + [weird, tailbad])
check("normal stories survive the perplexity stage", all(r.ok for r in reps[:8]),
      str([r.reason for r in reps[:8]]))
check("an outlier is flagged high_perplexity", "high_perplexity" in reps[8].reasons,
      reps[8].reason)
check("a story that degrades at the end is flagged garbage_tail",
      "garbage_tail" in reps[9].reasons, reps[9].reason)
check("the tail gap is recorded", abs(reps[9].scores["nll_tail_gap"] - 3.0) < 1e-6)

check("robust_z is 0 for a constant set", robust_z([5.0] * 6) == [0.0] * 6)
zs = robust_z([1.0, 1.1, 0.9, 1.0, 1.2, 9.0])
check("robust_z isolates the outlier", zs[-1] > 5 and all(abs(z) < 3 for z in zs[:-1]),
      f"{[round(z, 2) for z in zs]}")

# --------------------------------------------------------------------------- #
print("\n== summary ==")
summary = summarise(reps)
check("summary counts", summary["n"] == 10 and summary["kept"] == 8,
      f"{summary['kept']}/{summary['n']}")
check("summary pass rate", abs(summary["pass_rate"] - 0.8) < 1e-9)
check("summary lists the reasons", summary.get("high_perplexity") == 1.0 and
      summary.get("garbage_tail") == 1.0)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("all coherence tests passed")
