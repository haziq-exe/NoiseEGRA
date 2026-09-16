"""Coherence checks. No model downloads: the perplexity stage uses a stub."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from noiseegra.coherence import (  # noqa: E402
    CoherenceFilter, CoherenceThresholds, apply_sentence_coherence, detect_script,
    ends_mid_sentence, function_word_ratio, junk_char_ratio, near_dup_sentence_ratio,
    nll_reference, nonlexical_ratio, quote_density, repeat_ratio, robust_z,
    sentence_coherence, split_sentences, summarise, tiny_sentence_run, trim_lead,
    trim_tail, window_entropy_min, words_per_sentence,
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

# --------------------------------------------------------------------------- #
# The four checks added after stories that passed everything above were read and
# found broken. Each broken example below is modelled on a real story from a run
# (r15-directions and r16-spread), not invented.
print("\n== loops the n-gram rules miss ==")

# A slot-substitution loop: the vocabulary is tiny but the rotating names keep
# any five-gram from repeating often enough for the n-gram rule.
SLOT_LOOP = (
    'The tree said, said Mia. The tree said, said Mom. The leaves said, said Mia. '
) * 5
check("window entropy is healthy on prose", window_entropy_min(GOOD) > 2.3,
      f"{window_entropy_min(GOOD):.2f}")
check("window entropy collapses on a slot loop", window_entropy_min(SLOT_LOOP) < 1.7,
      f"{window_entropy_min(SLOT_LOOP):.2f}")
check("window entropy is nan on too-short text",
      window_entropy_min("Too short to judge.") != window_entropy_min("Too short to judge."))
check("the filter flags the slot loop", "vocab_loop" in filt.check(SLOT_LOOP).reasons,
      filt.check(SLOT_LOOP).reason)

# The affect stall: sentences come back with one word changed, never verbatim.
STALL = (
    "She went to the park. The park was quiet. She felt happy. She was happy. "
    "She felt good. She smiled again. She felt happy. She was so happy. "
    "She felt good. She was happy. She felt happy then. She was happy. She felt good."
)
check("near-dup ratio is low on prose", near_dup_sentence_ratio(GOOD) < 0.1,
      f"{near_dup_sentence_ratio(GOOD):.2f}")
check("near-dup ratio is high on the stall", near_dup_sentence_ratio(STALL) > 0.3,
      f"{near_dup_sentence_ratio(STALL):.2f}")
check("the filter flags the stall", "stalled" in filt.check(STALL).reasons,
      filt.check(STALL).reason)

# Collapse into fragments: pieces too short to hold an n-gram.
FRAGMENTS = (
    "Mia holds the balloon and laughs at the sky. Mia. Balloon. Fly. Mia. Balloon. "
    "Fly. Mia. Balloon. Fly. The wind. Blow. Mia. Balloon. Fly."
)
check("tiny-run is short on prose", tiny_sentence_run(GOOD) < 3,
      str(tiny_sentence_run(GOOD)))
check("tiny-run is long on fragment collapse", tiny_sentence_run(FRAGMENTS) >= 8,
      str(tiny_sentence_run(FRAGMENTS)))
check("the filter flags the fragments", "fragments" in filt.check(FRAGMENTS).reasons,
      filt.check(FRAGMENTS).reason)
# Terse prose is the style this task asks for; a run of four two-word sentences
# is legitimate and must NOT fire.
TERSE = ("A boy runs outside. He jumps. He shouts. Mom looks. She smiles. "
         "She touches his hand. The sun shines. The air is warm. They play together "
         "in the garden until the light goes.")
check("terse but coherent prose passes", filt.check(TERSE).ok, filt.check(TERSE).reason)

# Quote salad: quotation marks opening and closing at random, several per
# sentence, the way the dialogue-direction probe wrote at strength 3.
SALAD = ('"The mat said" "It\'s cold," Ben said." "The cat said" "I want" "a '
         'sweater," I said." "The mat said" said Sam." "I need" "a coat," said '
         'Mia." "The cat said" "I said" said Ben."')
check("quote density is low on prose", quote_density(GOOD) < 1.5,
      f"{quote_density(GOOD):.2f}")
check("quote density is high on salad", quote_density(SALAD) > 3.0,
      f"{quote_density(SALAD):.2f}")

print("\n== refusals are not stories ==")
REFUSAL = ("I'm sorry, but I can't generate content that includes the words you "
           "asked for. Would you like me to write something else instead?")
check("a refusal is flagged", "refusal" in filt.check(REFUSAL).reasons,
      filt.check(REFUSAL).reason)
IN_STORY = ('Mia drops the cup. "I\'m sorry," she says. Mom smiles and helps her '
            'clean the floor. They laugh together and pour another glass of milk. '
            'The kitchen is warm and the day goes on.')
check("a character apologising is not a refusal",
      "refusal" not in filt.check(IN_STORY).reasons, filt.check(IN_STORY).reason)

print("\n== the leading preamble is trimmed ==")
lead, n = trim_lead("Certainly! Here's a short story for young readers:\n\n" + GOOD)
check("preamble removed", lead == GOOD and n > 0, f"removed {n} words")
lead, n = trim_lead("Here is the sun. It shines on the sea.\nMira watches it.")
check("a story opening with 'Here is' is left alone", n == 0)
rep = filt.check("Sure! Here's a story:\n\n" + GOOD)
check("the filter trims the preamble and passes the story", rep.ok and rep.trimmed_words > 0,
      f"{rep.reason}, trimmed {rep.trimmed_words}")

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
