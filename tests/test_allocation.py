#!/usr/bin/env python
"""Dividing a fixed steering budget by what the model actually gets wrong.

The hand-picked set of five directions for twelve requirements included the one
requirement that never fails and left out the second worst. These checks fix the
behaviour that prevents both: a requirement that always passes must not take
budget, and a requirement that usually fails must take the most.

Offline, no GPU, no model.

    python tests/test_allocation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from noiseegra.allocation import allocation_cosine, shortfall_weights  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAIL.append(name)


# The measured Qwen3-1.7B profile, 200 stories under the middle-school prompt.
SMALL = {
    "present_tense": 0.00, "mature_register": 0.61, "dialogue_min": 0.62,
    "varied_openers": 0.30, "plain_words": 0.99, "sensory": 0.64,
    "fresh_words": 0.81, "named_character": 0.66, "no_repetition": 0.96,
    "distinct_sentences": 0.86, "fresh_openings": 0.80, "story_format": 1.00,
}
# The same twelve on Qwen3-8B, 100 stories, same prompt.
BIG = {
    "present_tense": 0.03, "mature_register": 0.88, "dialogue_min": 0.97,
    "varied_openers": 0.99, "plain_words": 1.00, "sensory": 0.65,
    "fresh_words": 1.00, "named_character": 0.82, "no_repetition": 1.00,
    "distinct_sentences": 1.00, "fresh_openings": 1.00, "story_format": 1.00,
}
NAMES = list(SMALL)

w = shortfall_weights(SMALL, NAMES)

check("the requirement that always fails takes the largest share",
      max(w, key=w.get) == "present_tense", f"largest is {max(w, key=w.get)}")
check("the requirement that never fails takes nothing",
      w["story_format"] == 0.0, f"{w['story_format']:.3f}")
check("the second worst requirement is second, not left out",
      sorted(w, key=w.get, reverse=True)[1] == "varied_openers",
      f"second is {sorted(w, key=w.get, reverse=True)[1]}")
check("a requirement passing 99% of the time gets a token share",
      0 < w["plain_words"] < 0.05, f"{w['plain_words']:.4f}")
check("weights are proportions: the largest is one",
      abs(max(w.values()) - 1.0) < 1e-12)
check("nothing is negative", min(w.values()) >= 0.0)

# The specific defect this replaces: the picked five spent budget on a
# requirement at 100% and skipped one at 30%.
picked = {"present_tense", "sensory", "named_character", "story_format",
          "something_happens"}
check("the hand-picked set contained a requirement with no shortfall at all",
      SMALL["story_format"] == 1.0 and "story_format" in picked)
check("and omitted one failing seven times in ten",
      "varied_openers" not in picked and SMALL["varied_openers"] < 0.35)

# --- the two models disagree, which is why this is measured not picked ------
cos = allocation_cosine(shortfall_weights(SMALL, NAMES),
                        shortfall_weights(BIG, NAMES))
check("the two models do not fail in the same pattern", cos < 0.9, f"cosine {cos:.3f}")
check("but they are not unrelated either", cos > 0.5, f"cosine {cos:.3f}")
check("a set measured on one model is the same as itself",
      abs(allocation_cosine(w, w) - 1.0) < 1e-12)

# The 8B's own allocation must drop what the 8B already does well, even though
# the small model needs it badly.
wb = shortfall_weights(BIG, NAMES)
check("varied openings take budget on the small model and almost none on the big one",
      w["varied_openers"] > 0.6 and wb["varied_openers"] < 0.05,
      f"{w['varied_openers']:.2f} vs {wb['varied_openers']:.3f}")

# --- the knobs ---------------------------------------------------------------
floored = shortfall_weights(SMALL, NAMES, floor=0.1)
check("a floor keeps a solved requirement alive without letting it dominate",
      0 < floored["story_format"] < 0.2, f"{floored['story_format']:.3f}")
sharp = shortfall_weights(SMALL, NAMES, power=2.0)
check("raising the power concentrates the budget on the worst requirements",
      sharp["sensory"] < w["sensory"], f"{sharp['sensory']:.3f} < {w['sensory']:.3f}")

# --- directions that are not requirements -----------------------------------
mixed = shortfall_weights(SMALL, NAMES + ["in_story"])
check("a direction with no requirement of its own is not silently switched off",
      mixed["in_story"] > 0, f"{mixed['in_story']:.3f}")
aliased = shortfall_weights(SMALL, ["dialogue", "no_heading"],
                            aliases={"dialogue": "dialogue_min",
                                     "no_heading": "story_format"})
check("a direction named differently from its requirement is matched by alias",
      aliased["dialogue"] > aliased["no_heading"],
      f"dialogue {aliased['dialogue']:.2f} vs no_heading {aliased['no_heading']:.2f}")

# --- fallbacks and refusals --------------------------------------------------
perfect = shortfall_weights({n: 1.0 for n in NAMES}, NAMES)
check("when nothing fails, every direction is treated alike rather than zeroed",
      all(v == 1.0 for v in perfect.values()))


def refuses(fn):
    try:
        fn()
        return False
    except ValueError:
        return True


check("an empty direction list is refused", refuses(lambda: shortfall_weights(SMALL, [])))
check("a negative floor is refused",
      refuses(lambda: shortfall_weights(SMALL, NAMES, floor=-1.0)))
check("a non-positive power is refused",
      refuses(lambda: shortfall_weights(SMALL, NAMES, power=0.0)))

print()
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
    sys.exit(1)
print("ALL TESTS PASS")
