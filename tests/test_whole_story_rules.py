#!/usr/bin/env python
"""Rules about the whole story, and a controller that needs no calibration.

The middle-school set is mostly thresholds -- six sensory words, three lines of
speech, a name used three times. A threshold is easy to satisfy by accident, it
says nothing about whether the story is any good, and it needs a number chosen
in advance. These eight ask for properties a reader could check without
counting.

The controller is the second half: every rule here is satisfied or not, so a
direction pushes while its rule is unmet and goes silent once it is met. Nothing
has to be known beforehand about how often the model breaks it, which is what a
constant coefficient set from a calibration run does need.

Offline, no GPU, no model.

    python tests/test_whole_story_rules.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from noiseegra.constraint_control import WholeStoryController  # noqa: E402
from noiseegra.constraint_metrics_en import (  # noqa: E402
    WHOLE_STORY_CONSTRAINTS, EnglishConstraintChecker, genders_present, has_simile,
)

FAIL = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAIL.append(name)


# --- none of the rules is a count -------------------------------------------
check("the set has no counting rule left",
      not any(k in WHOLE_STORY_CONSTRAINTS for k in
              ("sensory", "fresh_words", "plain_words", "varied_openers",
               "fresh_openings", "dialogue_min", "named_character",
               "no_repetition")),
      ", ".join(WHOLE_STORY_CONSTRAINTS))
check("and it keeps the one rule that is about the whole story already",
      "present_tense" in WHOLE_STORY_CONSTRAINTS)

# --- the detectors ----------------------------------------------------------
for text, want in [("The cloud looks like a dragon.", True),
                   ("It is as cold as a stone in here.", True),
                   ("I like apples.", False),
                   ("They would like the cake now.", False),
                   ("She likes him.", False)]:
    check(f"simile: {text[:38]}", has_simile(text) is want)
for text, want in [("She runs. He waits.", 2), ("She runs alone.", 1),
                   ("The dog runs.", 0), ("He finds his hat. She finds hers.", 2)]:
    check(f"genders in {text[:30]}", genders_present(text) == want, str(genders_present(text)))

ck = EnglishConstraintChecker(backend="regex", constraints=list(WHOLE_STORY_CONSTRAINTS),
                              max_words=200, min_words=10, min_grade_level=3.0)
good = ('Mia hears the gate creak behind her and the sound is like a cat complaining. '
        '"Are you coming or not?" he calls from the lane, already walking away. '
        'She pulls her coat around her shoulders and follows him into the rain.')
res = ck.evaluate_all([good])["stories"][0]
for rule in ("has_dialogue", "one_named_character", "both_genders", "simile",
             "distinct_sentences", "story_format"):
    check(f"a story written to the rules passes {rule}", res.checks.get(rule) is True,
          str(res.checks.get(rule)))

two_names = 'Mia and Sam run. "Go!" he shouts at her.'
check("two names fails the one-name rule",
      ck.evaluate_all([two_names])["stories"][0].checks.get("one_named_character") is False)

# --- the controller is silent when a rule is met ----------------------------
c = WholeStoryController(target_words=150, hard_words=200)


def errs(text):
    return {k: round(v, 2) for k, v in c.errors(text).items() if abs(v) > 0.01}


check("at the third word it asks for nothing",
      errs("She runs down") == {}, str(errs("She runs down")))

long_no_simile = "The bell rings above her head and she stops to listen to it. " * 8
e = errs(long_no_simile)
check("once the story is under way it asks for what is missing",
      e.get("simile") == 1.0 and e.get("dialogue") == 1.0
      and e.get("both_genders") == 1.0, str(e))

met = ('Mia hears the gate creak like a cat complaining and she waits there. '
       '"Are you coming?" he calls to her from the lane behind the wall. ') * 4
e2 = errs(met)
check("and goes silent on each rule the story has already met",
      e2.get("simile", 0) == 0 and e2.get("dialogue", 0) == 0
      and e2.get("both_genders", 0) == 0, str(e2))

check("a past-tense verb turns the tense direction on at once",
      errs("She walked home through the rain.").get("present_tense") == 1.0)
check("and present-tense prose leaves it off",
      errs("She walks home through the rain.").get("present_tense", 0) == 0)

check("closure is silent inside the length it was asked for",
      errs("word " * 100).get("closure", 0) == 0)
over = errs("word " * 190).get("closure", 0)
check("and rises once the story runs past it", over > 0.5, str(over))

check("naming pushes for a name when there is none",
      errs("word " * 60).get("named_character") == 1.0)
check("and against one when there are three",
      errs("Mia and Sam and Leo all run together here. " * 6).get("named_character", 0) < 0,
      str(errs("Mia and Sam and Leo all run together here. " * 6).get("named_character")))

check("every direction it watches is one the whole-story set steers",
      set(c.watched()) >= {"closure", "present_tense", "simile", "dialogue",
                           "both_genders", "named_character"},
      str(c.watched()))

print()
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
    sys.exit(1)
print("ALL TESTS PASS")
