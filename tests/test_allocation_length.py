#!/usr/bin/env python
"""A budget measured from truncated stories measures the truncation.

Half the twelve requirements ask for at least so many of something -- six
sensory words, three lines of speech, a name used three times -- and half forbid
repeating something. Cut a story off early and the first half all fail while the
second half all pass, whatever the model is like.

That is not hypothetical. The first run to divide the budget by measured failure
came back with 0% on the senses, 3% on dialogue and 3% on naming, and 100% on
every no-repeat rule, because the calibration sample generates 120 tokens by
default against a 150-word target. The whole budget went to the three rules
truncation breaks and none to the rule the model really fails seven times in ten.

This checks the refusal, on made-up stories, with no model.

    python tests/test_allocation_length.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_english_experiment import _allocate_by_shortfall  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAIL.append(name)


class Args:
    story_target = 150
    offset_basis_tokens = 120
    steer_vectors = ["present_tense", "sensory", "named_character"]
    allocate_floor = 0.0


class Axes:
    def __init__(self, texts):
        self.texts = texts


class Checker:
    """Stands in for the real scorer; every rule passes, so only length matters."""
    constraints = ["present_tense", "sensory", "named_character"]

    def evaluate_all(self, texts):
        class S:
            checks = {c: True for c in Checker.constraints}
        return {"stories": [S() for _ in texts]}


def run(words_per_story, n=32):
    story = " ".join(["word"] * words_per_story)
    return _allocate_by_shortfall(Args(), Axes([story] * n), Checker())


def refuses(fn):
    try:
        fn()
        return False
    except SystemExit:
        return True


check("stories at three fifths of the target are refused",
      refuses(lambda: run(90)),
      "90 words against a target of 150")
check("stories at the target are accepted", not refuses(lambda: run(150)))
check("stories a little short are still accepted", not refuses(lambda: run(120)))
check("stories far too short are refused", refuses(lambda: run(40)))

msg = ""
try:
    run(90)
except SystemExit as exc:
    msg = str(exc)
check("the refusal says which flag to raise",
      "--offset-basis-tokens" in msg and "120" in msg, msg[:90] + " ...")
check("and says why, not just that it refused",
      "truncation" in msg and "at least" in msg)

# No stories at all is a different case: nothing to measure, so the budget
# stays equal rather than the run being stopped.
class Empty:
    texts = []


check("an empty calibration sample leaves the budget equal instead of stopping",
      _allocate_by_shortfall(Args(), Empty(), Checker()) is None)

print()
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
    sys.exit(1)
print("ALL TESTS PASS")
