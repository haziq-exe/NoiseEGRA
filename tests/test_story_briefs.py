"""The story briefs: the middle-school prompt unchanged, the others differ only in reader and story.

    python tests/test_story_briefs.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import noiseegra.writingprompts as wp  # noqa: E402
from noiseegra.constraint_metrics_en import EnglishConstraintChecker, WHOLE_STORY_CONSTRAINTS  # noqa: E402

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


ck = EnglishConstraintChecker(constraints=list(WHOLE_STORY_CONSTRAINTS), min_grade_level=3.0,
                              present_ratio_threshold=0.9)
req, order = ck.requirements(), list(WHOLE_STORY_CONSTRAINTS)
m = wp.build_middle_messages(req, order, target=150)
EXPECTED = (
    "Write one short story for a middle-school reader.\n\n"
    "Aim for roughly 150 words -- long enough for something to happen in it.\n\n"
    "The story must satisfy every one of these requirements:\n"
    "- it is written in the present tense throughout (at least 90% of its verbs)\n"
    "- the sentences have some length and range to them rather than being as short as "
    "possible (Flesch-Kincaid grade at least 3)\n"
    "- somebody speaks out loud, inside quotation marks\n"
    "- exactly one character is given a name; anyone else is referred to without one\n"
    "- two characters appear, one referred to as he and one as she\n"
    "- something is compared to something else, with 'like' or 'as ... as'\n"
    "- no sentence is written twice\n\n"
    "Write only the story itself: no title, heading, preamble or commentary.")
check("the middle-school prompt is word for word what every earlier run was given",
      m[1]["content"] == EXPECTED and m[0]["content"] == wp.MIDDLE_SYSTEM)
for b in ("fable", "mystery", "scifi"):
    mb = wp.build_middle_messages(req, order, target=150, brief=b)
    sys_b, opening = wp.STORY_BRIEFS[b]
    rest_same = mb[1]["content"].split("\n\n", 1)[1] == m[1]["content"].split("\n\n", 1)[1]
    check(f"{b}: its own reader and story, everything after the first line unchanged",
          mb[0]["content"] == sys_b and mb[1]["content"].startswith(opening) and rest_same)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all passed")
