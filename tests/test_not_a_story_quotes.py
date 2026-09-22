"""A character speaking is not the model answering the asker.

    python tests/test_not_a_story_quotes.py

The not-a-story check looked for phrases like "you've got" in the opening, and
found them in characters' dialogue: two middle-school stories that were plainly
stories -- a football coach telling Jake he has the ball -- counted as not
stories. The rules ask for speech, so this struck the stories the task wants.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from noiseegra.coherence import is_not_a_story  # noqa: E402

FAILURES = []


def check(name, cond):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


check("dialogue in straight quotes is a story",
      not is_not_a_story('"You\'ve got the ball, Jake." He says, the words slithering out.'))
check("dialogue in curly quotes is a story",
      not is_not_a_story("“You’ve got the look of someone lost,” she says."))
check("a story told in the second person is a story",
      not is_not_a_story("You tread cautiously through the forest, leaves crunching underfoot."))
check("the model offering ideas is not",
      is_not_a_story("Ah! I see you've discovered your new adventure. If you're feeling inspired, here are ideas."))
check("the model greeting the reader is not",
      is_not_a_story("Hello! I'm excited to meet you. How about we talk about your favorite hobbies?"))
check("the same phrase outside quotation marks is still caught",
      is_not_a_story("You've got a great idea there. Would you like me to write a story about it?"))

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all passed")
