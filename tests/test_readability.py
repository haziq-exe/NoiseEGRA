"""The vocabulary-difficulty measure, and what it catches that grade level does not.

python tests/test_readability.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from noiseegra.readability import common_words, uncommon_word_share  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(f"{'[ok] ' if cond else '[FAIL] '}{name}" + (f"  {detail}" if detail else ""))
    if not cond:
        failures.append(name)


print("== the familiar-word list ==")
cw = common_words()
check("the list loaded", len(cw) == 3000, f"{len(cw)} words")
check("everyday words are in it", all(w in cw for w in ["the", "dog", "run", "little", "水".replace("水", "play")]))
check("rare words are not", not any(w in cw for w in ["luminous", "ravenous", "kaleidoscope"]))

print("\n== the measure ==")
easy = "The little dog runs to the park and plays with a big red ball in the sun."
hard = "The ravenous marsupial scrutinised the luminous kaleidoscope with palpable trepidation."
e, h = uncommon_word_share([easy, hard])
check("easy prose scores low", e < 0.15, f"{e:.2f}")
check("rare vocabulary scores high", h > 0.5, f"{h:.2f}")
check("it separates them", h > e + 0.4, f"easy={e:.2f} hard={h:.2f}")

named = "Mirabelle runs. Mirabelle jumps. Mirabelle plays with the ball in the sun."
with_names = uncommon_word_share([named])[0]
without = uncommon_word_share([named], skip_names=False)[0]
check("a character's name is not counted as hard vocabulary", with_names < without,
      f"names skipped={with_names:.2f} names counted={without:.2f}")

check("a story with no countable words returns 0", uncommon_word_share(["..."])[0] == 0.0)

# The demonstration that motivates the module: chopping text into fragments
# makes the grade level look easier while the vocabulary is unchanged.
whole = "Mia walks to the park with her mother and sees a small brown dog."
chopped = "Mia walks. To the park. With mother. She sees. A small brown dog."
a, b = uncommon_word_share([whole, chopped])
check("chopping does not change the vocabulary score", abs(a - b) < 0.1,
      f"whole={a:.2f} chopped={b:.2f}")

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("all readability tests passed")
