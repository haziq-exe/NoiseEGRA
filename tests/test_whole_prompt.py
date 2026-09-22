"""The whole-story rules are asked for in the middle-school instruction.

    python tests/test_whole_prompt.py

They were once given the children's instruction by falling through to it, and
every story in every run under them asked for "a young child to read".
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "scripts" / "run_english_experiment.py").read_text()
FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


check("the whole set takes the middle-school instruction",
      'if args.constraint_set in ("middle", "whole") else' in src)
check("and the merged middle-school pairs by default",
      'args.pairs = "middle_whole"' in src)
pairs = json.load(open(ROOT / "noiseegra" / "data" / "steering_pairs_en_middle_whole.json"))
want = ["present_tense", "mature_register", "dialogue", "named_character", "both_genders",
        "simile", "distinct_sentences", "no_heading", "in_story"]
missing = [w for w in want if w not in pairs["constraints"]]
check("the merged pairs cover every steered rule and the shield", not missing, str(missing))
mid = json.load(open(ROOT / "noiseegra" / "data" / "steering_pairs_en_middle.json"))["constraints"]
same = all(pairs["constraints"][k] == mid[k] for k in want if k in mid)
check("they take the middle-school pairs wherever those exist", same)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all passed")
