"""The English runner's own seeds honour --story-seed-offset.

    python tests/test_runner_seed_offset.py

An offset was once added to a seed function this runner does not call, verified
on that function, and a "second 25 stories" run reproduced the first 25 word for
word. This checks the function the runner uses and the call that uses it.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "scripts" / "run_english_experiment.py").read_text()
ns = {}
exec(src[src.index("def seed_for("):src.index("def write_csvs(")], ns)
seed_for = ns["seed_for"]

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


old = lambda p, s: (42 + p * 100003 + s * 7919) % (2 ** 31)
check("no offset leaves every seed as it was",
      all(seed_for(p, k) == old(p, k) for p in range(3) for k in range(300)))
a = {seed_for(0, k) for k in range(1000)}
b = {seed_for(0, k, 7001) for k in range(1000)}
check("an offset gives a stream sharing no seed with the first", not (a & b))
check("the generation loop passes the offset",
      "seed_for(p_idx, k, args.story_seed_offset)" in src)
check("the leak diagnostic seeds as the generation loop does",
      "seed_for(0, x, args.story_seed_offset)" in src and "seed_for_story" not in src)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all passed")
