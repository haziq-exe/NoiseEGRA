"""The segment judge: thirds of each story, compared only within groups.

    python tests/test_segment_novelty.py

The judge is replaced by a rule (two texts are the same story when their first
words match), so the grouping, the thirds and the counts can be checked exactly.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import score_novelty as S  # noqa: E402
import segment_novelty as G  # noqa: E402

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


seg = G.segments(" ".join(f"w{i}" for i in range(9)))
check("a story splits into thirds of its words",
      seg["opening"] == "w0 w1 w2" and seg["middle"] == "w3 w4 w5" and seg["ending"] == "w6 w7 w8")

calls = []


class Tok:
    def encode(self, t, **kw):
        return t.split()[:1]


def fake_pairs(pairs, enc, tok, model, torch, device, batch, half):
    calls.append(len(pairs))
    return [0.9 if enc[i] == enc[j] else 0.0 for i, j in pairs]


S._pair_probs = fake_pairs
# 25 texts: groups of 10 are texts 0-9 and 10-19; 20-24 are left out.
texts = [("a" if i % 2 else "b") + " x" for i in range(10)] + [f"t{i} x" for i in range(15)]
mats = G.grouped_same(texts, 10, Tok(), None, None, "cpu")
check("only full groups are judged, 45 pairs each", len(mats) == 2 and calls == [90], str(calls))
check("within a group the verdicts come back in place",
      S.distinct_of(mats[0], range(10)) == 2 and S.distinct_of(mats[1], range(10)) == 10)

S.coherent = lambda texts: list(texts)
story = lambda o, m, e: " ".join([o] * 3 + [m] * 3 + [e] * 3)
# Same opening, different middles and endings.
arm = [story("once", f"mid{i}", f"end{i}") for i in range(10)]
r = G.score_arm(arm, 10, Tok(), None, None, "cpu")
check("one opening told ten ways: one opening, ten middles, ten endings",
      r["opening"]["distinct"] == [1.0] and r["middle"]["distinct"] == [10.0]
      and r["ending"]["distinct"] == [10.0] and r["whole"]["distinct"] == [1.0], str(r))

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all passed")
