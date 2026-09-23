"""The NoveltyBench scoring that runs at the end of a Kaggle run.

    python tests/test_score_novelty.py

The judge itself is a 1.7 GB model, so a stand-in replaces it: stories are "the
same" when their first word matches. Checks the measures on cases worked by
hand, that the arms are read from a run's checkpoint through the coherence
checks, and that the summary lands in novelty.json with intervals against the
untouched model and top-p sampling.
"""
import json, sys, tempfile, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import score_novelty as S  # noqa: E402

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


print("== the measures ==")
all_same = np.ones((10, 10), dtype=bool)
none_same = np.eye(10, dtype=bool)
check("ten copies of one story are one class", S.distinct_k(all_same) == 1.0)
check("ten different stories are ten", S.distinct_k(none_same) == 10.0)
check("and share no pairs", S.share_same(none_same) == 0.0 and S.share_same(all_same) == 1.0)
two = np.zeros((10, 10), dtype=bool)
two[:5, :5] = two[5:, 5:] = True
check("two groups of five are two classes", S.distinct_k(two) == 2.0)
check("with 20 of 45 pairs the same", abs(S.share_same(two) - 20 / 45) < 1e-12)
chain = np.eye(3, dtype=bool)
chain[0, 1] = chain[1, 0] = chain[1, 2] = chain[2, 1] = True
check("a story joins the first class whose first member it matches",
      S.distinct_of(chain, [0, 2, 1]) == 2 and S.distinct_of(chain, [1, 0, 2]) == 1)
sh, di = S.half_draws(two, draws=50, inner=10)
check("half-size draws keep the measure's scale", 0.2 < sh.mean() < 0.7 and 1.0 <= di.mean() <= 2.0,
      f"share {sh.mean():.2f}, distinct {di.mean():.2f}")
lo = S.interval(np.full(50, 0.1), np.full(50, 0.4))
check("an interval is the difference and its spread",
      abs(lo[0] + 0.3) < 1e-9 and lo[1] - 1e-9 <= lo[0] <= lo[2] + 1e-9)

print("\n== a run ==")
openers = ["Maya", "Leo", "Sam", "Ava", "Noor", "Kai", "Ivy", "Omar", "Zoe", "Eli", "Ana", "Ben"]
story = ("{w} walks to the old library after school. The rain taps on the windows. "
         "She finds a map inside a book. \"Look at this,\" she says to her brother, "
         "who reads it slowly. The map shows a garden that nobody has seen for years, "
         "and they decide to go and find it together before dark.")
runs = {
    "Qwen3-1.7B__BASELINE": [story.format(w="Maya") for _ in range(12)],
    "Qwen3-1.7B__BASELINE__temp1p8__topp0p95": [story.format(w=openers[i % 3]) for i in range(12)],
    "Qwen3-1.7B__ORTHO__method": [story.format(w=openers[i]) for i in range(12)],
}
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    (d / "Qwen3-1.7B").mkdir()
    (d / "Qwen3-1.7B" / "state.json").write_text(json.dumps(
        {"runs": {r: {f"0:{i}": t for i, t in enumerate(ts)} for r, ts in runs.items()}}))
    got = S.arms(d)
    check("every arm is read from the checkpoint, in order", set(got) == set(runs)
          and got["Qwen3-1.7B__ORTHO__method"][3].startswith("Ava"))
    check("the coherence checks keep well-formed stories", len(S.coherent(got["Qwen3-1.7B__BASELINE"])) == 12)

    def fake_same(texts, *a, **k):
        first = [t.split()[0] for t in texts]
        return np.array([[a == b for b in first] for a in first], dtype=bool)

    S.load_judge = lambda device: (None, None, None)
    S.same_matrix = fake_same
    judged = S.score(d, sorted(runs), 0, 200, "cpu")
    rows = S.summarise(d, judged)
    m = rows["Qwen3-1.7B__ORTHO__method"]
    check("every story different: none of the pairs the same",
          m["same_story_share"] == 0.0 and m["distinct10"] == 10.0)
    check("one story told twelve times: all of them", rows["Qwen3-1.7B__BASELINE"]["same_story_share"] == 1.0)
    check("the method is compared with the untouched model",
          abs(m["same_vs_untouched"][0] + 1.0) < 1e-9 and m["distinct_vs_untouched"][0] > 8)
    check("and with top-p sampling", "same_vs_top_p" in m and m["same_vs_top_p"][0] < 0)
    check("the untouched model is not compared with itself",
          "same_vs_untouched" not in rows["Qwen3-1.7B__BASELINE"])
    saved = json.loads((d / "novelty.json").read_text())
    check("the summary is saved next to the stories",
          set(saved["arms"]) == set(runs) and saved["threshold"] == 0.102)

print("\n== runs of other tasks ==")
cases = ('is_palindrome("level") -> True\nis_palindrome("") -> True\nis_palindrome("abc") -> False\n'
         'is_palindrome("Noon") -> False\nis_palindrome("x y x") -> True')
check("a task's own validity check is used for its runs",
      len(S.coherent([cases, ""], "dom-tests__Qwen3-1.7B__BASELINE")) == 1)
check("where the story checks would reject good test cases as a broken story",
      len(S.coherent([cases], "Qwen3-1.7B__BASELINE")) == 0)
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    (d / "Qwen3-1.7B").mkdir()
    runs = {}
    for dom in ("poem", "plan"):
        runs[f"dom-{dom}__Qwen3-1.7B__BASELINE"] = [f"{dom} one {i} words here and more" for i in range(8)]
        runs[f"dom-{dom}__Qwen3-1.7B__BASELINE__temp1p8__topp0p95"] = [
            f"{dom} {i % 2} words here and more text" for i in range(8)]
        runs[f"dom-{dom}__Qwen3-1.7B__ORTHO__method"] = [f"{dom} {i} words here and more text"
                                                         for i in range(8)]
    (d / "Qwen3-1.7B" / "state.json").write_text(json.dumps(
        {"runs": {r: {f"0:{i}": t for i, t in enumerate(ts)} for r, ts in runs.items()}}))

    def by_second_word(texts, *a, **k):
        key = [t.split()[1] for t in texts]
        return np.array([[a == b for b in key] for a in key], dtype=bool)

    S.same_matrix = by_second_word
    rows = S.summarise(d, S.score(d, sorted(runs), 0, 100, "cpu"))
    m = rows["dom-plan__Qwen3-1.7B__ORTHO__method"]
    check("each arm is compared with its own task's untouched model",
          abs(m["same_vs_untouched"][0] - (0.0 - 1.0)) < 1e-9, str(m["same_vs_untouched"][0]))
    check("and labelled with its task", m["label"].startswith("[plan]"), m["label"])

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all passed")
