"""Hidden context before an untouched story: the controls for the hidden daydream.

    python tests/test_hidden_context.py
"""
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))

from noiseegra.hidden_context import SENTENCES, hidden_text  # noqa: E402
from noiseegra.setup_experiment import ExperimentSpec, _spec_to_run_id  # noqa: E402

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


from tiny_model import Tiny  # noqa: E402
egra = Tiny()

print("== the hidden text ==")
lines = [s for s in SENTENCES.read_text().splitlines() if s.strip()]
check("the sentence file holds the sentences", len(lines) == 2000, str(len(lines)))
a, b, c = (hidden_text(egra, "sentence", s) for s in (11, 11, 12))
check("a sentence comes from the file, the same for the same seed", a in lines and a == b)
check("and usually another for another seed", a != c or hidden_text(egra, "sentence", 13) != a)
try:
    w1, w2, w3 = (hidden_text(egra, "words", s, 6) for s in (11, 11, 12))
    check("random words: the same for the same seed, others for another", w1 == w2 and w1 != w3,
          f"{w1!r} / {w3!r}")
except ValueError:
    print("  [skip] the stand-in vocabulary has no whole words")

print("\n== the run ==")
plain = ExperimentSpec()
check("the untouched id is unchanged", _spec_to_run_id("M", plain) == "M__BASELINE")
check("random words have their own id", _spec_to_run_id(
      "M", ExperimentSpec(hidden_context="words", hidden_tokens=32)) == "M__HIDDENwords32")
check("so does a random sentence", _spec_to_run_id(
      "M", ExperimentSpec(hidden_context="sentence")) == "M__HIDDENsentence")
seen = []
orig = egra.tokenizer.__call__ if hasattr(egra.tokenizer, "__call__") else None
_tok = egra.tokenizer


class Spy:
    def __getattr__(self, k):
        return getattr(_tok, k)

    def __call__(self, text, *a, **kw):
        seen.append(text)
        return _tok(text, *a, **kw)

    def __len__(self):
        return len(_tok)


egra.tokenizer = Spy()
out = egra.generate([{"role": "user", "content": "write a story"}], max_new_tokens=8, seed=3,
                    hidden_prefix="The committee met on Tuesday.")
egra.tokenizer = _tok
check("the model reads the hidden sentence, then a paragraph break",
      any(t.endswith("The committee met on Tuesday.\n\n") for t in seen if isinstance(t, str)))
check("and the story is generated", isinstance(out, str))
check("the hidden text is not returned", "committee" not in out)
a = egra.generate([{"role": "user", "content": "write a story"}], max_new_tokens=8, seed=3)
b = egra.generate([{"role": "user", "content": "write a story"}], max_new_tokens=8, seed=3)
check("without it, untouched writing is unchanged and reproducible", a == b)

print()
print("all passed" if not FAILURES else f"FAILED: {FAILURES}")
sys.exit(1 if FAILURES else 0)
