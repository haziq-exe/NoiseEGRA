"""Hidden context before an untouched story: the controls for the hidden daydream.

    python tests/test_hidden_context.py
"""
import sys, warnings
import torch
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

print("\n== generated hidden openings ==")
from noiseegra.hidden_context import generate_after_hidden  # noqa: E402
P = [{"role": "user", "content": "write a story"}]
check("a latent daydream has its own id", _spec_to_run_id(
      "M", ExperimentSpec(hidden_context="latent:0.2")) == "M__HIDDENlatent0p2")
check("so does a self-written one", _spec_to_run_id(
      "M", ExperimentSpec(hidden_context="self")) == "M__HIDDENself")
s1 = generate_after_hidden(egra, P, "self", seed=5, n_tokens=4, max_new_tokens=8)
s2 = generate_after_hidden(egra, P, "self", seed=5, n_tokens=4, max_new_tokens=8)
check("a self-written opening, then a story, reproducibly", isinstance(s1, str) and s1 == s2)
blocks = egra._get_transformer_blocks()
seen = []
h = blocks[2].register_forward_hook(lambda m, i, o: seen.append(1))
l1 = generate_after_hidden(egra, P, "latent:0.2", seed=5, n_tokens=4, layers=(2, 3),
                           max_new_tokens=8)
h.remove()
l2 = generate_after_hidden(egra, P, "latent:0.2", seed=5, n_tokens=4, layers=(2, 3),
                           max_new_tokens=8)
l0 = generate_after_hidden(egra, P, "latent:0.0", seed=5, n_tokens=4, layers=(2, 3),
                           max_new_tokens=8)
check("a latent daydream, then a story, reproducibly", isinstance(l1, str) and l1 == l2)
check("the hooks are removed afterwards", len(blocks[2]._forward_hooks) == 0)
lb = generate_after_hidden(egra, P, "latent:0.2", seed=6, n_tokens=4, layers=(2, 3),
                           max_new_tokens=8)
check("it runs at another seed and with no push", isinstance(lb, str) and isinstance(l0, str))

import noiseegra.hidden_context as HC  # noqa: E402
check("a noisy opening has its own id", _spec_to_run_id(
      "M", ExperimentSpec(hidden_context="noisy:0.1")) == "M__HIDDENnoisy0p1")
seen_states = []
_orig = HC._turning_push


def _spy(*a, **kw):
    st, hs = _orig(*a, **kw)
    seen_states.append(st)
    return st, hs


HC._turning_push = _spy
n1 = generate_after_hidden(egra, P, "noisy:0.5", seed=5, n_tokens=4, layers=(2, 3),
                           max_new_tokens=8)
HC._turning_push = _orig
st = seen_states[-1]
check("the push fires while the opening is sampled", st.get("fired", 0) > 0 and st["t"] > 0,
      str(st))
check("at two layers per opening step, not on the prompt",
      st.get("fired", 0) <= 2 * st["t"], str(st))
check("and is off for the story", st["on"] is False and len(blocks[2]._forward_hooks) == 0
      and len(egra.model._forward_pre_hooks) == 0)
n2 = generate_after_hidden(egra, P, "noisy:0.5", seed=5, n_tokens=4, layers=(2, 3),
                           max_new_tokens=8)
check("a noisy opening, then a story, reproducibly", isinstance(n1, str) and n1 == n2)

from noiseegra.hidden_context import generate_selected_opening  # noqa: E402
check("a selected opening has its own id", _spec_to_run_id(
      "M", ExperimentSpec(hidden_context="select:6:0.2")) == "M__HIDDENselect6s0p2")
g1 = generate_selected_opening(egra, P, seed=5, k=3, sigma=0.5, layers=(2, 3), n_tokens=4,
                               max_new_tokens=10, story_index=0)
g2 = generate_selected_opening(egra, P, seed=5, k=3, sigma=0.5, layers=(2, 3), n_tokens=4,
                               max_new_tokens=10)
check("a story continues from the chosen opening, reproducibly", isinstance(g1, str) and g1 == g2)
check("and no hooks are left behind", len(blocks[2]._forward_hooks) == 0
      and len(egra.model._forward_pre_hooks) == 0)

import noiseegra.hidden_context as HC2  # noqa: E402
check("a weight-noise story has its own id", _spec_to_run_id(
      "M", ExperimentSpec(hidden_context="weights:0.05")) == "M__HIDDENweights0p05")
mats = HC2._down_projections(egra, (2, 3))
check("the perturbed maps write into the residual stream", len(mats) > 0, str(len(mats)))
before = [m.weight.detach().clone() for m in mats]
w1 = HC2.generate_with_weight_noise(egra, P, seed=5, rho=0.05, layers=(2, 3), max_new_tokens=8)
check("a story is written", isinstance(w1, str))
check("and the weights are restored exactly afterwards",
      all(torch.equal(b, m.weight) for b, m in zip(before, mats)))
w2 = HC2.generate_with_weight_noise(egra, P, seed=5, rho=0.05, layers=(2, 3), max_new_tokens=8)
check("reproducibly", w1 == w2)

print()
print("all passed" if not FAILURES else f"FAILED: {FAILURES}")
sys.exit(1 if FAILURES else 0)
