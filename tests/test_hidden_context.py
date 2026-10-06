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

print("\n== a random sentence's meaning where the reply is planned ==")
import noiseegra.hidden_context as HC3  # noqa: E402
d1, t1 = HC3.transplant_delta(egra, 5, 2, 1.0)
d2, t2 = HC3.transplant_delta(egra, 5, 2, 2.0)
check("the push comes from the same sentence the sentence control shows", t1 == hidden_text(egra, "sentence", 5))
check("and scales with its size", torch.allclose(d2, 2 * d1))
di, _ = HC3.transplant_delta(egra, 5, 2, 1.0, iso=True)
check("the control is a different direction of the same length",
      abs(float(di.norm() - d1.norm())) < 1e-4 and float(torch.nn.functional.cosine_similarity(di, d1, 0)) < 0.9)
other, _ = HC3.transplant_delta(egra, 6, 2, 1.0)
check("each story draws its own", not torch.allclose(other, d1))
s1 = HC3.generate_with_transplant(egra, P, 5, 1.0, (2, 3), max_new_tokens=8, story_index=0)
s2 = HC3.generate_with_transplant(egra, P, 5, 1.0, (2, 3), max_new_tokens=8)
plain = egra.generate(P, max_new_tokens=8, do_sample=True, seed=5)
check("a story is written, reproducibly", isinstance(s1, str) and s1 == s2)
seen_logits = []
_orig_gen = egra.generate
def _spy(prompt, **kw):
    ids = egra.tokenizer(egra.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True),
                         return_tensors="pt")["input_ids"]
    seen_logits.append(egra.model(input_ids=ids).logits[0, -1].detach().clone())
    return _orig_gen(prompt, **kw)
egra.generate = _spy
HC3.generate_with_transplant(egra, P, 5, 5.0, (2, 3), max_new_tokens=4)
egra.generate = _orig_gen
ids = egra.tokenizer(egra.apply_chat_template(P, tokenize=False, add_generation_prompt=True),
                     return_tensors="pt")["input_ids"]
clean_logits = egra.model(input_ids=ids).logits[0, -1]
check("the push changes what the model predicts after the prompt",
      not torch.allclose(seen_logits[0], clean_logits, atol=1e-4),
      f"max change {float((seen_logits[0] - clean_logits).abs().max()):.4f}")
check("and nothing is left hooked afterwards", egra.generate(P, max_new_tokens=8, do_sample=True, seed=5) == plain)
check("the run id names it", _spec_to_run_id("M", ExperimentSpec(hidden_context="transplant:1.0")) == "M__HIDDENtransplant1"
      and _spec_to_run_id("M", ExperimentSpec(hidden_context="transplantiso:2.0:4")) == "M__HIDDENtransplantiso2p4",
      _spec_to_run_id("M", ExperimentSpec(hidden_context="transplantiso:2.0:4")))

print("\n== a random direction grown into one the model amplifies ==")
import noiseegra.hidden_context as HC4  # noqa: E402
ids = egra.tokenizer(egra.apply_chat_template(P, tokenize=False, add_generation_prompt=True),
                     return_tensors="pt")["input_ids"]
t0, f0, l0 = HC4.amplified_direction(egra, ids, 3, 0.3, 0, 1, 3)
t5, f5, l5 = HC4.amplified_direction(egra, ids, 3, 0.3, 5, 1, 3)
check("both start from the same random direction", abs(f0 - f5) < 1e-6)
check("the grown direction keeps its length", abs(float(t5.norm() - t0.norm())) < 1e-3 * float(t0.norm()))
check("and moves the planning state further than the random start", l5 > f5, f"{f5:.4f} -> {l5:.4f}")
check("nothing is left needing gradients", not any(p.requires_grad for p in egra.model.parameters()))
a1 = HC4.generate_with_amplified(egra, P, 3, 0.3, 3, (1, 2), max_new_tokens=8, story_index=0)
a2 = HC4.generate_with_amplified(egra, P, 3, 0.3, 3, (1, 2), max_new_tokens=8)
check("a story is written, reproducibly", isinstance(a1, str) and a1 == a2)
check("and nothing stays hooked", egra.generate(P, max_new_tokens=8, do_sample=True, seed=5) == plain)
check("the run id names it", _spec_to_run_id("M", ExperimentSpec(hidden_context="amplify:0.25:8")) == "M__HIDDENamplify0p25s8",
      _spec_to_run_id("M", ExperimentSpec(hidden_context="amplify:0.25:8")))

print("\n== a hidden story idea ==")
from noiseegra.hidden_context import generate_after_hidden as gah  # noqa: E402
calls = []
_mg = egra.model.generate
def _spy_gen(**kw):
    calls.append(kw["input_ids"][0].tolist())
    return _mg(**kw)
egra.model.generate = _spy_gen
s0 = gah(egra, P, "premise:0", 4, n_tokens=32, layers=(1, 2), max_new_tokens=8, story_index=0)
egra.model.generate = _mg
cue_ids = egra.tokenizer("Story idea:", add_special_tokens=False)["input_ids"]
check("the idea is written after the cue", calls[0][-len(cue_ids):] == cue_ids)
check("and the story after the idea, the cue included", calls[1][len(calls[0]) - len(cue_ids):][:len(cue_ids)] == cue_ids)
s1 = gah(egra, P, "premise:0.3", 4, n_tokens=32, layers=(1, 2), max_new_tokens=8, story_index=0)
check("a story follows it", isinstance(s1, str) and "Story idea" not in s1)
check("the run id names it", _spec_to_run_id("M", ExperimentSpec(hidden_context="premise:0.2")) == "M__HIDDENpremise0p2")

print()
print("all passed" if not FAILURES else f"FAILED: {FAILURES}")
sys.exit(1 if FAILURES else 0)
