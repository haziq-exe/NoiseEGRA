"""The published comparison methods, run through the same runner as everything else.

    python tests/test_prior_methods.py

Checks the parsers on replies worked by hand (a full Verbalized Sampling JSON
reply, one cut off mid-way, String Seed of Thought's tags with and without the
closing one); that each method gets a run id of its own and a readable name;
that grouped methods make a group once and hand it out in order; that STARS's
hook and the noise injection's hooks fire during generation and are gone after;
and that every method writes stories end to end on a tiny model.
"""
import sys, types, warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

import noiseegra.prior_methods as P  # noqa: E402
from noiseegra.run_labels import label_run  # noqa: E402
from noiseegra.setup_experiment import _spec_mode, _spec_to_run_id, make_specs  # noqa: E402

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


print("== parsing ==")
full = ('{"responses": [{"text": "A girl named Ana runs.", "probability": 0.3}, '
        '{"text": "He says \\"hi\\" to her.", "probability": 0.2}]}')
check("a Verbalized Sampling reply gives its texts", P.parse_vs(full) ==
      ["A girl named Ana runs.", 'He says "hi" to her.'])
cut = full[:full.index('"probability": 0.2')]
check("and a reply cut off mid-way still gives what it finished",
      P.parse_vs("Sure!\n" + cut)[:2] == ["A girl named Ana runs.", 'He says "hi" to her.'])
check("text with no JSON gives nothing", P.parse_vs("Once upon a time.") == [])
s = "<random_string>x9!Q</random_string><thinking>use x</thinking><answer>\nThe story.\n</answer>"
check("String Seed of Thought's answer is what is inside its tags", P.parse_ssot(s) == "The story.")
check("an unclosed answer runs to the end", P.parse_ssot("<answer> Tale") == "Tale")
check("with no answer tags, what follows the thinking",
      P.parse_ssot("<thinking>a</thinking> Tale.") == "Tale.")
sysmsg = P.vs_system(5, 150)
check("the VS instruction asks for 5 of about 150 words with probabilities",
      "Generate 5 responses" in sysmsg and "approximately 150 words" in sysmsg
      and "'probability'" in sysmsg)

m2 = P.creative_messages(MSG0 := [{"role": "system", "content": "s"}, {"role": "user", "content": "Write a story."}])
check("the creative prompt ends the user's request by asking for a very unique story",
      m2[1]["content"] == "Write a story.\n\nBe creative and think of a very unique story."
      and MSG0[1]["content"] == "Write a story." and m2[0] == MSG0[0])

print("\n== run ids ==")
items = [{"prior_method": m, "prior_params": {"n": 3}} for m in P.METHODS]
specs = make_specs(*items)
ids = [_spec_to_run_id("M", sp) for sp in specs]
check("each method is its own mode", all(_spec_mode(sp) == "prior_method" for sp in specs))
check("and has its own run id", len(set(ids)) == len(ids), ids[0])
sid = ids[P.METHODS.index("stars")]
check("with its name readable", label_run(sid).text.startswith("STARS"), label_run(sid).text)
try:
    make_specs({"prior_method": "stars", "plan": object()})
    bad = _spec_mode(make_specs({"prior_method": "stars", "plan": object()})[0])
    check("a prior method cannot also be steered", False)
except ValueError:
    check("a prior method cannot also be steered", True)

print("\n== on a model ==")
from tiny_model import Tiny  # noqa: E402
egra = Tiny()
MSG = [{"role": "system", "content": "write"}, {"role": "user", "content": "a story"}]

calls = []
real = egra.generate
def counting(*a, **k):
    calls.append(k.get("seed")); return real(*a, **k)
egra.generate = counting
P._CACHE.clear()
out = [P.generate_prior(egra, "incontext", {"n": 3}, MSG, seed=1, story_index=i,
                        max_new_tokens=6) for i in range(6)]
check("in-context regeneration writes a conversation of n per group",
      len(calls) == 6 and len(out) == 6 and all(isinstance(o, str) for o in out))
calls.clear()
P.generate_prior(egra, "incontext", {"n": 3}, MSG, seed=1, story_index=4, max_new_tokens=6)
check("and a story already written is not written again", calls == [])

seen = {}
orig = egra.generate
def spy(msgs, **k):
    seen["msgs"] = msgs
    return '{"responses": [{"text": "One tale.", "probability": 0.5}]}'
egra.generate = spy
P._CACHE.clear()
v = [P.generate_prior(egra, "verbalized", {"k": 3, "words": 150}, MSG, seed=1, story_index=i,
                      max_new_tokens=6) for i in range(3)]
check("Verbalized Sampling keeps the task and adds its instruction to the system turn",
      seen["msgs"][0]["content"].startswith("write") and "Generate 3 responses" in seen["msgs"][0]["content"]
      and seen["msgs"][1]["content"] == "a story")
check("responses it did not return count as failed stories", v == ["One tale.", "", ""])
egra.generate = orig

got = {}
def spy2(msgs, **k):
    got["sys"] = msgs[0]["content"]
    return "<random_string>q</random_string><thinking>t</thinking><answer>A tale.</answer>"
egra.generate = spy2
t = P.generate_prior(egra, "ssot", {}, MSG, seed=1, story_index=0, max_new_tokens=6)
check("String Seed of Thought puts its system prompt first and returns the answer",
      got["sys"].startswith("You are a helpful AI Assistant") and got["sys"].endswith("write")
      and t == "A tale.")
egra.generate = real

blocks = egra._get_transformer_blocks()
fired = {"o": 0, "mlp": 0}
h1 = blocks[3].self_attn.o_proj.register_forward_pre_hook(lambda m, i: fired.__setitem__("o", fired["o"] + 1))
P._CACHE.clear()
st = [P.generate_prior(egra, "stars", {"n": 4, "layer": 3, "C": 0.1}, MSG, seed=1, story_index=i,
                       max_new_tokens=5) for i in range(4)]
h1.remove()
check("STARS writes its batch together", len(st) == 4 and all(isinstance(x, str) for x in st))
check("with its hook running at every step", fired["o"] >= 5, f"{fired['o']} calls")
check("and gone afterwards",
      len(blocks[3].self_attn.o_proj._forward_pre_hooks) == 0)
H = torch.randn(4, 64)
V = P.svd_one_step(H)
check("its update has one column per story", V.shape == (64, 4) and bool(torch.isfinite(V).all()))
x = torch.randn(4, 3, 64)
before = blocks[3].self_attn.o_proj(x.clone())
hk = P._stars_hook(blocks[3].self_attn.o_proj, 0.1)
blocks[3].self_attn.o_proj(x.clone())
after = blocks[3].self_attn.o_proj(x.clone())
hk.remove()
check("and it changes what the projection is given", not torch.allclose(before, after))

# Noise injection: look at the first noised MLP (its input is untouched) from inside the generation.
ids = torch.tensor([[1, 5, 9, 12]])
clean = {}
hk = blocks[6].mlp.register_forward_hook(lambda m, i, o: clean.__setitem__("o", o.detach().clone()))
egra.model(ids)
hk.remove()
noised = {}
def peek(msgs, **k):
    h = blocks[6].mlp.register_forward_hook(lambda m, i, o: noised.__setitem__("o", o.detach().clone()))
    egra.model(ids)
    h.remove()
    return "A tale."
egra.generate = peek
a = P.generate_prior(egra, "noiseinject", {"alpha": 0.07, "lo": 6, "hi": 8}, MSG, seed=3,
                     story_index=0, max_new_tokens=5)
egra.generate = real
d = (noised["o"] - clean["o"])[0]
check("noise injection adds one draw to the MLP output",
      a == "A tale." and torch.allclose(d, d[0:1].expand_as(d), atol=1e-5),
      f"spread across positions {float((d - d[0]).abs().max()):.1e}")
check("uniform between zero and alpha", float(d.min()) >= -1e-6 and float(d.max()) < 0.07 + 1e-6
      and float(d.mean()) > 0.02, f"mean {float(d.mean()):.3f}")
check("and leaves no hooks behind",
      all(len(blocks[l].mlp._forward_hooks) == 0 for l in range(len(blocks))))

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all passed")
