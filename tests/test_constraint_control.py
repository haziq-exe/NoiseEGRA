"""Steering driven by the constraint error of the text written so far.

Constant steering doubles compliance on the monotone requirements and cuts the
counting ones from 23% to 6%: a count has no "more is better" direction, so a
fixed coefficient sails past the target. Activation feedback does not fix it
either -- it saturates at "quote-like enough", because that is what the contrast
examples encode, not at "two quotes". The generation loop, however, can count.

CPU only.  python tests/test_constraint_control.py
"""

import sys, warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))

from noiseegra.constraint_control import (  # noqa: E402
    ConstraintController, ConstraintProbe, read_partial,
)
from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


C = ConstraintController()

print("== reading the story so far ==")
st = read_partial('Mira feeds the hens. "Come here," she says. She waits and')
check("words are counted", st.words == 11, str(st.words))
check("finished sentences are counted", st.sentences == 2, str(st.sentences))
check("quoted spans are counted", st.quotes == 1, str(st.quotes))
check("the sentence in progress is measured separately",
      st.words_in_current_sentence == 3, str(st.words_in_current_sentence))

print("\n== a count stops pushing the moment it is met ==")
none = C.errors('Mira feeds the hens. ' + 'word ' * 40)
one = C.errors('"One," she says. ' + 'word ' * 40)
two = C.errors('"One," she says. "Two," he says. ' + 'word ' * 40)
three = C.errors('"One," she says. "Two," he says. "Three," she adds. ' + 'word ' * 40)
check("no quoted lines yet -> push for them", none["dialogue"] > 0.9, f"{none['dialogue']:+.2f}")
check("one of two -> push less", 0 < one["dialogue"] < none["dialogue"],
      f"{one['dialogue']:+.2f}")
check("exactly two -> silent", two["dialogue"] == 0.0, f"{two['dialogue']:+.2f}")
check("three -> push the other way", three["dialogue"] < 0, f"{three['dialogue']:+.2f}")

print("\n== length pushes toward closure only once the budget is spent ==")
check("a short story is not hurried", C.errors("word " * 10)["closure"] == 0.0)
check("a story inside its band is left alone", C.errors("word " * 58)["closure"] == 0.0,
      f"{C.errors('word ' * 58)['closure']:+.2f}")
check("one running over is pushed to end", C.errors("word " * 90)["closure"] > 0.9,
      f"{C.errors('word ' * 90)['closure']:+.2f}")

print("\n== sentence length is judged on the sentence in progress ==")
check("a sentence over the ceiling wants to end",
      C.errors("She runs and runs and runs and runs and runs and runs and")["terse"] >= 0.4,
      f"{C.errors('She runs and runs and runs and runs and runs and runs and')['terse']:+.2f}")
check("one under the floor wants to continue",
      C.errors("Mira feeds. She")["terse"] < 0)
check("one inside the band is left alone",
      C.errors("Mira feeds the hens today.")["terse"] == 0.0)

print("\n== the plan turns errors into a push ==")
DIM, LAYERS = 32, [2]
NAMES = ["closure", "terse", "dialogue"]
torch.manual_seed(0)
VECS = {n: {l: torch.randn(DIM) for l in LAYERS} for n in NAMES}
state = {"errors": {}}


def plan(**kw):
    kw.setdefault("noise_mode", "none")
    kw.setdefault("noise_alpha", 0.0)
    return SteeringPlan.build(VECS, LAYERS, [ConstraintSpec(n) for n in NAMES],
                              rms_scale=1.0, **kw)


p = plan(steer_mode="error", control_state=state, controller=C, steer_budget=3.0)
p.resample_offset()
check("error mode without a control state is refused",
      _raises(lambda: plan(steer_mode="error")))
check("nothing measured yet means no push", p.error_delta(2) is None)

state["errors"] = {"closure": 0.0, "terse": 0.0, "dialogue": 0.0}
check("a story satisfying everything is written unsteered", p.error_delta(2) is None)

state["errors"] = {"closure": 1.0, "terse": 0.0, "dialogue": 0.0}
only_closure = p.error_delta(2)
B = p.layer_plans[2].basis
comp = only_closure @ B
check("only the failing requirement is pushed",
      abs(float(comp[0])) > 2.9 and abs(float(comp[1])) < 1e-5 and abs(float(comp[2])) < 1e-5,
      f"{[round(float(x), 3) for x in comp]}")
check("and the constant arm would have pushed all three",
      float((plan().delta_for(2, 0) @ B).abs().min()) > 0.4)

state["errors"] = {"closure": -1.0, "terse": 0.0, "dialogue": 0.0}
check("the sign of the error sets the direction of the push",
      float(p.error_delta(2) @ B[:, 0]) < 0)

state["errors"] = {"closure": 1.0, "terse": 1.0, "dialogue": 1.0}
three_failing = p.error_delta(2)
check("the total push is the budget whether one requirement fails or three",
      abs(float(three_failing.norm()) - float(only_closure.norm())) < 1e-4,
      f"{float(three_failing.norm()):.3f} vs {float(only_closure.norm()):.3f}")

print("\n== the probe reads the generated tokens, not the prompt ==")


class Tok:
    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i)) for i in ids)


st2 = {}
probe = ConstraintProbe(Tok(), st2, C, prompt_len=3, every=1)
text = '"One," she says. "Two," he says. ' + "word " * 40
ids = torch.tensor([[65, 66, 67] + [ord(c) for c in text]])
probe(ids, torch.zeros(1, 4))
check("the prompt is skipped", st2["text"] == text, st2["text"][:24])
check("and the errors come from what was generated",
      st2["errors"]["dialogue"] == 0.0, str(st2["errors"]))

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("all constraint-control tests passed")
