"""Noise that grows as the story secures the rules that stay met.

    python tests/test_secured_noise.py
"""
import sys, warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))

from noiseegra.constraint_control import ConstraintProbe, WholeStoryController  # noqa: E402
from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


ctl = WholeStoryController()
check("an empty story has secured nothing", ctl.secured("") == 0.0)
full = ('Mia walks to the pier. The water shines like a mirror. "Look," she says. '
        'Her brother waves, and he laughs.')
check("a story with a simile, speech, a he and a she and one name has secured all four",
      ctl.secured(full) == 1.0, f"{ctl.secured(full)}")
half = "Mia walks to the pier. The water shines like a mirror."
check("a simile and a name are half", ctl.secured(half) == 0.5, f"{ctl.secured(half)}")

DIM, LAYERS = 64, [2, 3]
NAMES = ["present_tense", "simile", "dialogue"]
torch.manual_seed(0)
VECS = {n: {l: torch.randn(DIM) for l in LAYERS} for n in NAMES}
state = {}
p = SteeringPlan.build(VECS, LAYERS, [ConstraintSpec(n, beta=1.0) for n in NAMES],
                       rms_scale=1.0, noise_mode="none", noise_alpha=0.0,
                       offset_gamma=1.0, offset_mode="orth", offset_norm="energy",
                       offset_basis_kind="random", offset_random_rank=8, noise_beta=2.0,
                       offset_envelope="secured", offset_secured_boost=1.0,
                       steer_mode="error", control_state=state, controller=ctl)
check("before the controller has read anything the noise is at its base",
      p.envelope_at(5) == 1.0)
state["secured"] = 0.5
check("half secured raises it by half the boost", abs(p.envelope_at(5) - 1.5) < 1e-9)
state["secured"] = 1.0
check("all secured doubles it at a boost of 1", abs(p.envelope_at(5) - 2.0) < 1e-9)
torch.manual_seed(1); p.resample_offset()
lp = p.layer_plans[LAYERS[0]]
check("the offset's length follows the envelope",
      abs(float(lp.offset_at(3, p.envelope_at(3)).norm()) - 2.0 * lp.offset_length) < 1e-4)


class _Tok:
    def decode(self, ids, skip_special_tokens=True):
        return full


st = {}
probe = ConstraintProbe(_Tok(), st, ctl, prompt_len=2, every=1)
probe(torch.tensor([[1, 2, 3, 4]]), torch.zeros(1, 5))
check("the probe records what has been secured", st.get("secured") == 1.0,
      str(st.get("secured")))

sys.argv = ["x"]
from noiseegra.setup_experiment import ExperimentSpec, _ortho_tag  # noqa: E402
t = _ortho_tag("M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=p))
check("the run name records the envelope and its boost", "__envsecured1" in t, t[-60:])

st3 = {"secured": 1.0}
ConstraintProbe(_Tok(), st3, ctl, prompt_len=2)
check("a new story starts with nothing secured", st3["secured"] == 0.0)

print("\n== end to end on a 64-wide model ==")
from tiny_model import Tiny  # noqa: E402
egra = Tiny()
PROMPT = [{"role": "system", "content": "s"}, {"role": "user", "content": "write a story"}]
st2 = {}
q = SteeringPlan.build(VECS, LAYERS, [ConstraintSpec(n, beta=2.0) for n in NAMES],
                       rms_scale=1.0, noise_mode="none", noise_alpha=0.0,
                       offset_gamma=0.3, offset_mode="orth", offset_norm="fisher",
                       offset_basis_kind="random", offset_random_rank=8, noise_beta=2.0,
                       offset_prefill=True, steer_prefill=True, prompt_tail_clear=2,
                       offset_envelope="secured", offset_secured_boost=1.0,
                       steer_mode="error", control_state=st2, controller=ctl,
                       steer_budget=1.0)
out = [egra.generate_with_orthogonal_steering(PROMPT, q, max_new_tokens=12, seed=s)
       for s in range(2)]
check("steering by the story with sized, secured noise generates",
      all(isinstance(o, str) and o for o in out))
check("the controller read the story", "secured" in st2, str(sorted(st2)))
check("and each story was sized", len(getattr(q, "fisher_log", []) or []) == 2)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all passed")
