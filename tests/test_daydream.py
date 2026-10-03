"""A hidden daydream before the story: strong noise, a forced break, then the story.

    python tests/test_daydream.py
"""
import sys, warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))

from noiseegra.setup_experiment import ExperimentSpec, _ortho_tag  # noqa: E402
from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


from tiny_model import Tiny  # noqa: E402
egra = Tiny()
torch.manual_seed(0)
V = {n: {l: torch.randn(64) for l in [2, 3]} for n in ["a", "b"]}


def plan(dd=0, gain=4.0):
    return SteeringPlan.build(V, [2, 3], [ConstraintSpec(n, beta=1.0) for n in V], rms_scale=1.0,
        noise_mode="none", noise_alpha=0.0, steer_budget=1.0, offset_gamma=0.1,
        offset_mode="orth", offset_norm="energy", offset_basis_kind="random",
        offset_random_rank=8, noise_beta=2.0, offset_prefill=True, steer_prefill=True,
        prompt_tail_clear=2, offset_online=1.0, online_max_gain=10.0, online_rule_k=0.43,
        online_rule_start=True, offset_envelope="plateau", offset_envelope_steps=4,
        steer_envelope=True, daydream_steps=dd, daydream_gain=gain)


print("== the schedule ==")
p = plan(5, 4.0)
p._daydream_shift = 6                      # five daydream words and a one-token break
p.resample_offset(story_index=0)
lp = p.layer_plans[2]
def _d(t, with_offset):
    d = p.delta_for(2, t, with_noise=False, with_offset=with_offset)
    return torch.zeros(p.dim) if d is None else d.float()


off = lambda t: (_d(t, True) - _d(t, False)).norm().item()
check("the noise is boosted during the daydream", off(2) > 3.0 * off(8),
      f"{off(2):.3f} vs {off(8):.3f}")
check("and the story's schedule starts after it: full for its first span",
      abs(off(6) - off(9)) / off(9) < 0.25, f"{off(6):.3f} vs {off(9):.3f}")
check("then fades as before", off(6 + 8) < 0.05 * off(6), f"{off(14):.4f}")
st = lambda t: _d(t, False).norm().item()
check("the steering is not boosted", abs(st(2) - st(8)) / st(8) < 1e-5, f"{st(2):.3f} vs {st(8):.3f}")

print("\n== in a run ==")
PROMPT = [{"role": "user", "content": "write a story"}]
pd = plan(5, 4.0)
check("the run id records it", "__dd5g4" in _ortho_tag(
      "M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=pd)))
check("and a run without it does not", "__dd" not in _ortho_tag(
      "M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=plan(0))))
full = egra.generate_with_orthogonal_steering(PROMPT, pd, max_new_tokens=8, seed=3)
check("stories generate with it", isinstance(full, str))
check("the shadow stayed on the story's words",
      int(getattr(pd, "shadow_drift", 0) or 0) == 0, str(getattr(pd, "shadow_drift", 0)))
brk = egra.tokenizer("\n\n", add_special_tokens=False)["input_ids"]
check("the hidden part is the daydream plus the break", pd._daydream_shift == 5 + len(brk))
p0 = plan(0)
a = egra.generate_with_orthogonal_steering(PROMPT, p0, max_new_tokens=8, seed=3)
b = egra.generate_with_orthogonal_steering(PROMPT, plan(0), max_new_tokens=8, seed=3)
check("off, a run is unchanged and reproducible", a == b and p0._daydream_shift == 0)

print()
print("all passed" if not FAILURES else f"FAILED: {FAILURES}")
sys.exit(1 if FAILURES else 0)
