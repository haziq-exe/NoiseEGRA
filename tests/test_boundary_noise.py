"""Noise at every sentence-ending token, for the whole story.

    python tests/test_boundary_noise.py
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


def plan(bg=0.0):
    return SteeringPlan.build(V, [2, 3], [ConstraintSpec(n, beta=1.0) for n in V], rms_scale=1.0,
        noise_mode="none", noise_alpha=0.0, steer_budget=1.0, offset_gamma=0.1,
        offset_mode="orth", offset_norm="energy", offset_basis_kind="random",
        offset_random_rank=8, noise_beta=2.0, offset_prefill=True, steer_prefill=True,
        prompt_tail_clear=2, offset_online=1.0, online_max_gain=10.0, online_rule_k=0.43,
        online_rule_start=True, offset_envelope="plateau", offset_envelope_steps=4,
        steer_envelope=True, boundary_gain=bg)


print("== the schedule ==")
p = plan(4.0)
p.resample_offset(story_index=0)


def off(t, at):
    p._at_boundary = at
    a = p.delta_for(2, t, with_noise=False, with_offset=True)
    b = p.delta_for(2, t, with_noise=False, with_offset=False)
    a = torch.zeros(p.dim) if a is None else a.float()
    b = torch.zeros(p.dim) if b is None else b.float()
    return (a - b).norm().item()


check("long after the fade, no noise between sentence ends", off(40, False) < 1e-6, f"{off(40, False):.4f}")
check("but at a sentence end, four times the starting length", off(40, True) > 3.9 * off(1, False),
      f"{off(40, True):.3f} vs {off(1, False):.3f}")
check("during the first span a sentence end gets the larger of the two", off(1, True) >= off(1, False))
p0 = plan(0.0); p0.resample_offset(story_index=0); p0._at_boundary = True
d = p0.delta_for(2, 40, with_noise=False, with_offset=True)
e = p0.delta_for(2, 40, with_noise=False, with_offset=False)
check("off, a sentence end changes nothing",
      (d is None and e is None) or torch.allclose(d if d is not None else torch.zeros(1),
                                                  e if e is not None else torch.zeros(1)))

print("\n== in a run ==")
b = egra._boundary_ids()
check("the vocabulary has sentence-ending entries", len(b) > 0, str(len(b)))
check("the run id records it", "__bnd4" in _ortho_tag(
      "M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=plan(4.0))))
check("and a run without it does not", "__bnd" not in _ortho_tag(
      "M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=plan(0.0))))
seen = []
pr = plan(4.0)
orig = pr.delta_for


def spy(layer, t, **kw):
    if layer == 2:
        seen.append(bool(getattr(pr, "_at_boundary", False)))
    return orig(layer, t, **kw)


pr.delta_for = spy
out = egra.generate_with_orthogonal_steering([{"role": "user", "content": "write a story"}], pr,
                                             max_new_tokens=16, seed=3)
check("stories generate with it", isinstance(out, str))
check("the hook sees sentence ends while writing, and not every step", any(seen) and not all(seen),
      f"{sum(seen)} of {len(seen)}")
check("the shadow stayed on the story's words", int(getattr(pr, "shadow_drift", 0) or 0) == 0)
a = egra.generate_with_orthogonal_steering([{"role": "user", "content": "write a story"}], plan(0.0),
                                           max_new_tokens=8, seed=3)
c = egra.generate_with_orthogonal_steering([{"role": "user", "content": "write a story"}], plan(0.0),
                                           max_new_tokens=8, seed=3)
check("off, a run is unchanged and reproducible", a == c)

print()
print("all passed" if not FAILURES else f"FAILED: {FAILURES}")
sys.exit(1 if FAILURES else 0)
