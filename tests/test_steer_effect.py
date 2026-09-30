"""The rule steering sized by its effect, and the noise's divergence diagnostic.

    python tests/test_steer_effect.py
"""
import sys, warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))

from noiseegra.constraint_metrics_en import WHOLE_STORY_CONSTRAINTS, WHOLE_STORY_SCORED  # noqa: E402
from noiseegra.online_calibration import budget_for_effect, noise_divergence  # noqa: E402
from noiseegra.setup_experiment import ExperimentSpec, _ortho_tag  # noqa: E402
from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


print("== the scored rules ==")
check("he/she and simile are not scored", "both_genders" not in WHOLE_STORY_SCORED
      and "simile" not in WHOLE_STORY_SCORED)
check("the other six are, in order",
      list(WHOLE_STORY_SCORED) == [c for c in WHOLE_STORY_CONSTRAINTS
                                   if c not in ("both_genders", "simile")])

from tiny_model import Tiny  # noqa: E402
egra = Tiny()
torch.manual_seed(0)
V = {n: {l: torch.randn(64) for l in [2, 3]} for n in ["a", "b"]}


def plan(effect=0.0, budget=1.0, k=0.43):
    return SteeringPlan.build(V, [2, 3], [ConstraintSpec(n, beta=1.0) for n in V], rms_scale=1.0,
        noise_mode="none", noise_alpha=0.0, steer_budget=budget, offset_gamma=0.1,
        offset_mode="orth", offset_norm="energy", offset_basis_kind="random",
        offset_random_rank=8, noise_beta=2.0, offset_prefill=True, steer_prefill=True,
        prompt_tail_clear=2, offset_online=1.0, online_max_gain=10.0, online_rule_k=k,
        online_rule_start=True, steer_effect=effect)


PROMPT = [{"role": "user", "content": "write a story"}]
chat = egra.apply_chat_template(PROMPT, tokenize=False, add_generation_prompt=True)
ids = egra.tokenizer(chat, return_tensors="pt")["input_ids"]

print("\n== measuring the steering's effect ==")
p = plan()
m1 = budget_for_effect(egra, p, ids)
check("the given budget moves the predictions", m1["at_given"] > 0, f"{m1['at_given']:.4f}")
check("measuring leaves the budget as it was", p.steer_budget == 1.0 and m1["budget"] == 1.0)
p2 = plan(budget=2.0)
m2 = budget_for_effect(egra, p2, ids)
check("a larger budget moves them further", m2["at_given"] > m1["at_given"],
      f"{m1['at_given']:.4f} -> {m2['at_given']:.4f}")
goal = 0.5 * (m1["at_given"] + m2["at_given"])
s = budget_for_effect(egra, plan(), ids, goal)
check("sizing to an effect reaches it", abs(s["moves"] - goal) / goal < 0.05 and s["reached"],
      f"{s['moves']:.4f} vs {goal:.4f}")
check("by a budget between the two that bracket it", 1.0 < s["budget"] < 2.0, f"{s['budget']:.3f}")
far = budget_for_effect(egra, plan(), ids, 10 * m2["at_given"])
check("an effect out of reach is reported as such", far["reached"] == 0.0)
s0 = budget_for_effect(egra, plan(budget=0.0), ids, goal)
check("no steering, nothing to size", s0["budget"] == 0.0 and s0["at_given"] == 0.0)

print("\n== in a run ==")
ps = plan(effect=goal)
rid = _ortho_tag("M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=ps))
check("the run id records the sizing", "__se" in rid, rid[-60:])
check("and a run without it does not",
      "__se" not in _ortho_tag("M", ExperimentSpec(use_orthogonal_steering=True,
                                                   steering_plan=plan())))
outs = [egra.generate_with_orthogonal_steering(PROMPT, ps, max_new_tokens=6, seed=i) for i in range(2)]
check("stories generate with the sizing", all(isinstance(o, str) and o for o in outs))
check("the budget is set by the effect, once",
      abs(ps.steer_budget - s["budget"]) / s["budget"] < 1e-6, f"{ps.steer_budget:.4f}")
pk = plan()
egra.generate_with_orthogonal_steering(PROMPT, pk, max_new_tokens=6, seed=1)
check("without it the budget stays as given", pk.steer_budget == 1.0)

print("\n== the noise's divergence ==")
from noiseegra.online_calibration import _reference  # noqa: E402
pd = plan()
ref = _reference(egra, pd, ids, 12)
dv = noise_divergence(egra, pd, ids, ref[0], 0.0)
check("no noise, the text never departs", dv["departed"] == 0.0 and dv["first"] == dv["tokens"],
      str(dv))
big = noise_divergence(egra, pd, ids, ref[0], 50.0)
check("a large noise departs early", big["first"] < dv["first"], str(big))

print()
print("all passed" if not FAILURES else f"FAILED: {FAILURES}")
sys.exit(1 if FAILURES else 0)
