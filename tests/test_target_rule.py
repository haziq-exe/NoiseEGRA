"""The noise's target set from the model before any story.

    python tests/test_target_rule.py
"""
import math, sys, warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))

from noiseegra.online_calibration import measure_for_rule, target_from_top_share  # noqa: E402
from noiseegra.setup_experiment import ExperimentSpec, _ortho_tag  # noqa: E402
from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


print("== the rule ==")
q = target_from_top_share(0.6854, 0.762)
l = target_from_top_share(1.3450, 0.678)
check("it gives Qwen3-1.7B's best target", abs(q - 1.0) < 0.03, f"{q:.3f}")
check("and Llama-3.2-3B's", abs(l - 0.43) < 0.02, f"{l:.3f}")
check("a less certain model gets a smaller target",
      target_from_top_share(1.0, 0.5) < target_from_top_share(1.0, 0.8))
check("so does one top-p moves further", target_from_top_share(2.0, 0.7) < target_from_top_share(1.0, 0.7))
d = q * 0.6854
check("the mass the noise may move is k of the top word's probability",
      abs(math.sin(d / 2) - 0.43 * 0.762) < 1e-9)

print("\n== in a run ==")
from tiny_model import Tiny  # noqa: E402
egra = Tiny()
torch.manual_seed(0)
V = {n: {l: torch.randn(64) for l in [2, 3]} for n in ["a", "b"]}


def plan(k):
    return SteeringPlan.build(V, [2, 3], [ConstraintSpec(n, beta=1.0) for n in V], rms_scale=1.0,
        noise_mode="none", noise_alpha=0.0, steer_budget=1.0, offset_gamma=0.1, offset_mode="orth",
        offset_norm="energy", offset_basis_kind="random", offset_random_rank=8, noise_beta=2.0,
        offset_prefill=True, steer_prefill=True, prompt_tail_clear=2, offset_online=1.0,
        online_max_gain=10.0, online_rule_k=k)


p = plan(0.43)
rid = _ortho_tag("M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=p))
check("the run id records the rule", "rule0p43" in rid, rid[-50:])
PROMPT = [{"role": "user", "content": "write a story"}]
chat = egra.apply_chat_template(PROMPT, tokenize=False, add_generation_prompt=True)
ids = egra.tokenizer(chat, return_tensors="pt")["input_ids"]
ref = measure_for_rule(egra, plan(0.43), ids)
want = target_from_top_share(ref["unit"], ref["top_prob"], 0.43)
calls = []
import noiseegra.online_calibration as OC
real = OC.measure_for_rule
def counting(*a, **k):
    calls.append(1); return real(*a, **k)
OC.measure_for_rule = counting
outs = [egra.generate_with_orthogonal_steering(PROMPT, p, max_new_tokens=8, seed=s) for s in range(3)]
OC.measure_for_rule = real
check("stories generate with the rule", all(isinstance(o, str) and o for o in outs))
check("the model is measured once per prompt, not per story", len(calls) == 1, str(len(calls)))
check("with no hooks on, so it matches a measurement made outside a run",
      abs(p.offset_online - want) < 1e-6, f"{p.offset_online:.5f} vs {want:.5f}")
off = plan(0.0)
egra.generate_with_orthogonal_steering(PROMPT, off, max_new_tokens=6, seed=1)
check("without the rule the target stays as set", off.offset_online == 1.0
      and getattr(off, "_rule_cache", None) is None)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all passed")
