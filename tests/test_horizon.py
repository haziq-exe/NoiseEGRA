"""Noise drawn in the directions that move far-ahead predictions, not the next word.

    python tests/test_horizon.py
"""
import sys, warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))

from noiseegra.horizon import horizon_basis  # noqa: E402
from noiseegra.online_calibration import _reference  # noqa: E402
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


def plan(rank=0):
    return SteeringPlan.build(V, [2, 3], [ConstraintSpec(n, beta=1.0) for n in V], rms_scale=1.0,
        noise_mode="none", noise_alpha=0.0, steer_budget=1.0, offset_gamma=0.1,
        offset_mode="orth", offset_norm="energy", offset_basis_kind="random",
        offset_random_rank=8, noise_beta=2.0, offset_prefill=True, steer_prefill=True,
        prompt_tail_clear=2, offset_online=1.0, online_max_gain=10.0, online_rule_k=0.43,
        online_rule_start=True, horizon_rank=rank)


PROMPT = [{"role": "user", "content": "write a story"}]
chat = egra.apply_chat_template(PROMPT, tokenize=False, add_generation_prompt=True)
ids = egra.tokenizer(chat, return_tensors="pt")["input_ids"]

print("== the basis ==")
p = plan(4)
ref = _reference(egra, p, ids, 48)
bases, diag = horizon_basis(egra, p, ids, ref[0], rank=4, samples=8)
check("one basis per noise layer", sorted(bases) == [2, 3], str(sorted(bases)))
b = bases[2]
check("of the asked rank", b.shape == (p.dim, 4), str(tuple(b.shape)))
check("orthonormal", torch.allclose(b.t() @ b, torch.eye(4), atol=1e-4))
prot = p.layer_plans[2].protect
check("clear of the rule directions",
      prot is None or float((prot.float().cpu().t() @ b).abs().max()) < 1e-4)
check("moves far-ahead predictions more per unit of next-word movement than a random direction",
      diag["gain"] > 1.0, f"{diag['gain']:.2f}x (lowest layer {diag['gain_min']:.2f}x)")
b2, _ = horizon_basis(egra, p, ids, ref[0], rank=4, samples=8)
check("the same prompt gives the same basis", torch.allclose(b2[2], b, atol=1e-5))

print("\n== in a run ==")
rid = _ortho_tag("M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=p))
check("the run id records it", "__hz4" in rid, rid[-60:])
check("and a run without it does not", "__hz" not in _ortho_tag(
      "M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=plan(0))))
outs = [egra.generate_with_orthogonal_steering(PROMPT, p, max_new_tokens=6, seed=i) for i in range(2)]
check("stories generate", all(isinstance(o, str) and o for o in outs))
hb = getattr(p.layer_plans[2], "horizon_basis", None)
check("the basis is set on the plan once per prompt", hb is not None and hb.shape[1] == 4)
p.resample_offset(story_index=0)
off = p.layer_plans[2].offset.float().cpu()
inside = hb.float() @ (hb.float().t() @ off)
check("each story's noise lies in that basis",
      float((off - inside).norm() / off.norm().clamp_min(1e-12)) < 1e-3)
p.resample_offset(story_index=1)
off2 = p.layer_plans[2].offset.float().cpu()
check("and differs from story to story",
      float(torch.nn.functional.cosine_similarity(off, off2, dim=0)) < 0.999)
q = plan(0)
egra.generate_with_orthogonal_steering(PROMPT, q, max_new_tokens=6, seed=1)
check("without it the plan draws a random subspace",
      getattr(q.layer_plans[2], "horizon_basis", None) is None)

print()
print("all passed" if not FAILURES else f"FAILED: {FAILURES}")
sys.exit(1 if FAILURES else 0)
