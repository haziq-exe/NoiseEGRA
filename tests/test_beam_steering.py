"""Beam search over the rule-steered model, and the paths the noise search compares.

    python tests/test_beam_steering.py
"""
import sys, warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))

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
PROMPT = [{"role": "user", "content": "write a story"}]


def steer_only():
    return SteeringPlan.build(V, [2, 3], [ConstraintSpec(n, beta=1.0) for n in V], rms_scale=1.0,
        noise_mode="none", noise_alpha=0.0, steer_budget=1.0, offset_gamma=0.0, offset_mode="none",
        steer_prefill=True, prompt_tail_clear=2)


print("== beam search over the steered model ==")
p = steer_only()
out = egra.generate_with_orthogonal_steering(PROMPT, p, max_new_tokens=6, num_beams=3,
                                             num_return_sequences=3, seed=1)
check("it returns the best beam", isinstance(out, str) and out == egra.last_candidates[0])
check("and keeps every returned beam", len(egra.last_candidates) == 3)
again = egra.generate_with_orthogonal_steering(PROMPT, steer_only(), max_new_tokens=6, num_beams=3,
                                               num_return_sequences=3, seed=2)
check("deterministic: a different seed gives the same beams", again == out)
greedy = egra.generate_with_orthogonal_steering(PROMPT, steer_only(), max_new_tokens=6, do_sample=False, seed=1)
check("a single path decodes greedily", isinstance(greedy, str) and len(egra.last_candidates) == 1)

noisy = SteeringPlan.build(V, [2, 3], [ConstraintSpec(n, beta=1.0) for n in V], rms_scale=1.0,
    noise_mode="none", noise_alpha=0.0, steer_budget=1.0, offset_gamma=0.1, offset_mode="orth",
    offset_norm="energy", offset_basis_kind="random", offset_random_rank=8, noise_beta=2.0,
    offset_prefill=True, steer_prefill=True, prompt_tail_clear=2, offset_online=1.0)
try:
    egra.generate_with_orthogonal_steering(PROMPT, noisy, max_new_tokens=4, num_beams=3)
    check("beam search refuses the noise sized while writing", False)
except ValueError:
    check("beam search refuses the noise sized while writing", True)
g1 = egra.generate_with_orthogonal_steering(PROMPT, noisy, max_new_tokens=6, do_sample=False, seed=5)
g2 = egra.generate_with_orthogonal_steering(PROMPT, noisy, max_new_tokens=6, do_sample=False, seed=6)
check("greedy paths run with the full method (noise sized while writing)", isinstance(g1, str) and isinstance(g2, str))

print("\n== the search kinds (the tiny model writes the same canned text every time, so this checks the plumbing) ==")
from noiseegra.search import candidate_logprob, search  # noqa: E402
sp = steer_only()
for kind in ("beam", "sample", "noise"):
    best = search(egra, kind, 3, sp, PROMPT, seed=7, noise_plan=noisy, max_new_tokens=6)
    c = egra.last_search
    check(f"'{kind}' gives three scored candidates and returns the best",
          len(c) == 3 and best == max(c, key=lambda x: x["score"])["text"]
          and all(isinstance(x["score"], float) for x in c), str([round(x["score"], 2) for x in c]))
c = egra.last_search
check("the noise search's first path is the steered model with no noise", c[0]["path"] == 0)
t = c[0]["text"]
check("a candidate's score is its mean log-probability under the steered model",
      abs(candidate_logprob(egra, sp, PROMPT, t) - c[0]["score"]) < 1e-5 and c[0]["score"] <= 0)
from noiseegra.subspace import schedule_factor  # noqa: E402
check("NPAD's schedule anneals as 1/t from the first step",
      [schedule_factor("inv_t", t, 0) for t in range(4)] == [1.0, 0.5, 1 / 3, 0.25])
npad = SteeringPlan.build(V, [2, 3], [ConstraintSpec(n, beta=1.0) for n in V], rms_scale=1.0,
    noise_mode="iso", noise_alpha=0.1, noise_schedule="inv_t", steer_budget=1.0,
    offset_gamma=0.0, offset_mode="none", steer_prefill=True, prompt_tail_clear=2)
best = search(egra, "npad", 3, sp, PROMPT, seed=7, noise_plan=npad, max_new_tokens=6)
check("an NPAD search runs its annealed noise paths", len(egra.last_search) == 3)
d0 = npad.delta_for(2, 0, with_noise=True, with_offset=False)
d9 = npad.delta_for(2, 9, with_noise=True, with_offset=False)
st = npad.delta_for(2, 0, with_noise=False, with_offset=False)
check("its noise at step 10 is about a tenth of step 1's",
      0.02 < float((d9 - st).norm()) / float((d0 - st).norm()) < 0.5,
      f"{float((d9 - st).norm()) / float((d0 - st).norm()):.3f}")
try:
    search(egra, "noise", 3, sp, PROMPT, seed=7, max_new_tokens=4)
    check("the noise search needs the full method's plan", False)
except ValueError:
    check("the noise search needs the full method's plan", True)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all passed")
