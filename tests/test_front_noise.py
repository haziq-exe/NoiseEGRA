"""Noise that starts larger and falls back to its size over the opening.

    python tests/test_front_noise.py
"""
import math, sys, warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))

from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402
from noiseegra.fisher_calibration import calibrate_offset  # noqa: E402

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


DIM, LAYERS = 64, [2, 3]
NAMES = ["present_tense", "mature_register", "dialogue"]
torch.manual_seed(0)
VECS = {n: {l: torch.randn(DIM) for l in LAYERS} for n in NAMES}


def plan(front=1.0, prompt=1.0, envelope="front"):
    return SteeringPlan.build(
        VECS, LAYERS, [ConstraintSpec(n, beta=1.0) for n in NAMES], rms_scale=1.0,
        noise_mode="none", noise_alpha=0.0, steer_budget=1.0,
        offset_gamma=0.15, offset_mode="orth", offset_norm="fisher",
        offset_basis_kind="random", offset_random_rank=8, noise_beta=2.0,
        offset_prefill=True, steer_prefill=True, prompt_tail_clear=2,
        offset_envelope=envelope, offset_envelope_steps=40,
        offset_front_gain=front, offset_prefill_gain=prompt)


p = plan(2.0)
check("it starts at the gain", abs(p.envelope_at(0) - 2.0) < 1e-9)
check("is halfway back at the midpoint", abs(p.envelope_at(20) - 1.5) < 1e-9)
check("and is back to the size by the end of the opening",
      abs(p.envelope_at(40) - 1.0) < 1e-9 and abs(p.envelope_at(300) - 1.0) < 1e-9)
check("a gain of one is flat", abs(plan(1.0).envelope_at(0) - 1.0) < 1e-9)

from tiny_model import Tiny  # noqa: E402
egra = Tiny()
ids = torch.tensor([[1] + [2 + (b % 250) for b in b"<s>write a story<a>"]])
maxlen = math.sqrt(DIM)
lens = []
for front, prompt in ((1.0, 1.0), (2.0, 1.0), (2.0, 2.0)):
    q = plan(front, prompt)
    torch.manual_seed(3); q.resample_offset()
    lens.append(calibrate_offset(egra, q, ids, 0.15, max_length=maxlen, n_tokens=6)["length"])
check("the calibrated size ignores both gains", max(lens) - min(lens) < 1e-6,
      ", ".join(f"{x:.4f}" for x in lens))

PROMPT = [{"role": "system", "content": "s"}, {"role": "user", "content": "write a story"}]
egra.tokenizer.decode = lambda ids, skip_special_tokens=True: " ".join(str(int(i)) for i in ids)
a = egra.generate_with_orthogonal_steering(PROMPT, plan(1.0, 1.0), max_new_tokens=8, seed=5)
b = egra.generate_with_orthogonal_steering(PROMPT, plan(1.0, 4.0), max_new_tokens=8, seed=5)
check("the prompt's gain reaches generation", a != b, f"{a[:30]!r} vs {b[:30]!r}")

sys.argv = ["x"]
from noiseegra.setup_experiment import ExperimentSpec, _ortho_tag  # noqa: E402
t = _ortho_tag("M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=plan(2.0, 2.0)))
check("the run name records both", "__envfront40g2" in t and "__opg2" in t, t[-70:])

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all passed")
