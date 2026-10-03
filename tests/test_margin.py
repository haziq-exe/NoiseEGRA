"""The noise's push on each decision, scaled by the clean model's sureness there.

    python tests/test_margin.py
"""
import sys, warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))

from noiseegra.online_calibration import MarginScaler, reference_margin  # noqa: E402
from noiseegra.setup_experiment import ExperimentSpec, _ortho_tag  # noqa: E402
from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


print("== the scaler ==")
ms = MarginScaler(1.0, power=1.0)
clean = torch.tensor([3.0, 0.0, -1.0])          # margin 3 nats over the runner-up
story = torch.tensor([2.5, 0.4, -1.0])
out = ms(None, torch.stack([story, clean]))
check("a sure step gets a larger push, along the noise's own direction",
      torch.allclose(out[0], clean + 3.0 * (story - clean), atol=1e-4), str(out[0].tolist()))
check("the clean row is left alone", torch.equal(out[1], clean))
unsure = torch.tensor([0.1, 0.0, -1.0])         # margin 0.1
out = ms(None, torch.stack([unsure + torch.tensor([0.0, 0.3, 0.0]), unsure]))
check("an unsure step gets a smaller one, never below a quarter",
      abs(ms.history[-1] - 0.25) < 1e-6, f"{ms.history[-1]:.3f}")
huge = torch.tensor([50.0, 0.0, -1.0])
ms(None, torch.stack([huge + 0.1, huge]))
check("and never above four times", abs(ms.history[-1] - 4.0) < 1e-6)
check("no noise, no change", torch.equal(ms(None, torch.stack([clean, clean]))[0], clean))
p = torch.softmax(torch.tensor([[2.0, 0.0, -1.0], [0.0, 0.0, 0.0], [5.0, 1.0, 0.0]]), -1)
check("the reference margin is the median top-two log-probability gap",
      abs(reference_margin(p) - 2.0) < 1e-5, f"{reference_margin(p):.3f}")

inside = torch.tensor([True, False, True])        # entry 1 continues a word
mw = MarginScaler(1.0, power=1.0, word_start=inside)
cont = torch.tensor([0.0, 3.0, -1.0])               # sure, but of the rest of a word
mw(None, torch.stack([cont + 0.1, cont]))
check("the inside of a word is never amplified", mw.history[-1] <= 1.0, f"{mw.history[-1]:.2f}")
mw(None, torch.stack([clean + 0.1, clean]))
check("a sure choice of the next word still is", mw.history[-1] > 1.0, f"{mw.history[-1]:.2f}")

print("\n== in a run ==")
from tiny_model import Tiny  # noqa: E402
egra = Tiny()
torch.manual_seed(0)
V = {n: {l: torch.randn(64) for l in [2, 3]} for n in ["a", "b"]}


def plan(scale=0.0):
    return SteeringPlan.build(V, [2, 3], [ConstraintSpec(n, beta=1.0) for n in V], rms_scale=1.0,
        noise_mode="none", noise_alpha=0.0, steer_budget=1.0, offset_gamma=0.1,
        offset_mode="orth", offset_norm="energy", offset_basis_kind="random",
        offset_random_rank=8, noise_beta=2.0, offset_prefill=True, steer_prefill=True,
        prompt_tail_clear=2, offset_online=1.0, online_max_gain=10.0, online_rule_k=0.43,
        online_rule_start=True, margin_scale=scale)


PROMPT = [{"role": "user", "content": "write a story"}]
p1 = plan(1.0)
check("the run id records it", "__ms1" in _ortho_tag(
      "M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=p1)))
check("and a run without it does not", "__ms" not in _ortho_tag(
      "M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=plan(0.0))))
outs = [egra.generate_with_orthogonal_steering(PROMPT, p1, max_new_tokens=8, seed=i) for i in range(2)]
check("stories generate with it", all(isinstance(o, str) and o for o in outs))
r = next(iter(p1._rule_cache.values()))
check("the reference margin is measured once per prompt", r.get("ref_margin", 0) > 0,
      f"{r.get('ref_margin', 0):.3f}")
mask = egra._word_start_mask()
check("the word-start mask covers the whole vocabulary",
      mask.dtype == torch.bool and mask.numel() == egra.model.config.vocab_size, str(mask.numel()))
check("the shadow stayed on the story's words",
      int(getattr(p1, "shadow_drift", 0) or 0) == 0, str(getattr(p1, "shadow_drift", 0)))
same = egra.generate_with_orthogonal_steering(PROMPT, plan(0.0), max_new_tokens=8, seed=0)
again = egra.generate_with_orthogonal_steering(PROMPT, plan(0.0), max_new_tokens=8, seed=0)
check("off, a run is unchanged and reproducible", same == again)

print()
print("all passed" if not FAILURES else f"FAILED: {FAILURES}")
sys.exit(1 if FAILURES else 0)
