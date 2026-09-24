"""The scale probe: what a model is like under the noise, before any story.

    python tests/test_scale_probe.py
"""
import sys, warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests")); sys.path.insert(0, str(ROOT / "scripts"))

from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402
from noiseegra.scale_probe import probe  # noqa: E402
from tiny_model import Tiny  # noqa: E402
import probe_scale  # noqa: E402

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


egra = Tiny()
torch.manual_seed(0)
V = {n: {l: torch.randn(64) for l in [2, 3]} for n in ["a", "b"]}
plan = SteeringPlan.build(V, [2, 3], [ConstraintSpec(n, beta=1.0) for n in V], rms_scale=1.0,
    noise_mode="none", noise_alpha=0.0, steer_budget=1.0, offset_gamma=0.1, offset_mode="orth",
    offset_norm="energy", offset_basis_kind="random", offset_random_rank=8, noise_beta=2.0,
    offset_prefill=True, steer_prefill=True, prompt_tail_clear=2, offset_online=1.0)
ids = torch.tensor([[1] + [2 + (b % 250) for b in b"<s>write a story<a>"]])
r = probe(egra, plan, ids, seeds=[1, 2], n_tokens=6, fractions=(0.05, 0.4))
check("it measures top-p's shift per step", r["unit"] > 0 and r["norm"] == 8.0)
check("and the model's uncertainty", 0 < r["top_prob"] <= 1 and r["entropy"] > 0)
bf = r["by_fraction"]
check("a longer noise moves the predictions further",
      bf[0.4]["moved"] > bf[0.05]["moved"] > 0, f"{bf[0.05]['moved']:.3f} < {bf[0.4]['moved']:.3f}")
check("in top-p's units too", abs(bf[0.4]["units"] - bf[0.4]["moved"] / r["unit"]) < 1e-9)
check("and reports the fluency it costs", all("nll_rise" in d for d in bf.values()))
cmds = list(probe_scale.commands("/o", ["Qwen3-1.7B:6:14", "Qwen3-4B:8:17"], ["--probe-scale", "8"]))
check("the wrapper runs each model on its own band into its own folder",
      cmds[1][0] == "Qwen3-4B" and "--layers" in cmds[1][1]
      and cmds[1][1][cmds[1][1].index("--layers") + 1:][:2] == ["8", "17"]
      and cmds[1][1][-1] == "/o/Qwen3-4B")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all passed")
