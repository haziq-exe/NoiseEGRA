"""A random per-story split of the steering budget.

    python tests/test_steer_split.py
"""
import sys, warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


DIM, LAYERS = 64, [2, 3]
NAMES = ["present_tense", "mature_register", "dialogue", "simile"]
torch.manual_seed(0)
VECS = {n: {l: torch.randn(DIM) for l in LAYERS} for n in NAMES}


def plan(conc, betas=None):
    betas = betas or {n: 1.0 for n in NAMES}
    return SteeringPlan.build(VECS, LAYERS, [ConstraintSpec(n, beta=betas[n]) for n in NAMES],
                              rms_scale=1.0, noise_mode="none", noise_alpha=0.0,
                              steer_budget=2.0, steer_split_concentration=conc)


p = plan(1.0)
torch.manual_seed(3); p.resample_offset(); g1 = list(p.gains)
torch.manual_seed(4); p.resample_offset(); g2 = list(p.gains)
torch.manual_seed(3); p.resample_offset(); g3 = list(p.gains)
check("each story gets its own split", g1 != g2)
check("the same seed gives the same split", g1 == g3)
check("the gains average one", abs(sum(g1) / len(g1) - 1.0) < 1e-5, f"{sum(g1)/len(g1):.6f}")

lp = p.layer_plans[LAYERS[0]]
norms = []
for s in (3, 4, 5):
    torch.manual_seed(s); p.resample_offset()
    d = lp.steering_delta(0, 200, p.specs, p.rms_scale, gains=p.gains, budget=p.steer_budget)
    coef = lp.basis.t() @ d
    norms.append(float(coef.norm()))
check("the total push is the same whatever the split", max(norms) - min(norms) < 1e-3,
      ", ".join(f"{x:.4f}" for x in norms))

q = plan(1.0, {"present_tense": 1.0, "mature_register": 1.0, "dialogue": 1.0, "simile": 0.0})
torch.manual_seed(3); q.resample_offset()
lq = q.layer_plans[LAYERS[0]]
d = lq.steering_delta(0, 200, q.specs, q.rms_scale, gains=q.gains, budget=q.steer_budget)
coef = lq.basis.t() @ d
check("a rule given no share gets none, whatever its gain",
      abs(float(coef[NAMES.index("simile")])) < 1e-4, f"{float(coef[3]):.2e}")

r = plan(0.0)
torch.manual_seed(3); r.resample_offset()
check("no concentration leaves the split alone", r.gains is None or all(g == 1.0 for g in r.gains))

sys.argv = ["x"]
from noiseegra.setup_experiment import ExperimentSpec, _ortho_tag  # noqa: E402
check("the run name records the split", "__split1" in _ortho_tag(
    "M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=p)))

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all passed")
