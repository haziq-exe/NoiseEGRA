"""The headline suite: which arms it builds, and what each one is.

    python tests/test_headline_suite.py

By default it builds the noise sized while writing, sized before, two fixed
lengths, and the first length with the prompt or the start at 1.5x. Asked for a
subset -- the arms a young-child run is missing, say -- it builds exactly those,
with the base length for the additions given separately from the fixed arms.
"""
import math, sys, types, warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from noiseegra.setup_experiment import _spec_to_run_id, make_specs  # noqa: E402
from noiseegra.steering_vectors import SteeringVectorSet  # noqa: E402
from run_orthosteer_experiment import build_suite  # noqa: E402

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


DIM, LAYERS = 64, [2, 3]
NAMES = ["present_tense", "mature_register", "dialogue"]
torch.manual_seed(0)
VECS = SteeringVectorSet(
    vectors={n: {l: torch.randn(DIM) for l in LAYERS} for n in NAMES},
    components={n: {l: torch.linalg.qr(torch.randn(DIM, 8))[0] for l in LAYERS} for n in NAMES},
    positives={n: {l: torch.randn(DIM) for l in LAYERS} for n in NAMES},
)
RMS = 1.5
NORM = RMS * math.sqrt(DIM)


BASE = dict(protect_rank=8, horizon=200, steer_prefill=False, alpha=0.4, beta=1.0,
            noise_horizon=24, steer_budget=2.5, tail_sweep=[8], noise_beta_sweep=[2.0],
            offset_random_rank=16, front_tokens=40, baseline_temperature=1.8,
            baseline_top_p=0.95, baseline_top_k=40, offset_basis=None,
            offset_basis_kind="random", direction_source="extracted", keep_directions={})


def build(**kw):
    args = types.SimpleNamespace(**{**BASE, **kw})
    items, desc = build_suite("headline", VECS, LAYERS, list(NAMES), RMS, args)
    return [it["plan"] for it in items], desc


def length(p):
    return p.offset_gamma * NORM


plans, _ = build()
check("by default six arms", len(plans) == 6, str(len(plans)))
w, b, f1, f2, fp, ff = plans
check("the first is sized while writing, from a tenth of the norm, prompt at 1.5x",
      w.offset_online == 1.0 and w.offset_norm == "energy"
      and abs(w.offset_gamma - 0.10) < 1e-9 and w.offset_prefill_gain == 1.5)
check("the second is sized before, prompt at 1.5x",
      b.offset_norm == "fisher" and b.offset_online == 0 and b.offset_prefill_gain == 1.5)
check("then fixed at 14.83 and 18", abs(length(f1) - 14.83) < 1e-6 and abs(length(f2) - 18.0) < 1e-6
      and f1.offset_online == 0 and f1.offset_prefill_gain == 1.0)
check("then 14.83 with the prompt at 1.5x", abs(length(fp) - 14.83) < 1e-6 and fp.offset_prefill_gain == 1.5)
check("and 14.83 starting at 1.5x over 40 tokens",
      abs(length(ff) - 14.83) < 1e-6 and ff.offset_envelope == "front"
      and ff.offset_front_gain == 1.5 and ff.offset_envelope_steps == 40)

plans, desc = build(headline_arms=["while", "fixed", "fixedprompt", "fixedfront"],
                    fixed_lengths=[18.0], fixed_base=14.83)
check("asked for the four a young-child run lacks, it builds four", len(plans) == 4, str(len(plans)))
check("sized while writing, fixed 18, and 14.83 with each addition",
      plans[0].offset_online == 1.0 and abs(length(plans[1]) - 18.0) < 1e-6
      and abs(length(plans[2]) - 14.83) < 1e-6 and plans[2].offset_prefill_gain == 1.5
      and abs(length(plans[3]) - 14.83) < 1e-6 and plans[3].offset_envelope == "front")
check("with no sized-before arm and no plain 14.83",
      not any(p.offset_norm == "fisher" for p in plans)
      and not any(abs(length(p) - 14.83) < 1e-6 and p.offset_prefill_gain == 1.0
                  and p.offset_envelope == "flat" for p in plans))
ids = [_spec_to_run_id("Tiny", s) for s in make_specs(*[{"plan": p} for p in plans])]
check("every arm has its own run id", len(set(ids)) == len(ids))
plans, desc = build(headline_arms=["fixed"], fixed_lengths=[14.83], headline_budgets=[3.0, 3.5])
check("a steering sweep builds the chosen arms at each strength",
      len(plans) == 2 and [p.steer_budget for p in plans] == [3.0, 3.5]
      and all(abs(length(p) - 14.83) < 1e-6 for p in plans), str([p.steer_budget for p in plans]))
ids = [_spec_to_run_id("Tiny", s) for s in make_specs(*[{"plan": p} for p in plans])]
check("and records the strength in each run id", len(set(ids)) == 2
      and "bud3__" in ids[0] and "bud3p5" in ids[1], ids[0][-70:])
try:
    build(headline_arms=[])
    check("an empty choice is refused", False)
except ValueError:
    check("an empty choice is refused", True)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all passed")
