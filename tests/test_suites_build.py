"""Every suite builds, and no two of its conditions share a run id.

Two runs reached Kaggle, started a session and downloaded a model before dying on
a suite that passed an argument make_plan does not take. A third would have run
to completion with two conditions writing to the same file, because the steering
mode was not in the run id and a constant arm and an error-driven arm at the same
coefficients collided -- a silent wrong answer rather than a crash.

Both are caught here in a second, against stand-in vectors. No model, no GPU.

python tests/test_suites_build.py
"""

import sys, types, warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from noiseegra.constraint_control import ConstraintController  # noqa: E402
from noiseegra.setup_experiment import _spec_to_run_id, make_specs  # noqa: E402
from noiseegra.steering_vectors import SteeringVectorSet  # noqa: E402
from run_orthosteer_experiment import build_suite  # noqa: E402

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


DIM, LAYERS = 64, [2, 3]
NAMES = ["present_tense", "simple_register", "dialogue", "terse", "varied_openers",
         "closure"]
torch.manual_seed(0)
VECS = SteeringVectorSet(
    vectors={n: {l: torch.randn(DIM) for l in LAYERS} for n in NAMES},
    components={n: {l: torch.linalg.qr(torch.randn(DIM, 8))[0] for l in LAYERS}
                for n in NAMES},
    positives={n: {l: torch.randn(DIM) for l in LAYERS} for n in NAMES},
)
ARGS = types.SimpleNamespace(
    protect_rank=8, horizon=200, steer_prefill=False, alpha=0.4, beta=1.0,
    noise_horizon=24, steer_budget=3.0, budget_sweep=[2.0, 3.0], realloc_kappa=0.6,
    steer_horizon=32, feedback_betas=[0.5, 1.0], feedback_cap=0.1,
    control_betas=[1.0, 2.0], closure_betas=[2.0, 4.0], closure_horizon=80,
    probe_beta=[3.0], beta_sweep=[1.0, 2.0], tune_betas=[1.5, 3.0], combo_beta=3.0,
    gamma_sweep=[0.1, 0.15], kappa_sweep=[0.1], alpha_sweep=[0.4], lambda_sweep=[2.0],
    keep_directions={"simple_register": 1.0, "varied_openers": -1.0},
    offset_basis={l: torch.linalg.qr(torch.randn(DIM, 24))[0] for l in LAYERS},
    amplify_basis={l: torch.linalg.qr(torch.randn(DIM, 24))[0] for l in LAYERS},
    amplify_mean={l: torch.zeros(DIM) for l in LAYERS},
    offset_basis_kind="story", direction_source="extracted",
    targets=VECS.positives, controller=ConstraintController(),
    gate_thresholds={"none": 0.0, "median": 1.0, "high": 2.0},
    baseline_temperature=1.8, baseline_top_p=0.95, baseline_top_k=40,
)

# Taken from the runner's own --suite choices rather than listed here, so a suite
# cannot be added without this test covering it. A hardcoded list let `constdose`
# through, and the run-id collision this test exists to catch is exactly the kind
# of thing a new suite introduces: two arms that differ in a setting the id does
# not record share a file, and the second silently overwrites the first.
def _suite_choices():
    import re

    import run_english_experiment as R

    src = open(R.__file__).read()
    m = re.search(r'--suite[^)]*?choices=\[(.*?)\]', src, re.S)
    if not m:
        raise SystemExit("could not read the --suite choices from the runner")
    return [x.strip().strip('"\'') for x in m.group(1).split(",") if x.strip()]


SUITES = [s for s in _suite_choices() if s != "all"]

# A suite that draws an offset from an estimated basis must be listed in
# run_english_experiment.BASIS_SUITES, or the runner never builds the basis and
# the offset silently falls back to an isotropic draw while the run id still
# reads `obstory`. This is exactly what happened to r16-spread and r17-headline.
# Here every suite is built with NO basis (offset_basis=None, as the runner
# leaves it when the suite is not in the guard set) and any suite whose run ids
# then name a story/prompt basis is required to be in the guard set.
import run_english_experiment as _R  # noqa: E402

# A suite may legitimately refuse to build against this fixed name list: the
# `weighted` suite varies the share of the push given to one named direction, and
# without it every arm would be identical. Refusing loudly is the correct
# behaviour and is what this list records; silently building identical arms is
# the fault.
NEEDS_ITS_OWN_DIRECTIONS = {"weighted", "quieten", "eventvary"}

for suite in SUITES:
    if suite in NEEDS_ITS_OWN_DIRECTIONS:
        try:
            build_suite(suite, VECS, LAYERS, list(NAMES), 1.5, ARGS)
            check(f"{suite:<11} refuses a name set it cannot vary", False,
                  "built identical arms instead of raising")
        except ValueError as exc:
            check(f"{suite:<11} refuses a name set it cannot vary",
                  "in the steered set" in str(exc), str(exc))
        continue
    try:
        built, desc = build_suite(suite, VECS, LAYERS, list(NAMES), 1.5, ARGS)
    except Exception as exc:
        check(f"{suite} builds", False, f"{type(exc).__name__}: {exc}")
        continue
    specs = make_specs(*[({"mode": it} if isinstance(it, str) else dict(it))
                         for it in built])
    ids = [_spec_to_run_id("Tiny", sp) for sp in specs]
    ok = len(set(ids)) == len(ids)
    check(f"{suite:<11} builds {len(built):>2} conditions, all with distinct ids",
          ok, "" if ok else f"{len(set(ids))} unique of {len(ids)}")
    if not desc:
        check(f"{suite} describes itself", False)

# The guard-set coverage check. Build each suite with no basis at all and see
# whether it still emits a basis-naming run id.
NO_BASIS = types.SimpleNamespace(**{**vars(ARGS), "offset_basis": None})
for suite in SUITES:
    try:
        built, _ = build_suite(suite, VECS, LAYERS, list(NAMES), 1.5, NO_BASIS)
    except Exception:
        continue
    specs = make_specs(*[({"mode": it} if isinstance(it, str) else dict(it))
                         for it in built])
    ids = [_spec_to_run_id("Tiny", sp) for sp in specs]
    names_basis = any("__obstory" in r or "__obprompt" in r for r in ids)
    if names_basis:
        check(f"{suite:<11} names an estimated basis -> is in BASIS_SUITES",
              suite in _R.BASIS_SUITES,
              "" if suite in _R.BASIS_SUITES else
              f"{suite!r} emits obstory/obprompt but is not guarded")

    # The same shape of fault one level along. A suite that sets a per-arm
    # entropy gate needs the thresholds measured first, and only the suites in
    # GATE_SUITES trigger that measurement. Without it every level resolves to
    # "no gate" and the arms are identical, while their run ids name the gates.
    # `gatedwrite` shipped that way and produced three byte-identical arms.
    names_gate = any("__gate" in r for r in ids)
    if names_gate:
        check(f"{suite:<11} names an entropy gate -> is in GATE_SUITES",
              suite in _R.GATE_SUITES,
              "" if suite in _R.GATE_SUITES else
              f"{suite!r} emits a gate tag but never has its thresholds measured")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("all suites build with distinct run ids")
