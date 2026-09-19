#!/usr/bin/env python
"""A setting that applies to the whole run must reach every suite.

The displacement-size spread is a flag on the command line, so it belongs to the
run. It was threaded through by naming it at each place a plan is built -- and it
got named at two of the twenty-odd of them. Every other suite ran at zero while
the launch command said 0.5, and a three-arm run generated, finished and was
scored before the run ids showed the setting missing.

So the run's value lives in one place, `RUN_DEFAULTS`, and a suite that does not
sweep the setting inherits it. This checks that every suite does, against
stand-in vectors, with no model and no GPU.

python tests/test_run_wide_settings.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import run_orthosteer_experiment as ro  # noqa: E402
from noiseegra.setup_experiment import _spec_to_run_id, make_specs  # noqa: E402
from test_suites_build import ARGS, LAYERS, NAMES, SUITES, VECS  # noqa: E402

# `varysize` is the suite whose whole job is to sweep the spread, so it passes
# its own value at every arm and must not be overridden by the run's.
SWEEPS_IT = {"varysize"}
# These refuse this fixed name list for their own reasons, checked elsewhere.
SKIP = {"weighted", "quieten", "eventvary", "baseline", "sampling"}

SPREAD = 0.5


def test_every_suite_inherits_the_run_wide_spread() -> None:
    ro.RUN_DEFAULTS["offset_gamma_spread"] = SPREAD
    try:
        checked = 0
        for suite in SUITES:
            if suite in SKIP:
                continue
            try:
                built, _ = ro.build_suite(suite, VECS, LAYERS, list(NAMES), 1.5, ARGS)
                specs = make_specs(*built)
            except Exception:
                continue
            plans = [sp.steering_plan for sp in specs
                     if getattr(sp, "steering_plan", None) is not None
                     and getattr(sp.steering_plan, "offset_gamma", 0.0)]
            if not plans:
                continue
            checked += 1
            for plan in plans:
                got = float(getattr(plan, "offset_gamma_spread", 0.0) or 0.0)
                if suite in SWEEPS_IT:
                    continue
                assert got == SPREAD, (
                    f"suite '{suite}' built a perturbed condition at a "
                    f"displacement-size spread of {got}, but the run was "
                    f"launched at {SPREAD}. Its call site is not inheriting "
                    f"the run's value."
                )
        assert checked >= 5, f"only {checked} suites were exercised; the check is hollow"
        print(f"{checked} suites build perturbed conditions and all inherit "
              f"the run's spread of {SPREAD}")
    finally:
        ro.RUN_DEFAULTS["offset_gamma_spread"] = 0.0


def test_the_spread_reaches_the_run_id() -> None:
    """A setting that does not reach the id lets two arms overwrite each other."""
    ro.RUN_DEFAULTS["offset_gamma_spread"] = SPREAD
    try:
        built, _ = ro.build_suite("boundary", VECS, LAYERS, list(NAMES), 1.5, ARGS)
        ids = [_spec_to_run_id("M", sp) for sp in make_specs(*built)]
    finally:
        ro.RUN_DEFAULTS["offset_gamma_spread"] = 0.0
    named = [i for i in ids if "__gs" in i]
    assert named, ("no condition's run id records the displacement-size spread, so "
                   f"a run at {SPREAD} and one at zero would share a file:\n  " +
                   "\n  ".join(ids))
    print(f"{len(named)} of {len(ids)} ids name the spread")


if __name__ == "__main__":
    test_every_suite_inherits_the_run_wide_spread()
    test_the_spread_reaches_the_run_id()
    print("\nok")
