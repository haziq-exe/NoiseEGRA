#!/usr/bin/env python
"""The closure direction, quiet early and pressing late.

The failures this method has left are stories that do not end. At the quoted
setting the six rejected stories run 292 words against 206 for the ones kept,
and the repetition starts at word 220 of 291. A direction meaning "bring it to
an end", on a schedule that is silent while a story is inside its budget and
presses harder the longer it runs over, is the brake for exactly that: measured
on the children's task it took looping from 48% of stories to 15% and improved
compliance at the same time.

These checks are that the schedule reaches the plan and that the flag refuses to
do nothing quietly -- the failure mode this project keeps hitting is an arm that
silently equals its own control.

Offline, no GPU, no model.

    python tests/test_closure_ramp.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noiseegra.subspace import schedule_factor  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAIL.append(name)


# --- the schedule shape is the one the brake needs --------------------------
H = 200
early = schedule_factor("ramp", 0, H)
mid = schedule_factor("ramp", H // 2, H)
late = schedule_factor("ramp", H, H)
over = schedule_factor("ramp", H * 2, H)
check("it is silent at the first token", early == 0.0, f"{early}")
check("it presses harder as the story runs", early < mid < late, f"{early} < {mid} < {late}")
check("it is at full strength by the horizon", abs(late - 1.0) < 1e-9, f"{late}")
check("and stays there past it, so an over-running story keeps the brake",
      abs(over - 1.0) < 1e-9, f"{over}")
check("a story inside its budget is barely touched",
      schedule_factor("ramp", 40, H) < 0.25, f"{schedule_factor('ramp', 40, H):.2f}")


def run(extra, vectors=("present_tense", "sensory", "closure")):
    cmd = [sys.executable, str(ROOT / "scripts" / "run_english_experiment.py"),
           "--model", "Qwen3-1.7B", "--task", "generic", "--constraint-set", "middle",
           "--suite", "compare", "--stories", "2", "--dry-run",
           "--steer-vectors", *vectors] + extra
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    return r.returncode, r.stdout + r.stderr


code, out = run(["--closure-ramp", "200"])
check("the flag is accepted when the direction is steered", code == 0)
check("and the run says what it will do",
      "full strength by token 200" in out, out.strip().splitlines()[-1][:80])

code, out = run(["--closure-ramp", "200"],
                vectors=("present_tense", "sensory", "named_character"))
check("it refuses when 'closure' is not steered, rather than doing nothing",
      code != 0 and "not\nin" not in out and "closure" in out,
      "an arm that quietly does nothing is its own control")

code, out = run([])
check("without the flag nothing changes", code == 0)

# --- and it actually reaches a built plan -----------------------------------
import warnings  # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import io, contextlib  # noqa: E402

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    from test_suites_build import ARGS, LAYERS, VECS  # noqa: E402
    import run_orthosteer_experiment as ro  # noqa: E402
    names = list(VECS.names)[:3]
    ro.RUN_DEFAULTS["schedules"] = {names[0]: "ramp"}
    ro.RUN_DEFAULTS["horizon"] = 200
    plan = ro.make_plan(VECS, LAYERS, names, beta=1.0, rms_scale=1.0)
    got = {sp.name: sp.schedule for sp in plan.specs}
    ro.RUN_DEFAULTS["schedules"] = None
    ro.RUN_DEFAULTS["horizon"] = None
check("the schedule reaches the plan's own specs",
      got.get(names[0]) == "ramp", str(got))
check("and the other directions are left flat",
      all(v == "constant" for k, v in got.items() if k != names[0]), str(got))

print()
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
    sys.exit(1)
print("ALL TESTS PASS")
