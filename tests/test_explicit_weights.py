#!/usr/bin/env python
"""Explicit budget shares, for the second round of allocation.

The first allocation is measured on the untouched model. The displacement then
breaks requirements the untouched model did not: naming a character falls from
66% to 42% under it, and the unnamed first-person register that results is what
runs long and loops -- a story that loops is five times more likely to open in
the first person than one that does not, and runs 292 words against 206.

Re-measuring on the method's own output and feeding the shares back is one step
of a fixed point. These checks cover the refusals, because a weight that is
silently ignored turns an arm into its own control, which this project has
shipped before.

Offline, no GPU, no model.

    python tests/test_explicit_weights.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAIL = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAIL.append(name)


def run(weights, vectors=("present_tense", "sensory", "named_character")):
    """Parse-only: --dry-run stops before any model is touched."""
    cmd = [sys.executable, str(ROOT / "scripts" / "run_english_experiment.py"),
           "--model", "Qwen3-1.7B", "--task", "generic", "--constraint-set", "middle",
           "--suite", "compare", "--stories", "2", "--dry-run",
           "--steer-vectors", *vectors]
    if weights is not None:
        cmd += ["--steer-weights", *weights]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    return r.returncode, (r.stdout + r.stderr)


code, out = run(["present_tense=1.0", "sensory=0.61", "named_character=0.65"])
check("a complete set of shares is accepted", code == 0, out.strip().splitlines()[-1][:90] if out.strip() else "")

code, out = run(["present_tense=1.0", "sensory=0.61"])
check("leaving a steered direction out is refused, not defaulted",
      code != 0 and "missing" in out,
      "defaulting the rest would mean weights nobody wrote down")

code, out = run(["present_tense=1.0", "sensory=0.61", "named_character=0.65",
                 "closure=0.5"])
check("a weight on a direction that is not steered is refused",
      code != 0 and "not in" in out,
      "a share that does nothing makes an arm its own control")

code, out = run(["present_tense", "sensory=0.61", "named_character=0.65"])
check("a share without a value is refused", code != 0 and "NAME=SHARE" in out)

code, out = run(None)
check("no explicit shares still works, leaving the measured path alone", code == 0)

print()
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
    sys.exit(1)
print("ALL TESTS PASS")
