#!/usr/bin/env python
"""Run the scale probe on several models in one session, one after another.

    python scripts/probe_scale.py <out-dir> MODEL:LO:HI [MODEL:LO:HI ...] -- <runner args...>

For each model, runs the English runner with ``--probe-scale`` (see
``noiseegra.scale_probe``) on the steering band LO..HI and writes
``<out-dir>/<model>/scale_probe.json``; the runner args are shared (the prompt,
the rules, the steering and the headline arm sized while writing). A model that
fails does not stop the rest.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_english_experiment.py"


def commands(out: str, specs, rest):
    for spec in specs:
        model, lo, hi = spec.split(":")
        yield model, [sys.executable, "-u", str(RUNNER), "--model", model,
                      "--layers", lo, hi, *rest, "--out", f"{out}/{model}"]


def main() -> None:
    argv = sys.argv[1:]
    if "--" not in argv or len(argv) < 3:
        raise SystemExit(__doc__)
    cut = argv.index("--")
    out, specs, rest = argv[0], argv[1:cut], argv[cut + 1:]
    codes = []
    for model, cmd in commands(out, specs, rest):
        print(f"=== probing {model} ===", flush=True)
        codes.append(subprocess.run(cmd).returncode)
    print(f"=== probes finished, exit codes {codes} ===", flush=True)
    sys.exit(0 if any(c == 0 for c in codes) else 1)


if __name__ == "__main__":
    main()
