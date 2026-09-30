#!/usr/bin/env python
"""Run several Python scripts one after another in one Kaggle session.

    python scripts/run_sequence.py scripts/a.py --x 1 ::: scripts/b.py --y 2

The harness passes its command through shlex, so a shell ``;`` between two
commands reaches the first script as a literal argument. This splits on
``:::`` instead and runs each part with this interpreter, carrying on after a
failure so one model that crashes does not cost the others; the exit code is
non-zero if any part failed.
"""
import subprocess
import sys


def main() -> None:
    parts, cur = [], []
    for a in sys.argv[1:]:
        if a == ":::":
            if cur:
                parts.append(cur)
            cur = []
        else:
            cur.append(a)
    if cur:
        parts.append(cur)
    failed = []
    for i, p in enumerate(parts, 1):
        print(f"\n=== [{i}/{len(parts)}] {' '.join(p)[:160]}", flush=True)
        rc = subprocess.run([sys.executable, "-u", *p]).returncode
        if rc:
            failed.append((i, rc))
            print(f"=== [{i}/{len(parts)}] exited {rc}", flush=True)
    print(f"\n=== {len(parts) - len(failed)} of {len(parts)} finished"
          + (f"; failed: {failed}" if failed else ""), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
