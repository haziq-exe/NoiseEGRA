#!/usr/bin/env python
"""End-to-end check of the Kaggle harness: no GPU, no model, about a minute.

Exercises the three things that are easy to get wrong and expensive to discover
during a real run:

  live output   prints a line a second, so the log stream can be watched
  checkpoint    appends to a counter file under --out, which only grows across
                runs if the state round-trip through the Kaggle dataset works
  exit status   --fail makes it exit non-zero, so failures are visibly failures

    python scripts/kaggle_harness.py run --name harness-check --no-gpu --no-spacy \\
        -- scripts/harness_selftest.py --seconds 45
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--fail", action="store_true", help="exit non-zero at the end")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ledger = out / "selftest.json"

    runs = json.loads(ledger.read_text(encoding="utf-8")) if ledger.is_file() else []
    print(f"checkpoint: {len(runs)} previous run(s) recorded in {ledger.name}",
          flush=True)
    for r in runs[-3:]:
        print(f"  earlier run at {r['started']} lasted {r['seconds']}s", flush=True)

    print(f"python {platform.python_version()} on {platform.platform()}", flush=True)
    try:
        import torch

        print(f"torch {torch.__version__}, cuda={torch.cuda.is_available()}, "
              f"devices={torch.cuda.device_count() if torch.cuda.is_available() else 0}",
              flush=True)
    except ImportError:
        print("torch not importable", flush=True)

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.time()
    for i in range(args.seconds):
        print(f"  [{i + 1:>3}/{args.seconds}] still going, {time.time() - t0:5.1f}s "
              "elapsed", flush=True)
        time.sleep(1)

    runs.append({"started": started, "seconds": args.seconds})
    ledger.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(f"\nwrote {ledger} -- now {len(runs)} run(s) recorded", flush=True)
    print("if that number grows on the next run, the checkpoint round-trip works",
          flush=True)

    if args.fail:
        print("exiting non-zero on purpose", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
