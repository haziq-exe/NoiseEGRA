#!/usr/bin/env python3
"""Copy per-run RESULTS/*.txt from experiment subfolders into top-level RESULTS/.

After run_story_experiments(output_dir="experiment_results/ResidNoise"), per-run
reports live in experiment_results/ResidNoise/RESULTS/. Aggregation scripts
(final_scores.py, per_story_scores.py) expect experiment_results/RESULTS/.

Usage:
  python scripts/collect_run_results.py
  python scripts/collect_run_results.py --results-dir experiment_results
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring_common import resolve_results_dir


def collect_run_results(egra_results: Path, *, dry_run: bool = False) -> list[Path]:
    dest = egra_results / "RESULTS"
    dest.mkdir(parents=True, exist_ok=True)

    skip_names = {"RESULTS", "SCORES", "PER_STORY_SCORES", "PARTS_VENDI"}
    written: list[Path] = []

    for sub_results in sorted(egra_results.glob("*/RESULTS")):
        if sub_results.parent.name in skip_names:
            continue
        for src in sorted(sub_results.glob("*.txt")):
            out = dest / src.name
            if dry_run:
                print(f"would copy {src} -> {out}")
            else:
                shutil.copy2(src, out)
            written.append(out)

    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect per-run RESULTS into top-level RESULTS/")
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    egra_results = resolve_results_dir(args.results_dir)
    written = collect_run_results(egra_results, dry_run=args.dry_run)
    print(f"{'Would write' if args.dry_run else 'Wrote'} {len(written)} files under {egra_results / 'RESULTS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
