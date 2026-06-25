#!/usr/bin/env python3
"""Create per-story score CSV files for every matched run.

For each run with both files present:
- experiment_results/RESULTS/<stem>.txt
- experiment_results/SCORES/<stem>_SCORE.csv

Write:
- experiment_results/PER_STORY_SCORES/<stem>.csv

Columns:
- Story Number
- Quality Score
- Total Violations

Definitions:
- Quality Score = mean(Readability, Logic, GrammarandLinguistics)
- Total Violations = score violations + EGRA violations
  where score violations =
    (1 - TotalModalCollapse) + (1 - Structure) + (1 - VocabularyLevel)
    + (1 - Stereotypes) + (1 - Gender-balanced)
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring_common import align_story_data, iter_run_pairs, resolve_results_dir


def load_per_story_rows(results_txt: Path, score_csv: Path) -> List[Tuple[int, float, int]]:
    """Return rows as (story_number_1_based, quality_score, total_violations)."""
    aligned = align_story_data(results_txt, score_csv, run_id=results_txt.stem)
    return [
        (s.story_number, s.quality, int(round(s.total_violations)))
        for s in aligned
    ]


def write_run_csv(out_dir: Path, stem: str, rows: List[Tuple[int, float, int]]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.csv"

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Story Number", "Quality Score", "Total Violations"])
        for story_number, quality_score, total_violations in rows:
            writer.writerow([story_number, f"{quality_score:.4f}", total_violations])

    return out_path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=None, help="Experiment results root")
    args = ap.parse_args()

    base_dir = resolve_results_dir(args.results_dir)
    results_dir = base_dir / "RESULTS"
    scores_dir = base_dir / "SCORES"
    out_dir = base_dir / "PER_STORY_SCORES"

    runs = list(iter_run_pairs(results_dir, scores_dir))
    if not runs:
        raise SystemExit("No matching RESULTS/*.txt + SCORES/*_SCORE.csv pairs found")

    written = 0
    for results_txt, score_csv in runs:
        rows = load_per_story_rows(results_txt, score_csv)
        write_run_csv(out_dir, results_txt.stem, rows)
        written += 1

    print(f"Wrote {written} per-story CSV files to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
