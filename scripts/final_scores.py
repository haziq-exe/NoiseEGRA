#!/usr/bin/env python3
"""Create a consolidated Final_Scores.txt from RESULTS + SCORES.

For each run (one raw stories CSV), we expect:
- An EGRA constraint/creativity results text in experiment_results/RESULTS/<stem>.txt
- A per-story score file in experiment_results/SCORES/<stem>_SCORE.csv

We compute:
- Quality per story = mean(Readability, Logic, GrammarandLinguistics)
  and then report mean/std across stories.
- Constraints broken per story = EGRA violations (from results txt)
  + score violations (derived from score columns: TotalModalCollapse, Structure,
  VocabularyLevel, Stereotypes, Gender-balanced as 1 - value).
  and then report mean/std across stories.
- Reading level summary from score file.

Output: experiment_results/Final_Scores.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring_common import (
    align_story_data,
    iter_run_pairs,
    mean,
    resolve_results_dir,
    sample_std,
    summarize_reading_levels,
)


@dataclass(frozen=True)
class RunSummary:
    stem: str
    stories: int
    quality_mean: float
    quality_std: float
    combined_constraints_mean: float
    combined_constraints_std: float
    reading_level_mode: str
    reading_level_counts: Dict[str, int]


def compute_run_summary(
    results_txt: Path,
    score_csv: Path,
    first_n: Optional[int],
) -> RunSummary:
    aligned = align_story_data(results_txt, score_csv, run_id=results_txt.stem)
    if first_n is not None:
        aligned = aligned[:first_n]

    qualities = [s.quality for s in aligned]
    combined = [s.total_violations for s in aligned]
    reading_levels = [s.reading_level for s in aligned]
    mode, counts = summarize_reading_levels(reading_levels)

    return RunSummary(
        stem=results_txt.stem,
        stories=len(aligned),
        quality_mean=mean(qualities),
        quality_std=sample_std(qualities),
        combined_constraints_mean=mean(combined),
        combined_constraints_std=sample_std(combined),
        reading_level_mode=mode,
        reading_level_counts=counts,
    )


def write_final_scores(out_path: Path, summaries: List[RunSummary], first_n: Optional[int]) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write("Final Scores\n")
        f.write("===========\n\n")
        if first_n is not None:
            f.write(f"First-N cap: {first_n}\n\n")

        for s in summaries:
            f.write(f"---- {s.stem} ----\n")
            f.write(f"Stories used: {s.stories}\n")
            f.write(f"Quality mean/std: {s.quality_mean:.4f} / {s.quality_std:.4f}\n")
            f.write(
                "Violations/story (score+egra) mean/std: "
                f"{s.combined_constraints_mean:.4f} / {s.combined_constraints_std:.4f}\n"
            )
            if s.reading_level_mode:
                f.write(f"ReadingLevel mode: {s.reading_level_mode}\n")
                top = sorted(s.reading_level_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
                f.write("ReadingLevel counts (top): " + ", ".join(f"{k}={v}" for k, v in top) + "\n")
            else:
                f.write("ReadingLevel: (missing)\n")
            f.write("\n")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=None, help="Experiment results root")
    ap.add_argument("--first-n", type=int, default=None, help="Optional cap on number of stories to use")
    args = ap.parse_args()

    base_dir = resolve_results_dir(args.results_dir)
    results_dir = base_dir / "RESULTS"
    scores_dir = base_dir / "SCORES"

    runs = list(iter_run_pairs(results_dir, scores_dir))
    if not runs:
        raise SystemExit("No matching RESULTS/*.txt + SCORES/*_SCORE.csv pairs found")

    summaries = [compute_run_summary(rtxt, scsv, args.first_n) for rtxt, scsv in runs]
    summaries.sort(key=lambda s: s.stem)

    out = base_dir / "Final_Scores.txt"
    write_final_scores(out, summaries, args.first_n)
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
