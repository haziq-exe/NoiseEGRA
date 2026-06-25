#!/usr/bin/env python3
"""Compute EGRA experiment metrics with a strict first-N story cap.

Despite the script name, the default cap is 50 stories (override with --first-n).
Uses the same layout as final_scores.py: RESULTS/{run_id}.txt + SCORES/{run_id}_SCORE.csv.

Outputs (under the results directory):
  - metrics_first30_summary.csv
  - metrics_first30_details.json
  - metrics_first30_audit.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring_common import (
    align_story_data,
    iter_run_pairs,
    mean,
    resolve_results_dir,
    sample_std,
)

DEFAULT_FIRST_N = 50

CREATIVITY_RE = re.compile(r"^Combined Creativity Score:\s*([0-9.]+)\s*\(std\s*([0-9.]+)\)")


def parse_creativity_from_results(results_path: Path) -> Dict[str, float]:
    for line in results_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = CREATIVITY_RE.match(line.strip())
        if m:
            return {"mean": float(m.group(1)), "std": float(m.group(2))}
    return {"mean": 0.0, "std": 0.0}


def find_raw_csv(egra_results: Path, run_stem: str) -> Path | None:
    candidates = [
        p
        for p in egra_results.rglob("*.csv")
        if p.stem == run_stem
        and "SCORES" not in p.parts
        and "PARTS_VENDI" not in p.parts
        and "PER_STORY_SCORES" not in p.parts
    ]
    if len(candidates) == 1:
        return candidates[0]
    return candidates[0] if candidates else None


def read_first_column_stories(csv_path: Path, first_n: int) -> List[str]:
    stories: List[str] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            text = str(row[0]).strip()
            if text:
                stories.append(text)
            if len(stories) >= first_n:
                break
    return stories


def recompute_creativity_first_n(
    raw_csv: Path,
    first_n: int,
    scorer_obj: object,
) -> Dict[str, float]:
    stories = read_first_column_stories(raw_csv, first_n)
    if len(stories) < 2:
        return {"mean": 0.0, "std": 0.0, "story_count": float(len(stories))}

    scorer_obj.change_text(stories)
    semantic = scorer_obj.semantic_diversity()
    lexical = scorer_obj.lexical_diversity()

    combined_mean = 0.5 * semantic.semantic_score_mean + 0.5 * lexical.lexical_score_mean
    combined_std = math.sqrt(
        (0.5 * semantic.semantic_score_std) ** 2 + (0.5 * lexical.lexical_score_std) ** 2
    )

    return {"mean": float(combined_mean), "std": float(combined_std), "story_count": float(len(stories))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute EGRA metrics with a first-N cap.")
    parser.add_argument("--results-dir", type=Path, default=None, help="Experiment results root")
    parser.add_argument(
        "--first-n",
        type=int,
        default=DEFAULT_FIRST_N,
        help=f"Number of stories to include from start (default {DEFAULT_FIRST_N})",
    )
    parser.add_argument(
        "--recompute-creativity",
        action="store_true",
        help="Recompute creativity on first-N stories from raw csv files.",
    )
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-m3",
        help="Embedding model when --recompute-creativity is enabled (default matches CreativityScorer).",
    )
    args = parser.parse_args()

    base_dir = resolve_results_dir(args.results_dir)
    results_dir = base_dir / "RESULTS"
    scores_dir = base_dir / "SCORES"

    runs = list(iter_run_pairs(results_dir, scores_dir))
    if not runs:
        raise FileNotFoundError(
            f"No matching RESULTS/*.txt + SCORES/*_SCORE.csv pairs under {base_dir}"
        )

    scorer_obj = None
    if args.recompute_creativity:
        from noiseegra.creativity_metrics import CreativityScorer

        scorer_obj = CreativityScorer(texts=["x", "y"], embedding_model=args.embedding_model)

    rows: List[Dict[str, object]] = []

    for results_txt, score_csv in runs:
        run_id = results_txt.stem
        aligned = align_story_data(results_txt, score_csv, run_id=run_id)
        aligned = aligned[: args.first_n]

        qualities = [s.quality for s in aligned]
        score_constraints = [s.score_constraint_violations for s in aligned]
        egra_violations = [float(s.egra_violations) for s in aligned]
        combined = [s.total_violations for s in aligned]

        creativity = parse_creativity_from_results(results_txt)
        creativity_source = "results_txt"
        creativity_story_count = len(aligned)
        creativity_first_n_exact = creativity_story_count <= args.first_n

        if args.recompute_creativity:
            raw_csv = find_raw_csv(base_dir, run_id)
            if raw_csv is None:
                raise FileNotFoundError(f"No raw story CSV found for run_id={run_id}")
            assert scorer_obj is not None
            recomputed = recompute_creativity_first_n(raw_csv, args.first_n, scorer_obj)
            creativity = {"mean": recomputed["mean"], "std": recomputed["std"]}
            creativity_story_count = int(recomputed["story_count"])
            creativity_first_n_exact = True
            creativity_source = "recomputed_from_raw_first_n"

        rows.append(
            {
                "run_id": run_id,
                "score_file": str(score_csv),
                "results_file": str(results_txt),
                "stories_used": len(aligned),
                "quality_story_scores_used": qualities,
                "score_constraint_counts_used": score_constraints,
                "egra_constraint_counts_used": egra_violations,
                "combined_constraints_used": combined,
                "quality_mean": mean(qualities),
                "quality_std": sample_std(qualities),
                "constraints_broken_mean": mean(combined),
                "constraints_broken_std": sample_std(combined),
                "creativity_mean_from_results": creativity["mean"],
                "creativity_std_from_results": creativity["std"],
                "creativity_results_story_count": creativity_story_count,
                "creativity_is_exact_for_first_n": creativity_first_n_exact,
                "creativity_source": creativity_source,
            }
        )

    rows.sort(key=lambda r: str(r["run_id"]))

    summary_csv_path = base_dir / "metrics_first30_summary.csv"
    details_json_path = base_dir / "metrics_first30_details.json"
    audit_txt_path = base_dir / "metrics_first30_audit.txt"

    summary_columns = [
        "run_id",
        "score_file",
        "results_file",
        "stories_used",
        "quality_mean",
        "quality_std",
        "constraints_broken_mean",
        "constraints_broken_std",
        "creativity_mean_from_results",
        "creativity_std_from_results",
        "creativity_results_story_count",
        "creativity_is_exact_for_first_n",
        "creativity_source",
    ]

    with summary_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=summary_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in summary_columns})

    with details_json_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)

    with audit_txt_path.open("w", encoding="utf-8") as fh:
        fh.write("EGRA Metrics Audit (first-N story cap)\n")
        fh.write("=======================================\n\n")
        fh.write(f"FIRST_N = {args.first_n}\n\n")
        fh.write("Metric definitions:\n")
        fh.write("1) quality_story_score_i = mean(Readability, Logic, GrammarandLinguistics)\n")
        fh.write("2) score_constraint_count_i = sum(1 - value) across the 5 score constraints\n")
        fh.write("3) combined_constraints_i = score_constraint_count_i + egra_violations_i\n")
        fh.write("4) Stories aligned by Story number (not row order)\n")
        fh.write("5) creativity mean/std read from per-run RESULTS/*.txt by default\n\n")

        for row in rows:
            fh.write(f"Run ID: {row['run_id']}\n")
            fh.write(f"Score file: {row['score_file']}\n")
            fh.write(f"Results file: {row['results_file']}\n")
            fh.write(f"Stories used: {row['stories_used']}\n")
            fh.write(f"quality_story_scores_used: {row['quality_story_scores_used']}\n")
            fh.write(f"combined_constraints_used: {row['combined_constraints_used']}\n")
            fh.write(f"Quality mean/std: {row['quality_mean']:.6f} / {row['quality_std']:.6f}\n")
            fh.write(
                "Constraints broken mean/std: "
                f"{row['constraints_broken_mean']:.6f} / {row['constraints_broken_std']:.6f}\n"
            )
            fh.write(
                "Creativity mean/std: "
                f"{row['creativity_mean_from_results']:.6f} / {row['creativity_std_from_results']:.6f} "
                f"(source={row['creativity_source']})\n"
            )
            fh.write("\n" + "-" * 72 + "\n\n")

    print(f"Wrote: {summary_csv_path}")
    print(f"Wrote: {details_json_path}")
    print(f"Wrote: {audit_txt_path}")


if __name__ == "__main__":
    main()
