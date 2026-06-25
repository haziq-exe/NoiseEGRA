#!/usr/bin/env python3
"""
Compute EGRA experiment metrics with a strict first-N story cap.

Outputs:
  - experiment_results/metrics_first30_summary.csv
  - experiment_results/metrics_first30_details.json
  - experiment_results/metrics_first30_audit.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple


BASE_DIR = Path(__file__).resolve().parents[1] / "experiment_results"
FIRST_N = 50


QUALITY_ALIASES = {
    "readability": {"readability", "readability10"},
    "logic": {"logic", "logic10"},
    "grammarandlinguistic": {
        "grammarandlinguistic",
        "grammarlinguistic",
        "grammarandlinguistic10",
        "grammarlinguistic10",
    },
}

CONSTRAINT_ALIASES = {
    "totalmodalcollapse": {"totalmodalcollapse"},
    "structure": {"structure"},
    "vocabularylevel": {"vocabularylevel"},
    "stereotypes": {"stereotypes"},
    "genderbalanced": {"genderbalanced"},
}


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def parse_numbers(cell: object) -> List[float]:
    if cell is None:
        return []
    text = str(cell).strip()
    if not text:
        return []
    return [float(x) for x in re.findall(r"\d+\.?\d*", text)]


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def build_column_map(headers: List[str]) -> Dict[str, str]:
    normalized_headers = {h: norm(h) for h in headers}
    out: Dict[str, str] = {}

    for canonical, aliases in {**QUALITY_ALIASES, **CONSTRAINT_ALIASES}.items():
        matched = None
        for header, header_norm in normalized_headers.items():
            if header_norm in aliases:
                matched = header
                break
        if matched is None:
            raise ValueError(f"Missing expected column '{canonical}' in headers: {headers}")
        out[canonical] = matched

    return out


def parse_results_txt(path: Path) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[int]]]:
    creativity: Dict[str, Dict[str, float]] = {}
    egra_story_violations: Dict[str, Dict[int, int]] = {}

    current_run = None
    current_constraint_run = None

    run_header_re = re.compile(r"^----\s*(.+?)\s*----\s*$")
    constraint_header_re = re.compile(r"^-+\s*(.+?)\s+CONSTRAINT\s*-+$")
    creativity_re = re.compile(r"^Combined Creativity Score:\s*([0-9.]+)\s*\(std\s*([0-9.]+)\)")
    story_re = re.compile(r"^- Story #(\d+): violations=(\d+)")

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        m = constraint_header_re.match(line)
        if m:
            current_constraint_run = m.group(1)
            egra_story_violations.setdefault(current_constraint_run, {})
            continue

        m = run_header_re.match(line)
        if m and "CONSTRAINT" not in line:
            current_run = m.group(1)
            continue

        m = creativity_re.match(line)
        if m and current_run:
            creativity[current_run] = {"mean": float(m.group(1)), "std": float(m.group(2))}
            continue

        m = story_re.match(line)
        if m and current_constraint_run:
            idx = int(m.group(1))
            violations = int(m.group(2))
            egra_story_violations[current_constraint_run][idx] = violations

    egra_lists: Dict[str, List[int]] = {}
    for run_id, by_index in egra_story_violations.items():
        if not by_index:
            egra_lists[run_id] = []
            continue
        max_idx = max(by_index)
        egra_lists[run_id] = [by_index[i] for i in range(max_idx + 1)]

    return creativity, egra_lists


def map_score_to_run_id(score_file: Path, run_ids: List[str]) -> str:
    stem_normalized = norm(score_file.stem)

    if "zeroshot" in stem_normalized:
        baseline_candidates = [rid for rid in run_ids if rid.endswith("__BASELINE")]
        if len(baseline_candidates) != 1:
            raise ValueError(
                f"Expected exactly one BASELINE run in RESULTS for {score_file}, got {baseline_candidates}"
            )
        return baseline_candidates[0]

    layer_match = re.search(r"L(\d+-\d+)", score_file.name)
    if not layer_match:
        raise ValueError(f"Could not find layer range in score filename: {score_file.name}")
    layer = layer_match.group(1)

    layer_candidates = [rid for rid in run_ids if f"__L{layer}__" in rid]
    if len(layer_candidates) != 1:
        raise ValueError(
            f"Expected exactly one run containing '__L{layer}__' for {score_file}, got {layer_candidates}"
        )
    return layer_candidates[0]


def map_run_to_raw_csv(run_id: str, raw_csv_files: List[Path]) -> Path:
    if run_id.endswith("__BASELINE"):
        baseline_candidates = [p for p in raw_csv_files if p.stem.endswith("__BASELINE")]
        if not baseline_candidates:
            baseline_candidates = [p for p in raw_csv_files if "zeroshot" in norm(p.stem)]
        if not baseline_candidates:
            baseline_candidates = [p for p in raw_csv_files if "zeroshot" in norm(p.name)]
        if len(baseline_candidates) != 1:
            raise ValueError(
                f"Expected one raw baseline csv for run_id={run_id}, got {baseline_candidates}"
            )
        return baseline_candidates[0]

    layer_match = re.search(r"__L(\d+-\d+)__", run_id)
    if not layer_match:
        raise ValueError(f"Could not parse layer from run_id={run_id}")
    layer = layer_match.group(1)

    layer_candidates = [p for p in raw_csv_files if f"__L{layer}__" in p.stem]
    if len(layer_candidates) != 1:
        raise ValueError(
            f"Expected one raw csv containing __L{layer}__ for run_id={run_id}, got {layer_candidates}"
        )
    return layer_candidates[0]


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
    run_id: str,
    raw_csv_files: List[Path],
    first_n: int,
    scorer_obj: object,
) -> Dict[str, float]:
    raw_csv = map_run_to_raw_csv(run_id, raw_csv_files)
    stories = read_first_column_stories(raw_csv, first_n)
    if len(stories) < 2:
        return {"mean": 0.0, "std": 0.0, "story_count": float(len(stories))}

    scorer_obj.change_text(stories)
    semantic = scorer_obj.semantic_diversity()
    lexical = scorer_obj.lexical_diversity()

    combined_mean = 0.5 * semantic.semantic_score_mean + 0.5 * lexical.lexical_score_mean
    combined_std = math.sqrt((0.5 * semantic.semantic_score_std) ** 2 + (0.5 * lexical.lexical_score_std) ** 2)

    return {"mean": float(combined_mean), "std": float(combined_std), "story_count": float(len(stories))}


def compute_from_score_file(score_file: Path, first_n: int) -> Dict[str, object]:
    with score_file.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        headers = reader.fieldnames or []
        col_map = build_column_map(headers)

        quality_story_scores: List[float] = []
        score_constraint_counts: List[float] = []

        for row in reader:
            quality_values: List[float] = []
            for c in QUALITY_ALIASES.keys():
                quality_values.extend(parse_numbers(row.get(col_map[c], "")))
            if quality_values:
                quality_story_scores.append(mean(quality_values))

            score_constraint_violations: List[float] = []
            for c in CONSTRAINT_ALIASES.keys():
                parsed = parse_numbers(row.get(col_map[c], ""))
                score_constraint_violations.extend([1 - x for x in parsed])
            if score_constraint_violations:
                score_constraint_counts.append(sum(score_constraint_violations))

    return {
        "column_map": col_map,
        "quality_story_scores": quality_story_scores,
        "score_constraint_counts": score_constraint_counts,
        "first_n": first_n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute EGRA metrics with a first-N cap.")
    parser.add_argument("--first-n", type=int, default=FIRST_N, help="Number of stories to include from start.")
    parser.add_argument(
        "--recompute-creativity",
        action="store_true",
        help="Recompute creativity on first-N stories from raw csv files (requires creativity dependencies).",
    )
    parser.add_argument(
        "--embedding-model",
        default="Omartificial-Intelligence-Space/GATE-AraBert-v1",
        help="Embedding model used when --recompute-creativity is enabled.",
    )
    args = parser.parse_args()

    results_txt_files = sorted(BASE_DIR.glob("*/*_RESULTS.txt"))
    if not results_txt_files:
        raise FileNotFoundError("No *_RESULTS.txt files found under experiment_results/*/")

    scorer_obj = None
    if args.recompute_creativity:
        from noiseegra.creativity_metrics import CreativityScorer

        # Initialized once and reused across runs.
        scorer_obj = CreativityScorer(texts=["x", "y"], embedding_model=args.embedding_model)

    rows: List[Dict[str, object]] = []

    for results_txt in results_txt_files:
        model_folder = results_txt.parent.name
        creativity_by_run, egra_by_run = parse_results_txt(results_txt)
        run_ids = sorted(creativity_by_run.keys())

        score_files = sorted(results_txt.parent.glob("*_Scores.csv"))
        raw_csv_files = sorted(
            p
            for p in results_txt.parent.glob("*.csv")
            if not p.name.endswith("_Scores.csv")
        )
        if not score_files:
            continue

        for score_file in score_files:
            run_id = map_score_to_run_id(score_file, run_ids)
            score_data = compute_from_score_file(score_file, args.first_n)

            quality_story_scores = score_data["quality_story_scores"]  # type: ignore[index]
            score_constraint_counts = score_data["score_constraint_counts"]  # type: ignore[index]
            egra_violations = egra_by_run.get(run_id)
            if egra_violations is None:
                raise ValueError(f"Missing EGRA constraint section for run_id={run_id}")

            n_used = min(args.first_n, len(quality_story_scores), len(score_constraint_counts), len(egra_violations))
            q_first = quality_story_scores[:n_used]
            score_c_first = score_constraint_counts[:n_used]
            egra_first = egra_violations[:n_used]
            combined_constraints = [a + b for a, b in zip(score_c_first, egra_first)]

            creativity = creativity_by_run[run_id]
            creativity_story_count = len(egra_violations)
            creativity_first_n_exact = creativity_story_count <= args.first_n
            creativity_source = "results_txt"

            if args.recompute_creativity:
                assert scorer_obj is not None
                recomputed = recompute_creativity_first_n(
                    run_id=run_id,
                    raw_csv_files=raw_csv_files,
                    first_n=args.first_n,
                    scorer_obj=scorer_obj,
                )
                creativity = {"mean": recomputed["mean"], "std": recomputed["std"]}
                creativity_story_count = int(recomputed["story_count"])
                creativity_first_n_exact = True
                creativity_source = "recomputed_from_raw_first_n"

            rows.append(
                {
                    "model_folder": model_folder,
                    "run_id": run_id,
                    "score_file": str(score_file),
                    "results_file": str(results_txt),
                    "score_column_map": score_data["column_map"],
                    "stories_in_scores_quality": len(quality_story_scores),
                    "stories_in_scores_constraints": len(score_constraint_counts),
                    "stories_in_results_constraints": len(egra_violations),
                    "stories_used": n_used,
                    "quality_story_scores_used": q_first,
                    "score_constraint_counts_used": score_c_first,
                    "egra_constraint_counts_used": egra_first,
                    "combined_constraints_used": combined_constraints,
                    "quality_mean": mean(q_first),
                    "quality_std": sample_std(q_first),
                    "constraints_broken_mean": mean(combined_constraints),
                    "constraints_broken_std": sample_std(combined_constraints),
                    "creativity_mean_from_results": creativity["mean"],
                    "creativity_std_from_results": creativity["std"],
                    "creativity_results_story_count": creativity_story_count,
                    "creativity_is_exact_for_first30": creativity_first_n_exact,
                    "creativity_source": creativity_source,
                }
            )

    rows.sort(key=lambda r: (str(r["model_folder"]), str(r["run_id"])))

    summary_csv_path = BASE_DIR / "metrics_first30_summary.csv"
    details_json_path = BASE_DIR / "metrics_first30_details.json"
    audit_txt_path = BASE_DIR / "metrics_first30_audit.txt"

    # summary csv
    summary_columns = [
        "model_folder",
        "run_id",
        "score_file",
        "results_file",
        "stories_in_scores_quality",
        "stories_in_scores_constraints",
        "stories_in_results_constraints",
        "stories_used",
        "quality_mean",
        "quality_std",
        "constraints_broken_mean",
        "constraints_broken_std",
        "creativity_mean_from_results",
        "creativity_std_from_results",
        "creativity_results_story_count",
        "creativity_is_exact_for_first30",
        "creativity_source",
    ]

    with summary_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=summary_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in summary_columns})

    # details json
    with details_json_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)

    # audit txt
    with audit_txt_path.open("w", encoding="utf-8") as fh:
        fh.write("EGRA Metrics Audit (first-30 story cap)\n")
        fh.write("=======================================\n\n")
        fh.write(f"FIRST_N = {args.first_n}\n\n")
        fh.write("Metric definitions:\n")
        fh.write("1) quality_story_score_i = mean(Readability_i, Logic_i, Grammar_i)\n")
        fh.write("2) score_constraint_count_i = sum(1 - value) across the 5 score constraints\n")
        fh.write("3) combined_constraints_i = score_constraint_count_i + egra_violations_i\n")
        fh.write("4) quality_mean/std and constraints_mean/std are computed over first N stories\n")
        fh.write("5) creativity mean/std are read from *_RESULTS.txt by default\n")
        fh.write("   (or recomputed from raw csv first-N when --recompute-creativity is enabled)\n\n")

        for row in rows:
            fh.write(f"Model folder: {row['model_folder']}\n")
            fh.write(f"Run ID: {row['run_id']}\n")
            fh.write(f"Score file: {row['score_file']}\n")
            fh.write(f"Results file: {row['results_file']}\n")
            fh.write(f"Column map: {json.dumps(row['score_column_map'], ensure_ascii=False)}\n")
            fh.write(
                "Story counts: "
                f"scores_quality={row['stories_in_scores_quality']}, "
                f"scores_constraints={row['stories_in_scores_constraints']}, "
                f"results_constraints={row['stories_in_results_constraints']}, "
                f"used={row['stories_used']}\n"
            )
            fh.write(f"quality_story_scores_used: {row['quality_story_scores_used']}\n")
            fh.write(f"score_constraint_counts_used: {row['score_constraint_counts_used']}\n")
            fh.write(f"egra_constraint_counts_used: {row['egra_constraint_counts_used']}\n")
            fh.write(f"combined_constraints_used: {row['combined_constraints_used']}\n")
            fh.write(
                f"Quality mean/std: {row['quality_mean']:.6f} / {row['quality_std']:.6f}\n"
            )
            fh.write(
                "Constraints broken mean/std: "
                f"{row['constraints_broken_mean']:.6f} / {row['constraints_broken_std']:.6f}\n"
            )
            fh.write(
                "Creativity mean/std from results: "
                f"{row['creativity_mean_from_results']:.6f} / {row['creativity_std_from_results']:.6f}\n"
            )
            fh.write(
                "Creativity exact for first30: "
                f"{row['creativity_is_exact_for_first30']} "
                f"(results_story_count={row['creativity_results_story_count']})\n"
            )
            fh.write(f"Creativity source: {row['creativity_source']}\n")
            fh.write("\n" + "-" * 72 + "\n\n")

    print(f"Wrote: {summary_csv_path}")
    print(f"Wrote: {details_json_path}")
    print(f"Wrote: {audit_txt_path}")


if __name__ == "__main__":
    main()
