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
- Quality Score = mean(Readability, Logic, GrammarandLinguistics, ReadingLevel)
- Total Violations = score violations + EGRA violations
  where score violations =
    (1 - TotalModalCollapse) + (1 - Structure) + (1 - VocabularyLevel)
    + (1 - Stereotypes) + (1 - Gender-balanced)
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


BASE_DIR = Path(__file__).resolve().parents[1] / "experiment_results"
RESULTS_DIR = BASE_DIR / "RESULTS"
SCORES_DIR = BASE_DIR / "SCORES"
OUT_DIR = BASE_DIR / "PER_STORY_SCORES"

QUALITY_COLS = ["Readability", "Logic", "GrammarandLinguistics", "ReadingLevel"]
CONSTRAINT_SCORE_COLS = [
    "TotalModalCollapse",
    "Structure",
    "VocabularyLevel",
    "Stereotypes",
    "Gender-balanced",
]


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _as_float(cell: object) -> Optional[float]:
    if cell is None:
        return None
    s = str(cell).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group(0)) if m else None


def _as_int(cell: object) -> Optional[int]:
    v = _as_float(cell)
    if v is None:
        return None
    return int(v)


def parse_results_violations(results_path: Path) -> List[int]:
    """Parse '- Story #i: violations=k' lines from RESULTS/*.txt (0-based story index)."""

    story_re = re.compile(r"^-\s*Story\s*#(\d+):\s*violations=(\d+)")
    by_idx: Dict[int, int] = {}

    for raw in results_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = story_re.match(raw.strip())
        if not m:
            continue
        by_idx[int(m.group(1))] = int(m.group(2))

    if not by_idx:
        return []

    max_idx = max(by_idx)
    return [by_idx.get(i, 0) for i in range(max_idx + 1)]


def _norm_stem(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def iter_runs() -> Iterable[Tuple[Path, Path]]:
    """Yield matched pairs (results_txt, score_csv) by stem."""

    score_by_norm: Dict[str, List[Path]] = {}
    for score_csv in sorted(SCORES_DIR.glob("*_SCORE.csv")):
        score_stem = score_csv.stem
        if score_stem.endswith("_SCORE"):
            score_stem = score_stem[: -len("_SCORE")]
        score_by_norm.setdefault(_norm_stem(score_stem), []).append(score_csv)

    for results_txt in sorted(RESULTS_DIR.glob("*.txt")):
        stem = results_txt.stem

        exact = SCORES_DIR / f"{stem}_SCORE.csv"
        if exact.exists():
            yield results_txt, exact
            continue

        candidates = score_by_norm.get(_norm_stem(stem), [])
        if len(candidates) == 1:
            yield results_txt, candidates[0]


def load_per_story_rows(results_txt: Path, score_csv: Path) -> List[Tuple[int, float, int]]:
    """Return rows as (story_number_1_based, quality_score, total_violations)."""

    egra_violations = parse_results_violations(results_txt)

    quality_per_story: List[float] = []
    score_constraint_per_story: List[int] = []

    with score_csv.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Empty score file: {score_csv}")

        required = QUALITY_COLS + CONSTRAINT_SCORE_COLS + ["Story number"]
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Score file missing columns {missing}: {score_csv}")

        for row in reader:
            q_vals: List[float] = []
            for c in QUALITY_COLS:
                v = _as_float(row.get(c))
                if v is not None:
                    q_vals.append(v)
            quality_per_story.append(mean(q_vals) if q_vals else 0.0)

            score_v = 0
            for c in CONSTRAINT_SCORE_COLS:
                iv = _as_int(row.get(c))
                if iv is None:
                    continue
                score_v += 1 - iv
            score_constraint_per_story.append(score_v)

    n = min(len(egra_violations), len(quality_per_story), len(score_constraint_per_story))

    rows: List[Tuple[int, float, int]] = []
    for idx in range(n):
        total_violations = int(egra_violations[idx] + score_constraint_per_story[idx])
        rows.append((idx + 1, quality_per_story[idx], total_violations))

    return rows


def write_run_csv(stem: str, rows: List[Tuple[int, float, int]]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{stem}.csv"

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Story Number", "Quality Score", "Total Violations"])
        for story_number, quality_score, total_violations in rows:
            writer.writerow([story_number, f"{quality_score:.4f}", total_violations])

    return out_path


def main() -> int:
    runs = list(iter_runs())
    if not runs:
        raise SystemExit("No matching RESULTS/*.txt + SCORES/*_SCORE.csv pairs found")

    written = 0
    for results_txt, score_csv in runs:
        rows = load_per_story_rows(results_txt, score_csv)
        write_run_csv(results_txt.stem, rows)
        written += 1

    print(f"Wrote {written} per-story CSV files to: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
