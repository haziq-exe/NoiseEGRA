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
import csv
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


BASE_DIR = Path(__file__).resolve().parents[1] / "experiment_results"
RESULTS_DIR = BASE_DIR / "RESULTS"
SCORES_DIR = BASE_DIR / "SCORES"

QUALITY_COLS = ["Readability", "Logic", "GrammarandLinguistics"]
CONSTRAINT_SCORE_COLS = [
    "TotalModalCollapse",
    "Structure",
    "VocabularyLevel",
    "Stereotypes",
    "Gender-balanced",
]


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def sample_std(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


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
    """Parse '- Story #i: violations=k' lines from RESULTS/*.txt."""

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


def load_score_file(score_path: Path) -> Tuple[List[float], List[float], List[str]]:
    """Return (quality_per_story, score_constraint_per_story, reading_levels)."""

    quality_per_story: List[float] = []
    score_constraint_per_story: List[float] = []
    reading_levels: List[str] = []

    with score_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Empty score file: {score_path}")

        missing = [c for c in (QUALITY_COLS + CONSTRAINT_SCORE_COLS + ["ReadingLevel"]) if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Score file missing columns {missing}: {score_path}")

        for row in reader:
            q_vals: List[float] = []
            for c in QUALITY_COLS:
                v = _as_float(row.get(c))
                if v is not None:
                    q_vals.append(v)
            quality_per_story.append(mean(q_vals) if q_vals else 0.0)

            # constraint violations from score: sum(1 - value)
            cv = 0.0
            for c in CONSTRAINT_SCORE_COLS:
                iv = _as_int(row.get(c))
                if iv is None:
                    continue
                cv += 1 - iv
            score_constraint_per_story.append(cv)

            reading_levels.append(str(row.get("ReadingLevel", "")).strip())

    return quality_per_story, score_constraint_per_story, reading_levels


def summarize_reading_levels(levels: List[str]) -> Tuple[str, Dict[str, int]]:
    cleaned = [lv for lv in (l.strip() for l in levels) if lv]
    if not cleaned:
        return "", {}
    counts = Counter(cleaned)
    mode = counts.most_common(1)[0][0]
    return mode, dict(counts)


def _norm_stem(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def iter_runs() -> Iterable[Tuple[Path, Path]]:
    """Yield pairs (results_txt, score_csv) matched by stem."""

    score_by_norm: Dict[str, List[Path]] = {}
    for score_csv in sorted(SCORES_DIR.glob("*_SCORE.csv")):
        score_stem = score_csv.stem
        if score_stem.endswith("_SCORE"):
            score_stem = score_stem[: -len("_SCORE")]
        score_by_norm.setdefault(_norm_stem(score_stem), []).append(score_csv)

    for results_txt in sorted(RESULTS_DIR.glob("*.txt")):
        stem = results_txt.stem

        # Fast path: exact filename convention
        exact = SCORES_DIR / f"{stem}_SCORE.csv"
        if exact.exists():
            yield results_txt, exact
            continue

        # Fallback: normalized matching (handles '_' vs '__', case differences, etc.)
        candidates = score_by_norm.get(_norm_stem(stem), [])
        if len(candidates) == 1:
            yield results_txt, candidates[0]


def compute_run_summary(results_txt: Path, score_csv: Path, first_n: Optional[int]) -> RunSummary:
    egra_violations = parse_results_violations(results_txt)
    quality, score_constraints, reading_levels = load_score_file(score_csv)

    n = min(len(egra_violations), len(quality), len(score_constraints))
    if first_n is not None:
        n = min(n, first_n)

    q_first = quality[:n]
    sc_first = score_constraints[:n]
    egra_first = egra_violations[:n]

    combined = [a + b for a, b in zip(sc_first, egra_first)]

    mode, counts = summarize_reading_levels(reading_levels[:n])

    return RunSummary(
        stem=results_txt.stem,
        stories=n,
        quality_mean=mean(q_first),
        quality_std=sample_std(q_first),
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
                # keep it compact
                top = sorted(s.reading_level_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
                f.write("ReadingLevel counts (top): " + ", ".join(f"{k}={v}" for k, v in top) + "\n")
            else:
                f.write("ReadingLevel: (missing)\n")
            f.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-n", type=int, default=None, help="Optional cap on number of stories to use")
    args = ap.parse_args()

    runs = list(iter_runs())
    if not runs:
        raise SystemExit("No matching RESULTS/*.txt + SCORES/*_SCORE.csv pairs found")

    summaries = [compute_run_summary(rtxt, scsv, args.first_n) for rtxt, scsv in runs]
    summaries.sort(key=lambda s: s.stem)

    out = BASE_DIR / "Final_Scores.txt"
    write_final_scores(out, summaries, args.first_n)
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
