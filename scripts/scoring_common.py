"""Shared parsers and alignment for EGRA scoring aggregation scripts."""

from __future__ import annotations

import csv
import logging
import math
import os
import re
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

QUALITY_COLS = ["Readability", "Logic", "GrammarandLinguistics"]
CONSTRAINT_SCORE_COLS = [
    "TotalModalCollapse",
    "Structure",
    "VocabularyLevel",
    "Stereotypes",
    "Gender-balanced",
]
STORY_NUMBER_COL = "Story number"

RUN_HEADER_RE = re.compile(r"^----\s*(.+?)\s*----\s*$")
STORY_VIOLATION_RE = re.compile(r"^-\s*Story\s*#(\d+):\s*violations=(\d+)")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_results_dir(results_dir: Optional[Path] = None) -> Path:
    if results_dir is not None:
        return Path(results_dir)
    env = os.getenv("EGRA_RESULTS_DIR")
    if env:
        p = Path(env)
        return p if p.is_absolute() else repo_root() / p
    return repo_root() / "experiment_results"


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def sample_std(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def as_float(cell: object) -> Optional[float]:
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


def as_int(cell: object) -> Optional[int]:
    v = as_float(cell)
    if v is None:
        return None
    return int(round(v))


def norm_stem(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def score_stem_from_path(score_csv: Path) -> str:
    stem = score_csv.stem
    if stem.endswith("_SCORE"):
        return stem[: -len("_SCORE")]
    return stem


def parse_results_violations(
    results_path: Path,
    *,
    run_id: Optional[str] = None,
) -> Dict[int, int]:
    """Parse EGRA violations keyed by 0-based story index from a RESULTS file.

  When ``run_id`` is set, only lines inside that run's section are parsed
  (between ``---- {run_id} ----`` headers). For per-run RESULTS files, omit
  ``run_id`` to parse the whole file.
    """
    text = results_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    in_section = run_id is None
    if run_id is not None:
        target = norm_stem(run_id)

    by_idx: Dict[int, int] = {}
    for raw in lines:
        line = raw.strip()
        m_header = RUN_HEADER_RE.match(line)
        if m_header and "CONSTRAINT" not in line:
            header_id = m_header.group(1).strip()
            if run_id is None:
                in_section = True
            else:
                in_section = norm_stem(header_id) == target
            continue
        if not in_section:
            continue
        m_story = STORY_VIOLATION_RE.match(line)
        if m_story:
            by_idx[int(m_story.group(1))] = int(m_story.group(2))

    return by_idx


@dataclass(frozen=True)
class StoryScoreRow:
    story_number: int
    quality: float
    score_constraint_violations: float
    reading_level: str


def load_score_rows(score_path: Path) -> Dict[int, StoryScoreRow]:
    """Load per-story scores keyed by 1-based story number."""
    required = QUALITY_COLS + CONSTRAINT_SCORE_COLS + [STORY_NUMBER_COL, "ReadingLevel"]
    rows: Dict[int, StoryScoreRow] = {}

    with score_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Empty score file: {score_path}")

        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Score file missing columns {missing}: {score_path}")

        for row in reader:
            story_num = as_int(row.get(STORY_NUMBER_COL))
            if story_num is None:
                continue

            q_vals = [v for c in QUALITY_COLS if (v := as_float(row.get(c))) is not None]
            quality = mean(q_vals) if q_vals else 0.0

            cv = 0.0
            for c in CONSTRAINT_SCORE_COLS:
                iv = as_int(row.get(c))
                if iv is None:
                    continue
                cv += 1 - iv

            rows[int(story_num)] = StoryScoreRow(
                story_number=int(story_num),
                quality=quality,
                score_constraint_violations=cv,
                reading_level=str(row.get("ReadingLevel", "")).strip(),
            )

    return rows


@dataclass(frozen=True)
class AlignedStory:
    story_number: int
    quality: float
    score_constraint_violations: float
    egra_violations: int
    reading_level: str

    @property
    def total_violations(self) -> float:
        return self.score_constraint_violations + self.egra_violations


def align_story_data(
    results_path: Path,
    score_path: Path,
    *,
    run_id: Optional[str] = None,
    strict: bool = True,
) -> List[AlignedStory]:
    """Align EGRA violations (0-based index) with judge scores (1-based number)."""
    egra_by_idx = parse_results_violations(results_path, run_id=run_id)
    score_by_num = load_score_rows(score_path)

    if not egra_by_idx and not score_by_num:
        return []

    aligned: List[AlignedStory] = []
    egra_keys = {idx + 1 for idx in egra_by_idx}
    score_keys = set(score_by_num)
    common = sorted(egra_keys & score_keys)

    only_egra = sorted(egra_keys - score_keys)
    only_score = sorted(score_keys - egra_keys)

    if only_egra or only_score:
        msg = (
            f"Story alignment mismatch for {results_path.name} + {score_path.name}: "
            f"egra_only={only_egra[:10]}{'...' if len(only_egra) > 10 else ''}, "
            f"score_only={only_score[:10]}{'...' if len(only_score) > 10 else ''}"
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)
        common = sorted(egra_keys | score_keys)

    for story_num in common:
        egra_idx = story_num - 1
        egra_v = egra_by_idx.get(egra_idx, 0)
        score_row = score_by_num.get(story_num)
        if score_row is None:
            if strict:
                raise ValueError(f"Missing score row for story {story_num} in {score_path}")
            continue
        aligned.append(
            AlignedStory(
                story_number=story_num,
                quality=score_row.quality,
                score_constraint_violations=score_row.score_constraint_violations,
                egra_violations=egra_v,
                reading_level=score_row.reading_level,
            )
        )

    return aligned


def summarize_reading_levels(levels: List[str]) -> Tuple[str, Dict[str, int]]:
    cleaned = [lv for lv in (l.strip() for l in levels) if lv]
    if not cleaned:
        return "", {}
    counts = Counter(cleaned)
    mode = counts.most_common(1)[0][0]
    return mode, dict(counts)


def iter_run_pairs(
    results_dir: Path,
    scores_dir: Path,
) -> Iterable[Tuple[Path, Path]]:
    """Yield (results_txt, score_csv) pairs matched by run stem."""
    if not results_dir.is_dir():
        logger.warning("RESULTS directory does not exist: %s", results_dir)
        return

    score_by_norm: Dict[str, List[Path]] = {}
    if scores_dir.is_dir():
        for score_csv in sorted(scores_dir.glob("*_SCORE.csv")):
            score_by_norm.setdefault(norm_stem(score_stem_from_path(score_csv)), []).append(score_csv)
    else:
        logger.warning("SCORES directory does not exist: %s", scores_dir)

    matched_score_stems: set[str] = set()

    for results_txt in sorted(results_dir.glob("*.txt")):
        stem = results_txt.stem
        exact = scores_dir / f"{stem}_SCORE.csv"
        if exact.exists():
            matched_score_stems.add(norm_stem(stem))
            yield results_txt, exact
            continue

        candidates = score_by_norm.get(norm_stem(stem), [])
        if len(candidates) == 1:
            matched_score_stems.add(norm_stem(stem))
            yield results_txt, candidates[0]
        elif len(candidates) > 1:
            logger.warning(
                "Ambiguous score match for RESULTS/%s (%d candidates); skipped",
                results_txt.name,
                len(candidates),
            )
        else:
            logger.warning("No SCORE file for RESULTS/%s; skipped", results_txt.name)

    for score_csv in sorted(scores_dir.glob("*_SCORE.csv")) if scores_dir.is_dir() else []:
        stem = score_stem_from_path(score_csv)
        if norm_stem(stem) not in matched_score_stems:
            logger.warning("No RESULTS file for SCORES/%s; skipped", score_csv.name)
