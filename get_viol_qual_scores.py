from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── helpers ──────────────────────────────────────────────────────────────────

def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sample_std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def _parse_numbers(cell: object) -> List[float]:
    text = str(cell).strip() if cell is not None else ""
    return [float(x) for x in re.findall(r"\d+\.?\d*", text)] if text else []


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


# Column aliases — same sets as the original script
_QUALITY_ALIASES = {
    "readability": {"readability", "readability10"},
    "logic": {"logic", "logic10"},
    "grammarandlinguistic": {
        "grammarandlinguistic",
        "grammarlinguistic",
        "grammarandlinguistic10",
        "grammarlinguistic10",
    },
}

_CONSTRAINT_ALIASES = {
    "totalmodalcollapse": {"totalmodalcollapse"},
    "structure": {"structure"},
    "vocabularylevel": {"vocabularylevel"},
    "stereotypes": {"stereotypes"},
    "genderbalanced": {"genderbalanced"},
}


def _build_column_map(headers: List[str]) -> Dict[str, str]:
    norm_headers = {h: _norm(h) for h in headers}
    col_map: Dict[str, str] = {}
    for canonical, aliases in {**_QUALITY_ALIASES, **_CONSTRAINT_ALIASES}.items():
        matched = next((h for h, hn in norm_headers.items() if hn in aliases), None)
        if matched is None:
            raise ValueError(f"Missing expected column '{canonical}' in: {headers}")
        col_map[canonical] = matched
    return col_map


# ── parsers ───────────────────────────────────────────────────────────────────

def _parse_egra_violations(results_txt: Path, run_name: str) -> List[int]:
    """
    Extracts per-story violation counts from the CONSTRAINT section of the
    results file that belongs to `run_name`.
    Returns a list indexed by story number.
    """
    constraint_header_re = re.compile(r"^-+\s*(.+?)\s+CONSTRAINT\s*-+$")
    story_re = re.compile(r"^- Story #(\d+): violations=(\d+)")

    in_section = False
    by_index: Dict[int, int] = {}

    for raw in results_txt.read_text(encoding="utf-8").splitlines():
        line = raw.strip()

        m = constraint_header_re.match(line)
        if m:
            in_section = (m.group(1) == run_name)
            continue

        if in_section:
            m = story_re.match(line)
            if m:
                by_index[int(m.group(1))] = int(m.group(2))

    if not by_index:
        raise ValueError(
            f"No CONSTRAINT section found for run '{run_name}' in {results_txt}"
        )

    max_idx = max(by_index)
    return [by_index.get(i, 0) for i in range(max_idx + 1)]


def _parse_creativity(results_txt: Path, run_name: str) -> Tuple[float, float]:
    """
    Reads 'Combined Creativity Score' from the creativity block for `run_name`.
    Returns (mean, std).
    """
    run_header_re = re.compile(r"^----\s*(.+?)\s*----\s*$")
    creativity_re = re.compile(
        r"^Combined Creativity Score:\s*([0-9.]+)\s*\(std\s*([0-9.]+)\)"
    )

    in_section = False
    for raw in results_txt.read_text(encoding="utf-8").splitlines():
        line = raw.strip()

        m = run_header_re.match(line)
        if m and "CONSTRAINT" not in line:
            in_section = (m.group(1) == run_name)
            continue

        if in_section:
            m = creativity_re.match(line)
            if m:
                return float(m.group(1)), float(m.group(2))

    raise ValueError(
        f"No Combined Creativity Score found for run '{run_name}' in {results_txt}"
    )


def _parse_scores_csv(scores_csv: Path) -> Tuple[List[float], List[float]]:
    """
    Reads the scores CSV and returns:
      - quality_story_scores  : mean(Readability, Logic, Grammar) per story
      - score_constraint_counts: sum(1 - value) over the 5 binary constraint cols per story
    """
    quality_story_scores: List[float] = []
    score_constraint_counts: List[float] = []

    with scores_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        headers = list(reader.fieldnames or [])
        col_map = _build_column_map(headers)

        for row in reader:
            quality_vals: List[float] = []
            for c in _QUALITY_ALIASES:
                quality_vals.extend(_parse_numbers(row.get(col_map[c], "")))
            if quality_vals:
                quality_story_scores.append(_mean(quality_vals))

            violation_vals: List[float] = []
            for c in _CONSTRAINT_ALIASES:
                parsed = _parse_numbers(row.get(col_map[c], ""))
                violation_vals.extend(1 - v for v in parsed)
            if violation_vals:
                score_constraint_counts.append(sum(violation_vals))

    return quality_story_scores, score_constraint_counts


# ── public API ────────────────────────────────────────────────────────────────

def compute_combined_metrics(
    results_txt: str | Path,
    scores_csv: str | Path,
    run_name: str,
    first_n: Optional[int] = None,
) -> Dict[str, object]:
    """
    Combine a *_RESULTS.txt file and a *_Scores.csv file for a named run.

    Parameters
    ----------
    results_txt : path to the *_RESULTS.txt file
    scores_csv  : path to the *_Scores.csv file
    run_name    : exact run identifier, e.g. "Jais__BASELINE__temp1p8__topk40"
    first_n     : cap on number of stories to use (None = use all)

    Returns
    -------
    dict with keys:
        run_name, stories_used,
        quality_mean, quality_std,
        constraints_broken_mean, constraints_broken_std,
        creativity_mean, creativity_std,
        quality_story_scores, score_constraint_counts,
        egra_constraint_counts, combined_constraints
    """
    results_txt = Path(results_txt)
    scores_csv = Path(scores_csv)
    run_name= run_name.strip()  

    quality_scores, score_constraint_counts = _parse_scores_csv(scores_csv)
    egra_violations = _parse_egra_violations(results_txt, run_name)
    creativity_mean, creativity_std = _parse_creativity(results_txt, run_name)

    n_available = min(len(quality_scores), len(score_constraint_counts), len(egra_violations))
    n_used = min(first_n, n_available) if first_n is not None else n_available

    q = quality_scores[:n_used]
    sc = score_constraint_counts[:n_used]
    ev = egra_violations[:n_used]
    combined = [a + b for a, b in zip(sc, ev)]

    return {
        "run_name": run_name,
        "stories_used": n_used,
        # per-story lists
        "quality_story_scores": q,
        "score_constraint_counts": sc,
        "egra_constraint_counts": ev,
        "combined_constraints": combined,
        # summary stats
        "quality_mean": _mean(q),
        "quality_std": _sample_std(q),
        "constraints_broken_mean": _mean(combined),
        "constraints_broken_std": _sample_std(combined),
        "creativity_mean": creativity_mean,
        "creativity_std": creativity_std,
    }





result = compute_combined_metrics(
    results_txt="EGRA_RESULTS/ATT_JAIS/Jais_RESULTS.txt",
    scores_csv="EGRA_RESULTS/ATT_JAIS/ATTN_Jais_Middle_Scores.csv",
    run_name="Jais__ATTN__L12-20__std4p6375",
    first_n=50,   # or None to use everything
)

print(result["quality_mean"], result["quality_std"])           # e.g. 7.04
print(result["constraints_broken_mean"], result["constraints_broken_std"]) # e.g. 2.14
print(result["creativity_mean"], result["creativity_std"])         # e.g. 0.5021