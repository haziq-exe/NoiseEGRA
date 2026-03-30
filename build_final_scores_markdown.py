#!/usr/bin/env python3
"""Build markdown summary table from Final_Scores and RESULTS reports.

Output:
  EGRA_RESULTS/Final_Scores_Table.md
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional


BASE = Path("EGRA_RESULTS")
FINAL_TXT = BASE / "Final_Scores.txt"
RESULTS_DIR = BASE / "RESULTS"
SCORES_DIR = BASE / "SCORES"
OUT_MD = BASE / "Final_Scores_Table.md"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def parse_final_scores(path: Path) -> List[dict]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    block_re = re.compile(r"^----\s*(.+?)\s*----$")
    quality_re = re.compile(r"^Quality mean/std:\s*([0-9.]+)\s*/\s*([0-9.]+)")
    viol_re = re.compile(r"^Violations/story \(score\+egra\) mean/std:\s*([0-9.]+)\s*/\s*([0-9.]+)")
    read_re = re.compile(r"^ReadingLevel mode:\s*(.+)$")

    rows: List[dict] = []
    cur: Optional[dict] = None

    for ln in lines:
        m = block_re.match(ln)
        if m:
            if cur is not None:
                rows.append(cur)
            run_id = m.group(1).strip()
            if "__" in run_id:
                model_name, method = run_id.split("__", 1)
            else:
                model_name, method = run_id, ""
            cur = {
                "run_id": run_id,
                "model_name": model_name,
                "method": method,
                "quality": "",
                "violations": "",
                "reading_level": "",
            }
            continue

        if cur is None:
            continue

        m = quality_re.match(ln)
        if m:
            cur["quality"] = f"{m.group(1)} / {m.group(2)}"
            continue

        m = viol_re.match(ln)
        if m:
            cur["violations"] = f"{m.group(1)} / {m.group(2)}"
            continue

        m = read_re.match(ln)
        if m:
            cur["reading_level"] = m.group(1).strip()
            continue

    if cur is not None:
        rows.append(cur)

    return rows


def parse_results_metrics(results_dir: Path) -> Dict[str, dict]:
    vendi_re = re.compile(r"^Semantic\s*-\s*Vendi\s*Score:\s*([0-9.]+)")
    lexical_re = re.compile(r"^Lexical\s*Diversity\s*Score:\s*([0-9.]+)\s*\(std\s*([0-9.]+)\)")

    out: Dict[str, dict] = {}
    for txt in sorted(results_dir.glob("*.txt")):
        vendi = None
        lexical_mean = None
        lexical_std = None
        for raw in txt.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            m = vendi_re.match(line)
            if m:
                vendi = m.group(1)
                continue
            m = lexical_re.match(line)
            if m:
                lexical_mean = m.group(1)
                lexical_std = m.group(2)
                continue

        out[txt.stem] = {
            "vendi": vendi,
            "lexical_mean": lexical_mean,
            "lexical_std": lexical_std,
        }

    return out


def parse_modal_collapse_counts(scores_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for score_csv in sorted(scores_dir.glob("*_SCORE.csv")):
        stem = score_csv.stem
        if stem.endswith("_SCORE"):
            stem = stem[: -len("_SCORE")]

        c = 0
        with score_csv.open("r", encoding="utf-8-sig", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    value = int(float(str(row.get("TotalModalCollapse", "0")).strip() or "0"))
                except Exception:
                    value = 0
                if value == 1:
                    c += 1
        counts[stem] = c

    return counts


def build_markdown(
    final_rows: List[dict],
    results_map: Dict[str, dict],
    collapse_counts: Dict[str, int],
) -> str:
    # normalized fallback for minor naming differences
    results_norm = {norm(k): v for k, v in results_map.items()}
    collapse_norm = {norm(k): v for k, v in collapse_counts.items()}

    lines: List[str] = []
    lines.append("# Final Scores Table")
    lines.append("")
    lines.append(
        "| Model Name | Method | Vendi Score | Lexical Diversity (std) | Quality mean/std | Reading Level | Total Modal Collapse Instances | Violations/story mean/std |"
    )
    lines.append("|---|---|---:|---:|---:|---|---:|---:|")

    for row in final_rows:
        run_id = row["run_id"]
        result = results_map.get(run_id)
        if result is None:
            result = results_norm.get(norm(run_id), {})

        collapse_value = collapse_counts.get(run_id)
        if collapse_value is None:
            collapse_value = collapse_norm.get(norm(run_id))

        vendi = result.get("vendi") if result else None
        lexical_mean = result.get("lexical_mean") if result else None
        lexical_std = result.get("lexical_std") if result else None

        vendi_str = vendi if vendi else "N/A"
        if lexical_mean and lexical_std:
            lexical_str = f"{lexical_mean} (std {lexical_std})"
        else:
            lexical_str = "N/A"

        method = row["method"] if row["method"] else "-"
        reading = row["reading_level"] if row["reading_level"] else "-"
        quality = row["quality"] if row["quality"] else "-"
        violations = row["violations"] if row["violations"] else "-"
        collapse_str = str(collapse_value) if collapse_value is not None else "N/A"

        lines.append(
            f"| {row['model_name']} | {method} | {vendi_str} | {lexical_str} | {quality} | {reading} | {collapse_str} | {violations} |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    final_rows = parse_final_scores(FINAL_TXT)
    results_map = parse_results_metrics(RESULTS_DIR)
    collapse_counts = parse_modal_collapse_counts(SCORES_DIR)

    md = build_markdown(final_rows, results_map, collapse_counts)
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"Wrote: {OUT_MD}")
    print(f"Rows: {len(final_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
