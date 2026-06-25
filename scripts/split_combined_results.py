#!/usr/bin/env python3
"""Split a combined {model}_RESULTS.txt into per-run RESULTS/{run_id}.txt files.

Use this for experiment outputs created before per-run RESULTS files were written
automatically by run_story_experiments().
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring_common import RUN_HEADER_RE, resolve_results_dir


CONSTRAINT_HEADER_RE = re.compile(r"^-+\s*(.+?)\s+CONSTRAINT\s*-+$")


def split_combined_results(combined_path: Path, output_dir: Path) -> list[str]:
    """Split combined file into per-run sections. Returns run ids written."""
    text = combined_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    sections: dict[str, list[str]] = {}
    current_run: str | None = None

    for line in lines:
        m = RUN_HEADER_RE.match(line.strip())
        if m and "CONSTRAINT" not in line:
            current_run = m.group(1).strip()
            sections.setdefault(current_run, [])
            sections[current_run].append(line)
            continue
        if current_run is not None:
            sections[current_run].append(line)

    if not sections:
        raise ValueError(f"No run sections found in {combined_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for run_id, section_lines in sections.items():
        out_path = output_dir / f"{run_id}.txt"
        out_path.write_text("\n".join(section_lines).rstrip() + "\n", encoding="utf-8")
        written.append(run_id)

    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="Split combined *_RESULTS.txt into per-run files")
    ap.add_argument(
        "combined_file",
        type=Path,
        help="Path to combined {model}_RESULTS.txt",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for per-run .txt files (default: <parent>/RESULTS)",
    )
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Experiment results root (used when output-dir omitted and file lives under it)",
    )
    args = ap.parse_args()

    combined_path = args.combined_file
    if not combined_path.is_file():
        raise SystemExit(f"File not found: {combined_path}")

    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        base = resolve_results_dir(args.results_dir)
        if combined_path.parent == base or combined_path.parent.name in (
            "baseline",
            "ResidNoise",
            "AttnNoise",
            "EmbedNoise",
            "AENIMaxW",
        ):
            out_dir = combined_path.parent / "RESULTS"
        else:
            out_dir = base / "RESULTS"

    written = split_combined_results(combined_path, out_dir)
    print(f"Wrote {len(written)} per-run RESULTS files to {out_dir}")
    for run_id in written:
        print(f"  - {run_id}.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
