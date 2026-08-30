#!/usr/bin/env python
"""Score an orthogonal-steering ablation with the exact, LLM-free constraint checks.

    python scripts/score_orthosteer.py --input-dir experiment_results/OrthoSteer
    python scripts/score_orthosteer.py --input-dir experiment_results/OrthoSteer --diversity

Writes, under ``<input-dir>/EXACT_SCORES/``:
    <run_id>.csv                 per-story metrics
    Ortho_Constraint_Table.md    one row per condition
    Ortho_Constraint_Table.csv   the same, machine-readable

``--diversity`` additionally computes Vendi Score and lexical diversity, which
downloads the ``BAAI/bge-m3`` embedding model on first use; leave it off for a
fast constraint-only pass.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from noiseegra.constraint_metrics import ExactConstraintChecker  # noqa: E402

COLUMNS = [
    ("run", "Run", "{}"),
    ("n", "N", "{}"),
    ("length_pass", "Len↑", "{:.0%}"),
    ("tense_pass", "Tense↑", "{:.0%}"),
    ("register_pass", "Reg↑", "{:.0%}"),
    ("mean_violations", "Viol↓", "{:.2f}"),
    ("mean_words", "Words", "{:.0f}"),
    ("median_words", "Med.W", "{:.0f}"),
    ("mean_present_ratio", "PresR", "{:.2f}"),
    ("mean_sent_words", "W/Sent", "{:.1f}"),
    ("vendi", "Vendi↑", "{:.2f}"),
    ("lexdiv", "LexDiv↑", "{:.3f}"),
]


def read_stories(path: Path) -> list[str]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if row and row[0].strip():
                out.append(row[0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default="experiment_results/OrthoSteer")
    ap.add_argument("--out-dir", help="defaults to <input-dir>/EXACT_SCORES")
    ap.add_argument("--diversity", action="store_true", help="also compute Vendi + lexical diversity")
    ap.add_argument("--backend", default="auto", choices=["auto", "camel", "regex"],
                    help="tense backend; 'camel' needs camel-tools + its model data")
    ap.add_argument("--max-words", type=int, default=60)
    ap.add_argument("--present-ratio", type=float, default=0.8)
    ap.add_argument("--mean-sentence-words", type=float, default=12.0)
    ap.add_argument("--max-sentence-words", type=int, default=18)
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.is_dir():
        raise SystemExit(f"no such directory: {in_dir}")
    out_dir = Path(args.out_dir) if args.out_dir else in_dir / "EXACT_SCORES"
    out_dir.mkdir(parents=True, exist_ok=True)

    checker = ExactConstraintChecker(
        max_words=args.max_words,
        present_ratio_threshold=args.present_ratio,
        mean_sentence_words=args.mean_sentence_words,
        max_sentence_words=args.max_sentence_words,
        backend=args.backend,
    )
    print(f"tense backend: {checker.backend}")

    csvs = sorted(p for p in in_dir.glob("*.csv") if not p.name.endswith("_manifest.csv"))
    if not csvs:
        raise SystemExit(f"no story CSVs in {in_dir}")

    rows = []
    for path in csvs:
        stories = read_stories(path)
        if not stories:
            print(f"[skip] {path.name}: empty")
            continue
        res = checker.evaluate_all(stories)
        checker.to_csv(stories, out_dir / path.name)

        row = {
            "run": path.stem,
            "n": res["n_stories"],
            "length_pass": res["pass_rate"]["length"],
            "tense_pass": res["pass_rate"]["present_tense"],
            "register_pass": res["pass_rate"]["simple_register"],
            "mean_violations": res["mean_violations"],
            "mean_words": res["mean_word_count"],
            "median_words": res["median_word_count"],
            "mean_present_ratio": res["mean_present_ratio"],
            "mean_sent_words": res["mean_sentence_words"],
            "vendi": float("nan"),
            "lexdiv": float("nan"),
        }

        if args.diversity:
            from noiseegra.creativity_metrics import CreativityScorer

            scorer = CreativityScorer(stories)
            row["vendi"] = scorer.semantic_diversity().vendi_score
            row["lexdiv"] = scorer.lexical_diversity().lexical_score_mean

        rows.append(row)
        print(f"[ok]   {path.stem}: viol={row['mean_violations']:.2f} "
              f"len={row['length_pass']:.0%} tense={row['tense_pass']:.0%} "
              f"reg={row['register_pass']:.0%}")

    rows.sort(key=lambda r: (r["mean_violations"], r["run"]))
    keys = [k for k, _, _ in COLUMNS if args.diversity or k not in ("vendi", "lexdiv")]
    headers = {k: h for k, h, _ in COLUMNS}
    fmts = {k: f for k, _, f in COLUMNS}

    def cell(row, key):
        v = row[key]
        if isinstance(v, float) and v != v:  # NaN
            return "--"
        return fmts[key].format(v)

    lines = [
        "| " + " | ".join(headers[k] for k in keys) + " |",
        "|" + "|".join("---" for _ in keys) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(row, k) for k in keys) + " |")

    md = out_dir / "Ortho_Constraint_Table.md"
    md.write_text(
        "# Exact EGRA constraint adherence\n\n"
        f"Thresholds: words <= {args.max_words}, present-verb ratio >= "
        f"{args.present_ratio}, mean sentence <= {args.mean_sentence_words} words, "
        f"longest sentence <= {args.max_sentence_words} words. "
        f"Tense backend: `{checker.backend}`.\n\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    with (out_dir / "Ortho_Constraint_Table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in keys})

    print("\n" + "\n".join(lines))
    print(f"\nwrote {md} and {md.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
