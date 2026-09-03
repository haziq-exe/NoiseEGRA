#!/usr/bin/env python
"""Score the English WritingPrompts ablation.

    python scripts/score_english.py --input-dir /kaggle/working/english/Qwen3-8B
    python scripts/score_english.py --input-dir ... --diversity

Constraint adherence uses the exact, judge-free checks. Diversity is computed
**within each prompt group** and then averaged across prompts: stories written
from different prompts are trivially dissimilar, so pooling them would measure
the prompt set rather than the model.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from noiseegra.constraint_metrics_en import (  # noqa: E402
    CONSTRAINT_NAMES,
    EnglishConstraintChecker,
)
from noiseegra.writingprompts import as_constraint  # noqa: E402

COLUMNS = [
    ("run", "Run", "{}"),
    ("n", "N", "{}"),
    ("length_pass", "Len↑", "{:.0%}"),
    ("tense_pass", "Tense↑", "{:.0%}"),
    ("register_pass", "Reg↑", "{:.0%}"),
    ("dialogue_pass", "Dlg↑", "{:.0%}"),
    ("mean_violations", "Viol↓", "{:.2f}"),
    ("mean_words", "Words", "{:.0f}"),
    ("mean_grade", "Grade", "{:.1f}"),
    ("vendi", "Vendi↑", "{:.2f}"),
    ("lexdiv", "LexDiv↑", "{:.3f}"),
]


def read_run(path: Path):
    """Return (stories, prompt_indices) from a runner CSV."""
    stories, prompts = [], []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames and "story" in reader.fieldnames:
            for row in reader:
                if row["story"] and row["story"].strip():
                    stories.append(row["story"])
                    prompts.append(int(row["prompt_index"]))
            return stories, prompts
    # Fall back to the headerless one-story-per-row format.
    with path.open(encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if row and row[0].strip():
                stories.append(row[0])
                prompts.append(0)
    return stories, prompts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-dir")
    ap.add_argument("--diversity", action="store_true")
    ap.add_argument("--backend", default="auto", choices=["auto", "spacy", "regex"])
    ap.add_argument("--constraints", nargs="*", default=list(CONSTRAINT_NAMES))
    ap.add_argument("--max-words", type=int, default=150)
    ap.add_argument("--present-ratio", type=float, default=0.8)
    ap.add_argument("--max-grade", type=float, default=6.0)
    ap.add_argument("--min-group", type=int, default=2,
                    help="prompt groups smaller than this are skipped for diversity")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.is_dir():
        raise SystemExit(f"no such directory: {in_dir}")
    out_dir = Path(args.out_dir) if args.out_dir else in_dir / "EXACT_SCORES"
    out_dir.mkdir(parents=True, exist_ok=True)

    checker = EnglishConstraintChecker(
        max_words=args.max_words,
        present_ratio_threshold=args.present_ratio,
        max_grade_level=args.max_grade,
        backend=args.backend,
        constraints=[as_constraint(c) for c in args.constraints],
    )
    print(f"tense backend: {checker.backend}   constraints: {list(checker.constraints)}")

    scorer = None
    if args.diversity:
        from noiseegra.creativity_metrics import CreativityScorer
        # Built once: constructing it downloads and loads the embedding model.
        scorer = CreativityScorer(["placeholder text one", "placeholder text two"])

    csvs = sorted(p for p in in_dir.glob("*.csv") if p.name != "prompts.json")
    if not csvs:
        raise SystemExit(f"no run CSVs in {in_dir}")

    rows = []
    for path in csvs:
        stories, prompt_idx = read_run(path)
        if not stories:
            continue
        res = checker.evaluate_all(stories, prompt_idx)
        checker.to_csv(stories, out_dir / path.name, prompt_idx)

        row = {
            "run": path.stem,
            "n": res["n_stories"],
            "length_pass": res["pass_rate"]["length"],
            "tense_pass": res["pass_rate"]["present_tense"],
            "register_pass": res["pass_rate"]["simple_register"],
            "dialogue_pass": res["pass_rate"]["dialogue"],
            "mean_violations": res["mean_violations"],
            "mean_words": res["mean_word_count"],
            "mean_grade": res["mean_grade_level"],
            "vendi": float("nan"),
            "lexdiv": float("nan"),
        }

        if scorer is not None:
            groups = defaultdict(list)
            for text, p in zip(stories, prompt_idx):
                groups[p].append(text)
            usable = [g for g in groups.values() if len(g) >= args.min_group]
            if usable:
                vendis, lexes = [], []
                for g in usable:
                    scorer.change_text(g)
                    vendis.append(scorer.semantic_diversity().vendi_score)
                    lexes.append(scorer.lexical_diversity().lexical_score_mean)
                row["vendi"] = statistics.mean(vendis)
                row["lexdiv"] = statistics.mean(lexes)
                row["n_groups"] = len(usable)

        rows.append(row)
        print(f"[ok] {path.stem}: viol={row['mean_violations']:.2f} "
              f"len={row['length_pass']:.0%} tense={row['tense_pass']:.0%} "
              f"reg={row['register_pass']:.0%} dlg={row['dialogue_pass']:.0%}")

    rows.sort(key=lambda r: (r["mean_violations"], r["run"]))
    keys = [k for k, _, _ in COLUMNS if args.diversity or k not in ("vendi", "lexdiv")]
    headers = {k: h for k, h, _ in COLUMNS}
    fmts = {k: f for k, _, f in COLUMNS}

    def cell(row, key):
        v = row[key]
        return "--" if isinstance(v, float) and v != v else fmts[key].format(v)

    lines = ["| " + " | ".join(headers[k] for k in keys) + " |",
             "|" + "|".join("---" for _ in keys) + "|"]
    lines += ["| " + " | ".join(cell(r, k) for k in keys) + " |" for r in rows]

    md = out_dir / "English_Constraint_Table.md"
    md.write_text(
        "# WritingPrompts: exact constraint adherence and within-prompt diversity\n\n"
        f"Thresholds: words <= {args.max_words}, present-verb ratio >= {args.present_ratio}, "
        f"Flesch-Kincaid grade <= {args.max_grade}, at least one quoted line. "
        f"Tense backend: `{checker.backend}`. Diversity is averaged over per-prompt groups.\n\n"
        + "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "English_Constraint_Table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})

    print("\n" + "\n".join(lines))
    print(f"\nwrote {md}")


if __name__ == "__main__":
    main()
