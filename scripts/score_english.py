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
from noiseegra.diversity import read_run_csv  # noqa: E402
from noiseegra.embeddings import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS  # noqa: E402
from noiseegra.run_labels import label_run  # noqa: E402
from noiseegra.writingprompts import as_constraint  # noqa: E402

# key, header, format, one-line explanation printed under the table
COLUMNS = [
    ("run", "condition", "{}", ""),
    ("n", "stories", "{}", "how many stories were scored"),
    ("length_pass", "length", "{:.0%}", "share within the word limit"),
    ("tense_pass", "present tense", "{:.0%}", "share written entirely in the present tense"),
    ("register_pass", "reading level", "{:.0%}", "share at or below the grade-level limit"),
    ("dialogue_pass", "dialogue", "{:.0%}", "share containing at least one quoted line"),
    ("mean_violations", "broken", "{:.2f}", "mean number of the four constraints broken per story"),
    ("mean_words", "words", "{:.0f}", "mean story length"),
    ("mean_grade", "grade", "{:.1f}", "mean Flesch-Kincaid grade level"),
    ("vendi", "Vendi", "{:.2f}", "effective number of distinct stories per prompt"),
    ("lexdiv", "lexical", "{:.3f}", "1 minus self-BLEU: how much the wording varies"),
]


def render_table(rows, keys, headers, fmts, sep="  ", left=("run", "condition")):
    """Plain aligned columns; text columns left-justified, numbers right."""
    cells = [[headers[k] for k in keys]]
    for r in rows:
        cells.append(["--" if isinstance(r[k], float) and r[k] != r[k]
                      else fmts[k].format(r[k]) for k in keys])
    widths = [max(len(row[i]) for row in cells) for i in range(len(keys))]
    out = []
    for n, row in enumerate(cells):
        out.append(sep.join(c.ljust(w) if k in left else c.rjust(w)
                            for c, w, k in zip(row, widths, keys)).rstrip())
        if n == 0:
            out.append("-" * len(out[0]))
    return out


def constraint_legend(max_words, max_grade, present_ratio) -> list:
    return [
        f"length          at most {max_words} words",
        f"present tense   at least {present_ratio:.0%} of finite verbs in the present",
        f"reading level   Flesch-Kincaid grade at most {max_grade:g}",
        "dialogue        at least one line inside quotation marks",
    ]


def score_condition(stories, prompt_idx, checker, scorer=None, min_group=2):
    """Constraint pass rates plus within-prompt diversity for one condition."""
    res = checker.evaluate_all(stories, prompt_idx)
    vendi = lexdiv = float("nan")
    if scorer is not None:
        groups = defaultdict(list)
        for text, p in zip(stories, prompt_idx):
            groups[p].append(text)
        usable = [g for g in groups.values() if len(g) >= min_group]
        if usable:
            vs, ls = [], []
            for g in usable:
                scorer.change_text(g)
                vs.append(scorer.semantic_diversity().vendi_score)
                ls.append(scorer.lexical_diversity().lexical_score_mean)
            vendi, lexdiv = statistics.mean(vs), statistics.mean(ls)
    return {
        "n": res["n_stories"],
        "length_pass": res["pass_rate"]["length"],
        "tense_pass": res["pass_rate"]["present_tense"],
        "register_pass": res["pass_rate"]["simple_register"],
        "dialogue_pass": res["pass_rate"]["dialogue"],
        "mean_violations": res["mean_violations"],
        "mean_words": res["mean_word_count"],
        "mean_grade": res["mean_grade_level"],
        "vendi": vendi,
        "lexdiv": lexdiv,
    }


LIVE_KEYS = ["run", "n", "length_pass", "tense_pass", "register_pass", "dialogue_pass",
             "mean_violations", "mean_words", "mean_grade", "vendi", "lexdiv"]
_LIVE_HEAD = {k: h for k, h, _, _ in COLUMNS}
_LIVE_FMT = {k: f for k, _, f, _ in COLUMNS}
_LIVE_W = {"run": 36, "n": 8, "length_pass": 7, "tense_pass": 14, "register_pass": 14,
           "dialogue_pass": 9, "mean_violations": 7, "mean_words": 6, "mean_grade": 6,
           "vendi": 7, "lexdiv": 8}

LIVE_HEADER = "".join(
    _LIVE_HEAD[k].ljust(_LIVE_W[k]) if k == "run" else _LIVE_HEAD[k].rjust(_LIVE_W[k])
    for k in LIVE_KEYS
)


def live_row(label: str, r: dict) -> str:
    """One line of the table printed as each condition finishes generating."""
    out = [label[: _LIVE_W["run"] - 1].ljust(_LIVE_W["run"])]
    for k in LIVE_KEYS[1:]:
        v = r[k]
        cell = "--" if isinstance(v, float) and v != v else _LIVE_FMT[k].format(v)
        out.append(cell.rjust(_LIVE_W[k]))
    return "".join(out)


def read_run(path: Path):
    """Return (stories, prompt_indices) from a runner CSV."""
    return read_run_csv(path)


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
    ap.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL,
                    help=f"registry key or HF id. keys: {', '.join(EMBEDDING_MODELS)}. "
                         "use bge-m3 to match the published Arabic runs")
    ap.add_argument("--truncate-words", type=int, default=None,
                    help="cut every story to its first N words before scoring diversity, "
                         "so the score does not partly measure output length")
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
        scorer = CreativityScorer(
            ["placeholder text one", "placeholder text two"],
            embedding_model=args.embedding_model,
            truncate_words=args.truncate_words,
        )

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

        label = label_run(path.stem)
        row = {
            "run": label.text,
            "_order": label.sort_key,
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
        print(f"  scored {label.text}", flush=True)

    rows.sort(key=lambda r: r["_order"])
    keys = [k for k, _, _, _ in COLUMNS if args.diversity or k not in ("vendi", "lexdiv")]
    headers = {k: h for k, h, _, _ in COLUMNS}
    fmts = {k: f for k, _, f, _ in COLUMNS}
    notes = {k: n for k, _, _, n in COLUMNS}

    legend = constraint_legend(args.max_words, args.max_grade, args.present_ratio)
    table = render_table(rows, keys, headers, fmts)

    print("\n" + "=" * max(len(line) for line in table))
    print("\n".join(table))
    print("\nwhat has to be true for a story to pass")
    for line in legend:
        print("  " + line)
    print("\nwhat the columns mean")
    for k in keys:
        if notes[k]:
            print(f"  {headers[k]:<15}{notes[k]}")
    print(f"\ntense checked with the {checker.backend} backend. Diversity is computed "
          "within a\nprompt group and averaged across groups.")

    md_rows = ["| " + " | ".join(headers[k] for k in keys) + " |",
               "|" + "|".join("---" for _ in keys) + "|"]
    for r in rows:
        md_rows.append("| " + " | ".join(
            "--" if isinstance(r[k], float) and r[k] != r[k] else fmts[k].format(r[k])
            for k in keys) + " |")

    md = out_dir / "English_Constraint_Table.md"
    md.write_text(
        "# WritingPrompts: constraint adherence and within-prompt diversity\n\n"
        "## What has to be true for a story to pass\n\n"
        + "".join(f"- `{line}`\n" for line in legend)
        + f"\nTense checked with the `{checker.backend}` backend. Diversity is computed "
          "within a prompt group and averaged across groups.\n\n"
        + "\n".join(md_rows)
        + "\n\n## What the columns mean\n\n"
        + "".join(f"- **{headers[k]}** {notes[k]}\n" for k in keys if notes[k]),
        encoding="utf-8",
    )
    with (out_dir / "English_Constraint_Table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})

    print(f"\nwrote {md}")


if __name__ == "__main__":
    main()
