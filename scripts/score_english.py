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
    CONSTRAINT_SHORT,
    EnglishConstraintChecker,
)
from noiseegra.diversity import read_run_csv  # noqa: E402
from noiseegra.embeddings import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS  # noqa: E402
from noiseegra.run_labels import label_run  # noqa: E402

# Two tables rather than one wide one: with twelve requirements a single table
# runs past 190 characters and stops being readable, which is the thing it is for.
#
#   summary       one row per condition, the numbers you compare first
#   requirements  one row per requirement, one column per condition
MAIN_COLUMNS = [
    ("run", "condition", "{}", ""),
    ("n", "stories", "{}", "how many stories were scored"),
    ("mean_violations", "broken", "{:.2f}", "mean requirements broken per story"),
    ("mean_words", "words", "{:.0f}", "mean story length"),
    ("mean_sentences", "sents", "{:.1f}", "mean number of sentences"),
    ("mean_grade", "FK", "{:.1f}", "mean Flesch-Kincaid grade level"),
]
DIVERSITY_COLUMNS = [
    ("vendi", "Vendi", "{:.2f}", "effective number of distinct stories per prompt"),
    ("lexdiv", "lexical", "{:.3f}", "1 minus self-BLEU: how much the wording varies"),
]


def build_columns(constraints=(), diversity: bool = True, per_constraint: bool = False):
    """(keys, headers, formats, notes) for the summary table."""
    cols = list(MAIN_COLUMNS)
    if per_constraint:
        cols += [(f"pass_{c}", CONSTRAINT_SHORT.get(c, c[:6]), "{:.0%}", "")
                 for c in constraints]
    if diversity:
        cols += DIVERSITY_COLUMNS
    return ([k for k, _, _, _ in cols],
            {k: h for k, h, _, _ in cols},
            {k: f for k, _, f, _ in cols},
            {k: n for k, _, _, n in cols})


def render_table(rows, keys, headers, fmts, sep="  ", left=("run", "condition",
                                                            "requirement")):
    """Plain aligned columns; text columns left-justified, numbers right."""
    cells = [[headers[k] for k in keys]]
    for r in rows:
        cells.append(["--" if isinstance(r.get(k), float) and r[k] != r[k]
                      else fmts[k].format(r[k]) for k in keys])
    widths = [max(len(row[i]) for row in cells) for i in range(len(keys))]
    out = []
    for n, row in enumerate(cells):
        out.append(sep.join(c.ljust(w) if k in left else c.rjust(w)
                            for c, w, k in zip(row, widths, keys)).rstrip())
        if n == 0:
            out.append("-" * len(out[0]))
    return out


def constraint_legend(checker) -> list:
    """One line per requirement, in the order the prompt lists them."""
    req = checker.requirements()
    return [f"{i + 1:>2}. {req[c]}" for i, c in enumerate(checker.constraints)]


def requirement_table(rows, checker, shorts):
    """Requirements down the side, conditions across the top."""
    req = checker.requirements_short()
    keys = ["requirement"] + [f"c{i}" for i in range(len(rows))]
    headers = {"requirement": "the story must ..."}
    headers.update({f"c{i}": shorts[i] for i in range(len(rows))})
    fmts = {k: ("{}" if k == "requirement" else "{:.0%}") for k in keys}
    table_rows = []
    for name in checker.constraints:
        row = {"requirement": req[name]}
        for i, r in enumerate(rows):
            row[f"c{i}"] = r.get(f"pass_{name}", float("nan"))
        table_rows.append(row)
    return render_table(table_rows, keys, headers, fmts)


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
    row = {
        "n": res["n_stories"],
        "mean_violations": res["mean_violations"],
        "mean_words": res["mean_word_count"],
        "mean_sentences": res["mean_sentences"],
        "mean_grade": res["mean_grade_level"],
        "vendi": vendi,
        "lexdiv": lexdiv,
    }
    for name in CONSTRAINT_NAMES:
        row[f"pass_{name}"] = res["pass_rate"][name]
    return row


def live_table(constraints=(), diversity: bool = True):
    """(header_line, row_function) for the table printed as conditions finish.

    Summary columns only. The per-requirement breakdown is printed once at the
    end, where it can have a table of its own.
    """
    keys, heads, fmts, _ = build_columns(constraints, diversity)
    widths = {k: max(len(heads[k]) + 2, 7) for k in keys}
    widths["run"] = 38

    header = "".join(heads[k].ljust(widths[k]) if k == "run" else heads[k].rjust(widths[k])
                     for k in keys)

    def row(label: str, r: dict) -> str:
        out = [label[: widths["run"] - 1].ljust(widths["run"])]
        for k in keys[1:]:
            v = r.get(k, float("nan"))
            cell = "--" if isinstance(v, float) and v != v else fmts[k].format(v)
            out.append(cell.rjust(widths[k]))
        return "".join(out)

    return header, row


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
    ap.add_argument("--max-words", type=int, default=60)
    ap.add_argument("--present-ratio", type=float, default=0.8)
    ap.add_argument("--max-grade", type=float, default=3.0)
    ap.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL,
                    help=f"registry key or HF id. keys: {', '.join(EMBEDDING_MODELS)}. "
                         "use bge-m3 to match the published Arabic runs")
    ap.add_argument("--truncate-words", type=int, default=None,
                    help="cut every story to its first N words before scoring diversity, "
                         "so the score does not partly measure output length")
    ap.add_argument("--embedding-device", default="auto",
                    help="where to put the embedding model: 'auto' picks a GPU with "
                         "room and falls back to the CPU")
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
        constraints=list(args.constraints),
    )

    scorer = None
    if args.diversity:
        from noiseegra.creativity_metrics import CreativityScorer
        # Built once: constructing it downloads and loads the embedding model.
        scorer = CreativityScorer(
            ["placeholder text one", "placeholder text two"],
            embedding_model=args.embedding_model,
            truncate_words=args.truncate_words,
            device=args.embedding_device,
        )

    csvs = sorted(p for p in in_dir.glob("*.csv") if p.name != "prompts.json")
    if not csvs:
        raise SystemExit(f"no run CSVs in {in_dir}")

    rows, seen_stems = [], {}
    for path in csvs:
        stories, prompt_idx = read_run(path)
        if not stories:
            continue
        checker.to_csv(stories, out_dir / path.name, prompt_idx)
        label = label_run(path.stem)
        row = score_condition(stories, prompt_idx, checker, scorer, args.min_group)
        row["run"] = label.text
        row["_order"] = label.sort_key
        seen_stems[label.text] = path.stem
        rows.append(row)
        print(f"  scored {label.text}", flush=True)

    rows.sort(key=lambda r: r["_order"])
    shorts = [label_run(seen_stems[r["run"]]).short for r in rows]
    keys, headers, fmts, notes = build_columns(diversity=args.diversity)
    table = render_table(rows, keys, headers, fmts)
    reqs = requirement_table(rows, checker, shorts)

    print("\n" + "=" * max(len(line) for line in table))
    print("\n".join(table))
    print("\n  " + "\n  ".join(f"{headers[k]:<10}{notes[k]}" for k in keys if notes[k]))

    print("\n" + "=" * max(len(line) for line in reqs))
    print("how often each requirement was met")
    print("\n".join(reqs))
    print(f"\n  tense and names checked with the {checker.backend} backend.")

    all_keys, all_head, all_fmt, _ = build_columns(
        checker.constraints, args.diversity, per_constraint=True)
    md_rows = ["| " + " | ".join(all_head[k] for k in all_keys) + " |",
               "|" + "|".join("---" for _ in all_keys) + "|"]
    for r in rows:
        md_rows.append("| " + " | ".join(
            "--" if isinstance(r.get(k), float) and r[k] != r[k] else all_fmt[k].format(r[k])
            for k in all_keys) + " |")

    md = out_dir / "English_Constraint_Table.md"
    md.write_text(
        "# Constrained writing: requirement adherence and within-prompt diversity\n\n"
        "## What a story has to do to pass\n\n"
        + "".join(f"- {line}\n" for line in constraint_legend(checker))
        + f"\nTense and names checked with the `{checker.backend}` backend. Diversity is "
          "computed within a prompt group and averaged across groups.\n\n"
        + "\n".join(md_rows)
        + "\n\n## The other columns\n\n"
        + "".join(f"- **{headers[k]}** {notes[k]}\n" for k in keys if notes[k]),
        encoding="utf-8",
    )
    with (out_dir / "English_Constraint_Table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=all_keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in all_keys})

    print(f"\nwrote {md}")


if __name__ == "__main__":
    main()
