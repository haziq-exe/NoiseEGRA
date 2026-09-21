#!/usr/bin/env python
"""A workbook of stories to read, with the metrics on a sheet of their own.

    python scripts/build_story_review.py \
        --condition "champion=r19-band6-14:g0p15orth__obstory__opre__ponly__bud3" \
        --condition "baseline=r19-band6-14:BASELINE" \
        --csv "high temp=experiments/r17-headline/state/Qwen3-1.7B/Qwen3-1.7B__BASELINE__temp1p8__topp0p95.csv" \
        --out story_review.xlsx

A table of scores says how often a rule is broken and nothing about whether the
writing is any good, and three separate results in this project were wrong until
somebody read the stories. This produces the thing to read: one sheet per
condition, every story in the order generated, with the flags the coherence
checks raised beside it so the filter can be audited as well as the method.

``--condition NAME=RUN:FRAGMENT`` takes stories from a run's checkpoint, which is
the deduplicated source (a resumed sharded run's merged CSV repeats its history
once per shard). FRAGMENT is any substring that picks out one run id.
``--csv NAME=PATH`` takes them from a file instead.

Diversity here follows the same order as everywhere else in the project: drop the
stories the coherence checks reject, cut each story to its first 40 words, trim
the ones sitting far from the rest of their own set, and pool every condition at
the same size. The embedding-based score is not computed -- it needs a GPU -- so
``--embedding-vendi NAME=VALUE`` carries it in from ``scripts/clean_vendi.py``.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from noiseegra.coherence import CoherenceFilter  # noqa: E402
from noiseegra.constraint_metrics_en import (  # noqa: E402
    MIDDLE_CONSTRAINTS, MONOTONE_CONSTRAINTS, WHOLE_STORY_CONSTRAINTS,
    EnglishConstraintChecker,
)
from noiseegra.diversity import read_run_csv  # noqa: E402
from noiseegra.readability import uncommon_word_share  # noqa: E402
from noiseegra.structure import (  # noqa: E402
    _set_vectors, content_lemma_sets, pos_trigram_vectors, rarefied_vendi,
    trim_isolated,
)

RULE_TEXT = {
    "present_tense": "present tense", "simple_register": "grade-2.5 reading level",
    "short_words": "no word over 2 syllables", "easy_opening": "first sentence 5 words or fewer",
    "short_sentences": "every sentence 8 words or fewer", "dialogue_min": "3 or more lines of speech",
    "varied_openers": "no word begins more than 3 sentences", "plain_words": "at most 2 adverbs",
    "sensory": "2 or more sensory words", "simple_syntax": "at most 1 subordinate clause",
    "fresh_words": "no word reused too often", "named_character": "a name used 3 or more times",
    "no_repetition": "no 5-word run repeated", "distinct_sentences": "no sentence written twice",
    "fresh_openings": "no two words begin more than 3 sentences",
    "mature_register": "sentences with some length and range",
    "story_format": "no title or heading, sentences keep their capitals",
}



def _checkpoints(run: str) -> list:
    """The checkpoint files for one run, wherever the download put them.

    A Kaggle pull leaves them under ``state/``; a Lightning pull leaves each shard's
    under ``shard<i>/``. Never both: a Kaggle run's second copy under ``output/``
    would count every story twice.
    """
    found = glob.glob(f"experiments/{run}/state/**/state.json", recursive=True)
    return found or sorted(glob.glob(f"experiments/{run}/shard*/**/state.json",
                                     recursive=True))

def from_state(run: str, fragment: str):
    """Stories for one run id, read from the checkpoint rather than the CSVs."""
    runs: dict = {}
    for f in _checkpoints(run):
        for rid, cells in json.load(open(f)).get("runs", {}).items():
            if fragment in rid:
                runs.setdefault(rid, {}).update(cells)
    if not runs:
        raise SystemExit(f"no run id in {run} contains {fragment!r}")
    if len(runs) > 1:
        rid = sorted(runs, key=len)[0]
        print(f"  note: {fragment!r} matched {len(runs)} run ids; taking {rid[-50:]}")
    else:
        rid = next(iter(runs))
    cells = runs[rid]
    return rid, [cells[k] for k in sorted(cells, key=lambda x: int(x.split(":")[1]))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", action="append", default=[], metavar="NAME=RUN:FRAG")
    ap.add_argument("--csv", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--embedding-vendi", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--constraint-set", choices=("monotone", "middle", "whole"),
                    default="monotone")
    ap.add_argument("--truncate-words", type=int, default=40)
    ap.add_argument("--min-grade", type=float, default=3.0)
    ap.add_argument("--max-word-uses", type=int, default=5)
    ap.add_argument("--max-opener-uses", type=int, default=5)
    ap.add_argument("--max-adverbs", type=int, default=5)
    ap.add_argument("--min-sensory", type=int, default=6)
    ap.add_argument("--present-ratio", type=float, default=0.9)
    ap.add_argument("--out", default="story_review.xlsx")
    args = ap.parse_args()

    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    rules = {"middle": MIDDLE_CONSTRAINTS, "monotone": MONOTONE_CONSTRAINTS,
             "whole": WHOLE_STORY_CONSTRAINTS}[args.constraint_set]
    ck = EnglishConstraintChecker(backend="spacy", constraints=rules,
                                  max_opener_uses=args.max_opener_uses,
                                  max_word_uses=args.max_word_uses,
                                  max_adverbs=args.max_adverbs,
                                  min_sensory=args.min_sensory,
                                  present_ratio_threshold=args.present_ratio,
                                  min_grade_level=args.min_grade, max_words=200)
    filt = CoherenceFilter()
    emb = dict(kv.split("=", 1) for kv in args.embedding_vendi)

    conds = []
    for spec in args.condition:
        name, rest = spec.split("=", 1)
        run, frag = rest.split(":", 1)
        rid, texts = from_state(run, frag)
        conds.append((name, texts, f"run {run} · {rid}"))
    for spec in args.csv:
        name, path = spec.split("=", 1)
        conds.append((name, read_run_csv(path)[0], path))
    if not conds:
        raise SystemExit("give at least one --condition or --csv")

    HDR_FILL = PatternFill("solid", fgColor="1F3864")
    HDR_FONT = Font(bold=True, color="FFFFFF")
    BAD = PatternFill("solid", fgColor="FCE4E4")
    WRAP = Alignment(wrap_text=True, vertical="top")
    TOP = Alignment(vertical="top")

    def header(ws, row, heads, widths):
        for i, (h, w) in enumerate(zip(heads, widths), start=1):
            c = ws.cell(row=row, column=i, value=h)
            c.fill, c.font = HDR_FILL, HDR_FONT
            c.alignment = Alignment(wrap_text=True, vertical="center")
            ws.column_dimensions[get_column_letter(i)].width = w
        ws.row_dimensions[row].height = 30

    scored, rows_by = {}, {}
    for name, texts, src in conds:
        reps = [filt.check(t) for t in texts]
        kept = [r.text for r in reps if r.ok]
        detail = ck.evaluate_all([r.text or t for t, r in zip(texts, reps)])["stories"]
        agg = ck.evaluate_all(kept)
        sv = trim_isolated(_set_vectors(content_lemma_sets(kept, args.truncate_words)))
        pv = trim_isolated(pos_trigram_vectors(kept, args.truncate_words))
        scored[name] = dict(n=len(texts), kept=len(kept), src=src, sv=sv, pv=pv,
                            broken=agg["mean_violations"], grade=agg["mean_grade_level"],
                            words=agg["mean_word_count"],
                            uncommon=float(np.mean(uncommon_word_share(kept))))
        rows_by[name] = [
            [i, "yes" if r.ok else "NO", "" if r.ok else r.reason, d.violations,
             "; ".join(RULE_TEXT.get(k, k) for k in rules if not d.checks.get(k, True)),
             d.word_count, round(d.grade_level, 1), " ".join((r.text or texts[i]).split())]
            for i, (r, d) in enumerate(zip(reps, detail))]

    pool = min(v["kept"] for v in scored.values())
    for v in scored.values():
        v["happens"] = rarefied_vendi(v["sv"], pool, draws=40)[0]
        v["wording"] = rarefied_vendi(v["pv"], pool, draws=40)[0]

    wb = Workbook()
    s = wb.active
    s.title = "summary"
    s["A1"] = "Story review"
    s["A1"].font = Font(bold=True, size=14)
    s["A2"] = (f"{len(conds)} conditions · {len(rules)} scored rules · variety over the first "
               f"{args.truncate_words} words of coherent stories, isolated stories trimmed, "
               f"every condition pooled at {pool}")
    heads = ["condition", "stories", "coherent", "rules broken", "variety of what happens",
             "variety of wording", "embedding Vendi", "reading grade", "uncommon words", "mean words"]
    header(s, 4, heads, [16, 9, 11, 13, 16, 14, 14, 12, 13, 11])
    for r, (name, *_x) in enumerate(conds, start=5):
        v = scored[name]
        vals = [name, v["n"], f'{v["kept"]} ({v["kept"]/v["n"]:.0%})', round(v["broken"], 2),
                round(v["happens"], 1), round(v["wording"], 1), emb.get(name, ""),
                round(v["grade"], 1), f'{v["uncommon"]:.1%}', round(v["words"])]
        for c, val in enumerate(vals, start=1):
            cell = s.cell(row=r, column=c, value=val)
            cell.alignment = TOP
            if c == 1:
                cell.font = Font(bold=True)
    s.freeze_panes = "A5"

    STORY_HEADS = ["story #", "coherent", "why rejected", "rules broken",
                   "which rules it broke", "words", "reading grade", "story"]
    for name, _t, src in conds:
        ws = wb.create_sheet(name[:31])
        ws["A1"] = name
        ws["A1"].font = Font(bold=True, size=14)
        ws["A2"] = src
        ws["A2"].font = Font(italic=True, size=10)
        ws["A4"] = ("Rows marked NO were rejected by the coherence checks and left out of the "
                    "scores; read them to audit the filter as well as the method.")
        ws["A4"].alignment = WRAP
        ws.merge_cells("A4:H4")
        header(ws, 6, STORY_HEADS, [9, 10, 26, 13, 46, 8, 10, 125])
        for i, row in enumerate(rows_by[name], start=7):
            bad = row[1] == "NO"
            for c, val in enumerate(row, start=1):
                cell = ws.cell(row=i, column=c, value=val)
                cell.alignment = WRAP if c in (3, 5, 8) else TOP
                if bad:
                    cell.fill = BAD
        ws.auto_filter.ref = f"A6:H{6 + len(rows_by[name])}"
        ws.freeze_panes = "A7"

    wb.save(args.out)
    print(f"wrote {args.out}")
    for name, *_x in conds:
        v = scored[name]
        print(f"  {name:<14} {v['kept']:>3}/{v['n']:<4} coherent  broken {v['broken']:.2f}  "
              f"happens {v['happens']:.1f}  wording {v['wording']:.1f}")


if __name__ == "__main__":
    main()
