#!/usr/bin/env python
"""Length-matched Vendi, distinct-k, and plot-level diversity for existing runs.

Takes story CSVs that have already been generated and produces diversity scores.
It does not generate anything.

    # English runs from scripts/run_english_experiment.py
    python scripts/score_diversity.py --input-dir /kaggle/working/english/Granite-3.1-8B

    # add plot-level diversity (needs a GPU for the extraction model)
    python scripts/score_diversity.py --input-dir ... --plot

    # reproduce the published Arabic numbers' embedding model
    python scripts/score_diversity.py --input-dir experiment_results/ResidNoise \\
        --embedding-model bge-m3

Three metrics, all computed within a prompt group and then averaged:

Vendi           the existing score, on the full stories. Reported for continuity,
                and because comparing it against the next column is the point.
Vendi@N         the same score after cutting every story to its first N words.
                Longer stories embed further apart whatever they say, so a gain
                that vanishes here was a length artefact.
Distinct        how many genuinely different stories are in each group of k, from
                threshold clustering over embeddings. Bounded by the group size,
                so degraded text cannot inflate it the way spread-based scores
                allow.
PlotVendi       Vendi over six-slot plot skeletons instead of prose (--plot).
PlotDist        Distinct over the same skeletons.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from noiseegra.diversity import (  # noqa: E402
    DiversityScorer,
    calibrate_threshold,
    distinct_k,
    group_by_prompt,
    read_run_csv,
    vendi_from_embeddings,
    word_count,
)
from noiseegra.embeddings import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS, describe  # noqa: E402

COLUMNS = [
    ("run", "Run", "{}"),
    ("n", "N", "{}"),
    ("groups", "Grp", "{}"),
    ("mean_words", "Words", "{:.0f}"),
    ("vendi_raw", "Vendi", "{:.2f}"),
    ("vendi_matched", "Vendi@N", "{:.2f}"),
    ("distinct_mean", "Distinct", "{:.2f}"),
    ("distinct_frac", "Dist/k", "{:.2f}"),
    ("plot_vendi", "PlotVendi", "{:.2f}"),
    ("plot_distinct", "PlotDist", "{:.2f}"),
]


def pearson(a, b):
    a = [x for x in a]
    b = [x for x in b]
    pairs = [(x, y) for x, y in zip(a, b) if not (math.isnan(x) or math.isnan(y))]
    if len(pairs) < 3:
        return float("nan")
    xs, ys = zip(*pairs)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else float("nan")




def _calibrate(scorer, runs, percentile, truncate):
    """Similarity cut-off above which two texts count as the same story.

    Reference pairs must be known-different stories. Within one condition, two
    stories written from different prompts qualify by construction. Only when no
    condition has more than one prompt -- the Arabic runs, where every story
    answers the same prompt -- does this fall back to cross-condition pairs,
    which are merely *probably* different.

    ``runs`` is a list of ``(name, texts, prompt_indices)``.
    """
    multi = [r for r in runs if len(set(r[2])) > 1]
    if multi:
        cross = []
        for _, texts, pidx in multi:
            e = scorer.encode(texts, truncate=truncate)
            pa = np.asarray(pidx)
            sim = e @ e.T
            cross.append(sim[pa[:, None] != pa[None, :]])
        pooled = np.concatenate(cross)
        if pooled.size:
            return float(np.percentile(pooled, percentile)), \
                "cross-prompt pairs within a condition"
    texts, prompts = [], []
    for i, (_, t, pi) in enumerate(runs):
        texts += list(t)
        prompts += [p + 10_000 * i for p in pi]
    emb = scorer.encode(texts, truncate=truncate)
    return (calibrate_threshold(emb, prompts, percentile),
            "cross-condition pairs (approximate: no condition has >1 prompt)")


def _threshold_diagnostics(scorer, runs, budget, threshold, indent="  ") -> None:
    """Show where the cut-off sits relative to actual within-prompt similarity.

    Two failure modes to catch: a cut-off above almost every within-prompt pair
    means nothing ever merges and distinct-k just returns the group size; a
    cut-off below most of them means everything merges and it always returns 1.
    """
    within = []
    for _, stories, pidx in runs:
        emb = scorer.encode(stories, truncate=budget)
        for ix in group_by_prompt(stories, pidx, 2):
            sim = emb[ix] @ emb[ix].T
            iu = np.triu_indices(len(ix), k=1)
            within.extend(sim[iu].tolist())
    if not within:
        return
    w = np.asarray(within)
    merged = float((w >= threshold).mean())
    print(f"{indent}within-prompt similarity: median {np.median(w):.3f}, "
          f"90th {np.percentile(w, 90):.3f}, max {w.max():.3f}")
    print(f"{indent}{merged:.1%} of within-prompt pairs sit above the cut-off")
    if merged < 0.005:
        print(f"{indent}WARNING: almost nothing merges, so distinct-k will just report "
              "the group size. Lower --distinct-percentile or set --distinct-threshold.")
    elif merged > 0.8:
        print(f"{indent}WARNING: almost everything merges, so distinct-k will sit near "
              "1. Raise --distinct-percentile or set --distinct-threshold.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input-dir", help="directory of run CSVs")
    ap.add_argument("--inputs", nargs="*", default=[], help="explicit run CSVs")
    ap.add_argument("--out-dir", help="default: <input-dir>/DIVERSITY")
    ap.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL,
                    help=f"registry key or HF id. keys: {', '.join(EMBEDDING_MODELS)}. "
                         "use bge-m3 to match the published Arabic runs")
    ap.add_argument("--truncate-words", default="auto",
                    help="word budget for Vendi@N: 'auto' (shortest condition's mean), "
                         "an integer, or 'none' to disable")
    ap.add_argument("--distinct-threshold", default="auto",
                    help="cosine similarity above which two stories count as the same: "
                         "'auto' (calibrated from cross-prompt pairs) or a float")
    ap.add_argument("--distinct-percentile", type=float, default=99.0,
                    help="percentile of the cross-prompt similarity distribution used "
                         "by --distinct-threshold auto")
    ap.add_argument("--min-group", type=int, default=2,
                    help="prompt groups smaller than this are skipped")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    # plot-level
    ap.add_argument("--plot", action="store_true", help="also score plot skeletons")
    ap.add_argument("--plot-backend", default="hf", choices=["hf", "spacy"])
    ap.add_argument("--plot-model", default="Qwen/Qwen2.5-7B-Instruct",
                    help="extraction model for --plot-backend hf")
    ap.add_argument("--plot-spacy-model", default="en_core_web_sm")
    ap.add_argument("--plot-batch-size", type=int, default=8)
    ap.add_argument("--plot-cache", help="default: <out-dir>/plot_skeletons.json")
    args = ap.parse_args()

    paths = [Path(p) for p in args.inputs]
    if args.input_dir:
        d = Path(args.input_dir)
        if not d.is_dir():
            raise SystemExit(f"no such directory: {d}")
        paths += sorted(p for p in d.glob("*.csv") if not p.name.startswith("live_scores"))
    if not paths:
        raise SystemExit("give --input-dir or --inputs")

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(args.input_dir) / "DIVERSITY" if args.input_dir else paths[0].parent / "DIVERSITY"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for p in paths:
        stories, pidx = read_run_csv(p)
        if len(stories) >= args.min_group:
            runs.append((p.stem, stories, pidx))
    if not runs:
        raise SystemExit("no run CSVs with enough stories")

    print(f"embedding model: {describe(args.embedding_model)}")
    print(f"{len(runs)} conditions, {sum(len(s) for _, s, _ in runs)} stories\n")

    # ---- word budget for the length-matched score --------------------------- #
    mean_words = {name: statistics.mean(word_count(s) for s in st) for name, st, _ in runs}
    if str(args.truncate_words).lower() in ("none", "0", "off"):
        budget = None
    elif str(args.truncate_words).lower() == "auto":
        budget = int(round(min(mean_words.values())))
    else:
        budget = int(args.truncate_words)
    if budget:
        short = min(mean_words, key=mean_words.get)
        print(f"length-matched at N = {budget} words (mean of the shortest condition, {short})")

    scorer = DiversityScorer(
        args.embedding_model, truncate_to=budget, device=args.device, batch_size=args.batch_size
    )

    # ---- distinct-k threshold, calibrated once and shared ------------------- #
    if str(args.distinct_threshold).lower() == "auto":
        threshold, basis = _calibrate(scorer, runs, args.distinct_percentile, budget)
        if math.isnan(threshold):
            print("cannot calibrate a distinct-k threshold from this input; "
                  "pass --distinct-threshold explicitly")
        else:
            print(f"calibrated on {basis}")
            print(f"distinct-k threshold = {threshold:.4f} "
                  f"({args.distinct_percentile:.0f}th percentile of cross-prompt similarity)")
            _threshold_diagnostics(scorer, runs, budget, threshold)
    else:
        threshold = float(args.distinct_threshold)
        print(f"distinct-k threshold = {threshold:.4f} (fixed)")
    print()

    # ---- plot skeletons ----------------------------------------------------- #
    skeletons = {}
    plot_threshold = float("nan")
    if args.plot:
        from noiseegra.plot_skeleton import (
            HFPlotExtractor, PlotCache, SpacyPlotExtractor, extract_skeletons, skeleton_text,
        )

        if args.plot_backend == "hf":
            extractor = HFPlotExtractor(args.plot_model, batch_size=args.plot_batch_size)
        else:
            extractor = SpacyPlotExtractor(args.plot_spacy_model)
        cache = PlotCache(Path(args.plot_cache) if args.plot_cache
                          else out_dir / "plot_skeletons.json")
        print(f"extracting plot skeletons with {extractor.tag}")
        for name, st, _ in runs:
            skels = extract_skeletons(st, extractor, cache)
            skeletons[name] = [skeleton_text(s) for s in skels]
            print(f"  [ok] {name}: {len(skels)} skeletons")
        # Skeletons are short, uniform and share a field scaffold, so they sit in a
        # much narrower similarity range than prose. Reusing the prose cut-off here
        # would merge everything, so calibrate a second one on the skeletons.
        if str(args.distinct_threshold).lower() == "auto":
            plot_runs = [(n, skeletons[n], pi) for n, _, pi in runs]
            plot_threshold, plot_basis = _calibrate(
                scorer, plot_runs, args.distinct_percentile, None
            )
            if not math.isnan(plot_threshold):
                print(f"plot distinct-k threshold = {plot_threshold:.4f} "
                      f"(calibrated on {plot_basis})")
                _threshold_diagnostics(scorer, plot_runs, None, plot_threshold,
                                       indent="  plot ")
        else:
            plot_threshold = threshold
        print()

    # ---- score every condition ---------------------------------------------- #
    rows = []
    for name, stories, pidx in runs:
        res = scorer.score(stories, pidx, threshold=threshold, min_group=args.min_group)
        row = {
            "run": name, "n": res.n, "groups": res.n_groups,
            "mean_words": res.mean_words,
            "vendi_raw": res.vendi_raw, "vendi_matched": res.vendi_matched,
            "distinct_mean": res.distinct_mean, "distinct_frac": res.distinct_frac,
            "plot_vendi": float("nan"), "plot_distinct": float("nan"),
        }
        if name in skeletons:
            # Skeletons are already a fixed length, so no truncation here.
            emb = scorer.encode(skeletons[name], truncate=None)
            groups = group_by_prompt(stories, pidx, args.min_group)
            vs = [vendi_from_embeddings(emb[ix]) for ix in groups]
            row["plot_vendi"] = float(np.mean(vs)) if vs else float("nan")
            if not math.isnan(plot_threshold):
                ds = [distinct_k(emb[ix], plot_threshold) for ix in groups]
                row["plot_distinct"] = float(np.mean(ds)) if ds else float("nan")
        rows.append(row)
        print(f"[ok] {name}: words={row['mean_words']:.0f} vendi={row['vendi_raw']:.2f} "
              f"vendi@N={row['vendi_matched']:.2f} distinct={row['distinct_mean']:.2f}")

    rows.sort(key=lambda r: r["run"])

    # ---- length-sensitivity check ------------------------------------------- #
    w = [r["mean_words"] for r in rows]
    checks = [(label, pearson(w, [r[k] for r in rows])) for label, k in [
        ("Vendi", "vendi_raw"), ("Vendi@N", "vendi_matched"),
        ("Distinct", "distinct_mean"), ("Dist/k", "distinct_frac"),
        ("PlotVendi", "plot_vendi"), ("PlotDist", "plot_distinct"),
    ]]
    checks = [(l, v) for l, v in checks if not math.isnan(v)]

    keys = [k for k, _, _ in COLUMNS if not (
        k in ("plot_vendi", "plot_distinct") and not skeletons)]
    headers = {k: h for k, h, _ in COLUMNS}
    fmts = {k: f for k, _, f in COLUMNS}

    def cell(row, key):
        v = row[key]
        return "--" if isinstance(v, float) and v != v else fmts[key].format(v)

    lines = ["| " + " | ".join(headers[k] for k in keys) + " |",
             "|" + "|".join("---" for _ in keys) + "|"]
    lines += ["| " + " | ".join(cell(r, k) for k in keys) + " |" for r in rows]

    check_lines = ["| Metric | r with mean word count |", "|---|---:|"]
    check_lines += [f"| {l} | {v:+.3f} |" for l, v in checks]

    md = out_dir / "Diversity_Table.md"
    md.write_text(
        "# Length-matched diversity\n\n"
        f"Embedding model: `{describe(args.embedding_model)}`. "
        f"Length-matched column truncates every story to its first {budget} words. "
        f"Distinct-k merges stories above cosine similarity {threshold:.4f}. "
        "All scores are computed within a prompt group and averaged across groups.\n\n"
        + "\n".join(lines)
        + "\n\n## Length sensitivity\n\nCorrelation across conditions between mean "
          "word count and each metric. A metric that still tracks length is still "
          "reporting length.\n\n"
        + "\n".join(check_lines) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "Diversity_Table.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r[k] for k in keys})

    print("\n" + "\n".join(lines))
    print("\nlength sensitivity (r with mean word count across conditions):")
    for label, v in checks:
        print(f"  {label:<10} {v:+.3f}")
    print(f"\nwrote {md}")


if __name__ == "__main__":
    main()
