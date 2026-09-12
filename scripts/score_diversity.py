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

    # report coherence, and score diversity only over the stories that pass
    python scripts/score_diversity.py --input-dir ... --drop-incoherent \\
        --coherence-model Qwen/Qwen2.5-0.5B --coherence-reference "steer only"

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
Coh             fraction of stories that pass the coherence checks (--coherence).

Diversity over broken text is not diversity: a story that degenerates embeds far
from its neighbours and every spread-based score rewards that. --coherence adds a
pass-rate column; --drop-incoherent also removes the failures from the diversity
scores. Read the drop rates before the diversity columns -- conditions that lose
different numbers of stories are no longer a like-for-like comparison, and the
pass rate is then the more honest headline.
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
    group_by_prompt as _group_by_prompt,
    calibrate_threshold,
    distinct_k,
    group_by_prompt,
    read_run_csv,
    vendi_from_embeddings,
    word_count,
)
from noiseegra.embeddings import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS, describe, resolve_embedding_model,
)
from noiseegra.run_labels import label_from_run_id, label_run, plan_summary  # noqa: E402

# key, header, format, one-line explanation printed under the table
COLUMNS = [
    ("run", "condition", "{}", ""),
    ("n", "stories", "{}", "how many stories went into the scores on this row"),
    ("groups", "prompts", "{}", "how many prompts they were spread over"),
    ("coherent", "coherent", "{:.0%}", "share of stories that passed the coherence checks"),
    ("mean_words", "words", "{:.0f}", "mean story length"),
    ("vendi_raw", "Vendi", "{:.2f}",
     "effective number of distinct stories per prompt, over the full text"),
    ("vendi_matched", "Vendi@N", "{:.2f}",
     "the same, after cutting every story to the same first N words"),
    ("distinct_mean", "distinct", "{:.2f}",
     "how many genuinely different stories are in each group, out of the group size"),
    ("distinct_frac", "of group", "{:.2f}",
     "the same, as a fraction of the group size -- read it only when the conditions "
     "kept the same number of stories"),
    ("distinct_rarefied", "distinct@m", "{:.2f}",
     "distinct stories when every condition is cut to the same group size, so a "
     "condition that lost stories is not flattered by having a smaller group"),
    ("plot_vendi", "plot Vendi", "{:.2f}",
     "Vendi over six-slot plot skeletons instead of prose: does the story differ, "
     "not just the wording"),
    ("plot_distinct", "plot distinct", "{:.2f}", "distinct classes over those skeletons"),
]

# Coherence reasons, spelled out for the table.
REASON_TEXT = {
    "too_short": "too short",
    "repetition": "repetition loop",
    "junk_chars": "junk characters",
    "nonlexical": "non-words",
    "no_function_words": "no function words",
    "run_on": "no sentence breaks",
    "dangling_end": "ends mid-sentence",
    "incoherent": "unrelated sentences",
    "high_perplexity": "high perplexity",
    "garbage_tail": "garbage at the end",
}


def rule(text: str = "", width: int = 78) -> str:
    return text.center(width, "=") if text else "=" * width


def render_table(rows, keys, headers, fmts, sep="  ", left=("run", "condition", "why")):
    """Plain aligned columns; text columns left-justified, numbers right."""
    cells = [[headers[k] for k in keys]]
    for r in rows:
        line = []
        for k in keys:
            v = r[k]
            line.append("--" if isinstance(v, float) and v != v else fmts[k].format(v))
        cells.append(line)
    widths = [max(len(row[i]) for row in cells) for i in range(len(keys))]
    out = []
    for n, row in enumerate(cells):
        parts = [c.ljust(w) if k in left else c.rjust(w)
                 for c, w, k in zip(row, widths, keys)]
        out.append(sep.join(parts).rstrip())
        if n == 0:
            out.append("-" * len(out[0]))
    return out


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





def _load_labels(paths) -> dict:
    """Map run id to a readable condition name.

    ``run_english_experiment.py`` writes ``live_scores.csv`` next to the story
    CSVs with a ``run,label`` pair per condition, but only once a whole sweep
    finishes, so a resumed or interrupted run has none. Anything that file does
    not cover is read back out of the run id itself.
    """
    labels: dict = {}
    for d in {p.parent for p in paths}:
        f = d / "live_scores.csv"
        if not f.is_file():
            continue
        with f.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("run") and row.get("label"):
                    labels[row["run"]] = row["label"]
    for p in paths:
        if p.stem not in labels:
            parsed = label_from_run_id(p.stem)
            if parsed:
                labels[p.stem] = parsed
    return labels


def _apply_coherence(runs, args, scorer=None):
    """Check every story, trim its tail, and optionally drop the failures.

    Returns the (possibly filtered) runs and a per-condition summary. Trimming
    happens whether or not stories are dropped, so the word counts and diversity
    scores downstream are over story text rather than trailing markup.
    """
    from noiseegra.coherence import (
        CoherenceFilter, CoherenceThresholds, PerplexityScorer,
        apply_sentence_coherence, nll_reference, sentence_coherence, summarise,
    )

    thresholds = CoherenceThresholds(
        min_words=args.coherence_min_words,
        max_ppl_z=args.coherence_ppl_z,
        max_tail_nll_gap=args.coherence_tail_gap,
    )
    ppl = None
    if args.coherence_model:
        print(f"loading {args.coherence_model} for the perplexity checks ...", flush=True)
        ppl = PerplexityScorer(args.coherence_model)
    filt = CoherenceFilter(thresholds, trim=args.trim_tail, ppl_scorer=ppl)

    reports = {name: [filt.check(t) for t in stories] for name, stories, _ in runs}

    def pick_reference(values_by_run, label):
        """Judge every condition against a shared reference, not against its own
        median: a condition where most stories are broken has a broken median and
        nothing in it stands out.

        Default is the pooled distribution over all conditions, which works as long
        as most stories overall are fine. When a clean condition exists, naming it
        with --coherence-reference is stricter and better -- and note that picking
        the "most coherent" condition automatically would not work, because a
        condition stuck in a repetition loop has near-perfect sentence-to-sentence
        similarity.
        """
        if args.coherence_reference:
            picks = [n for n in values_by_run
                     if args.coherence_reference.lower() in n.lower()]
            if not picks:
                raise SystemExit(
                    f"--coherence-reference {args.coherence_reference!r} matches no "
                    f"condition. Available: {', '.join(sorted(values_by_run))}"
                )
            name = (picks[0] if len(picks) == 1
                    else f"{len(picks)} conditions matching {args.coherence_reference!r}")
            pool = [v for n in picks for v in values_by_run[n]]
        else:
            name = f"all {len(values_by_run)} conditions pooled"
            pool = [v for vs in values_by_run.values() for v in vs]
        ref = nll_reference(pool)
        print(f"  {label} judged against: {name} (median {ref[0]:.3f}, "
              f"spread {ref[1]:.3f})")
        if ref[1] <= 1e-9:
            print(f"  WARNING: the {label} reference has no spread, so that check is "
                  "disabled.")
        return ref

    if scorer is not None:
        print("  sentence-to-sentence coherence ...", flush=True)
        sims = {name: sentence_coherence(
                    [r.text or " " for r in reports[name]],
                    lambda t: scorer.encode(t, truncate=None))
                for name, _, _ in runs}
        ref = pick_reference({n: [v for v in vs if v == v] for n, vs in sims.items()},
                             "coherence")
        for name, _, _ in runs:
            apply_sentence_coherence(reports[name], sims[name], thresholds, ref)

    if ppl is not None:
        triples = {}
        for name, _, _ in runs:
            print(f"  perplexity: {name}", flush=True)
            triples[name] = ppl.score([r.text or " " for r in reports[name]],
                                      thresholds.tail_fraction)
        reference = pick_reference({n: [o for o, _, _ in v] for n, v in triples.items()},
                                   "perplexity")
        for name, _, _ in runs:
            filt.apply_perplexity(reports[name], triples[name], reference)

    out_runs, summary = [], {}
    for name, stories, pidx in runs:
        reps = reports[name]
        summary[name] = summarise(reps)
        summary[name]["trimmed"] = float(sum(1 for r in reps if r.trimmed_words > 0))
        summary[name]["trimmed_words"] = float(sum(r.trimmed_words for r in reps))
        keep = [(r.text, p) for r, p in zip(reps, pidx)
                if r.ok or not args.drop_incoherent]
        if keep:
            texts, ps = zip(*keep)
            out_runs.append((name, list(texts), list(ps)))
        else:
            print(f"  [warn] {name}: every story failed the coherence checks, "
                  "dropping the condition")

    skip = ("n", "kept", "pass_rate", "trimmed", "trimmed_words")
    rows = []
    for name, v in summary.items():
        why = ", ".join(f"{REASON_TEXT.get(k, k)} {int(v[k])}"
                        for k in sorted(v, key=lambda k: -v[k]) if k not in skip)
        rows.append({"condition": name, "stories": int(v["n"]), "kept": v["pass_rate"],
                     "trimmed": int(v["trimmed"]), "why": why or "-"})
    keys = ["condition", "stories", "kept", "trimmed", "why"]
    heads = {"condition": "condition", "stories": "stories", "kept": "kept",
             "trimmed": "tails trimmed", "why": "why the rest were rejected"}
    fmts = {"condition": "{}", "stories": "{}", "kept": "{:.0%}", "trimmed": "{}",
            "why": "{}"}
    print("\n" + rule(" coherence "))
    for line in render_table(rows, keys, heads, fmts):
        print("  " + line)
    print("    kept                       stories that passed every check")
    print("    tails trimmed              stories whose trailing markup or "
          "commentary was stripped, then kept")

    rates = [v["pass_rate"] for v in summary.values()]
    if args.drop_incoherent and rates and (max(rates) - min(rates)) > 0.1:
        print(f"\n  WARNING: pass rates range from {min(rates):.0%} to {max(rates):.0%}.")
        print("  Dropping stories leaves the conditions with different sample sizes and")
        print("  different selection, so the diversity table below is not a like-for-like")
        print("  comparison. The pass rate itself is the more honest headline here.")
    print()
    return out_runs, summary


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
    print(f"{indent}Stories answering the same prompt are {np.median(w):.3f} alike on "
          f"average (90th percentile {np.percentile(w, 90):.3f}, most alike "
          f"{w.max():.3f}).")
    print(f"{indent}{merged:.1%} of them sit above the cut-off and will be merged.")
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
    ap.add_argument("--rarefy", default="auto",
                    help="group size every condition is subsampled to for the "
                         "distinct@m column: 'auto' (the smallest group any condition "
                         "has left), an integer, or 'off'")
    ap.add_argument("--rarefy-draws", type=int, default=60,
                    help="subsamples averaged for distinct@m")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto",
                    help="where to put the embedding model: 'auto' picks a GPU with "
                         "room and falls back to the CPU")
    # coherence
    ap.add_argument("--coherence", action="store_true",
                    help="check every story for repetition loops, junk characters, "
                         "collapsed function-word ratio, run-on text and trailing "
                         "markup; report the pass rate as a column")
    ap.add_argument("--drop-incoherent", action="store_true",
                    help="also exclude the failing stories from the diversity scores. "
                         "implies --coherence. the drop rate per condition is printed: "
                         "read it, because conditions that lose different numbers of "
                         "stories are no longer being compared on equal footing")
    ap.add_argument("--no-trim-tail", dest="trim_tail", action="store_false",
                    help="keep trailing markup and post-story commentary instead of "
                         "stripping it. trimming is on by default because it saves a "
                         "story that would otherwise be dropped, which avoids "
                         "selection bias")
    ap.add_argument("--coherence-model", default=None,
                    help="small LM for the perplexity checks, e.g. Qwen/Qwen2.5-0.5B. "
                         "off by default; the heuristics need no model")
    ap.add_argument("--coherence-ppl-z", type=float, default=3.5,
                    help="reject a story whose perplexity is this many robust "
                         "deviations above its condition's median")
    ap.add_argument("--coherence-tail-gap", type=float, default=1.5,
                    help="reject a story whose last fifth is this many nats per token "
                         "more surprising than the rest (1.5 is about 4.5x perplexity)")
    ap.add_argument("--coherence-reference", default=None,
                    help="condition name (substring) whose perplexity distribution the "
                         "other conditions are judged against. default: whichever "
                         "condition the small model finds least surprising")
    ap.add_argument("--no-semantic-coherence", action="store_true",
                    help="skip the sentence-to-sentence coherence check. it reuses the "
                         "embedding model already loaded, so it is close to free, and it "
                         "is the only check that sees fluent word salad without a "
                         "second model")
    ap.add_argument("--coherence-min-words", type=int, default=15)
    # plot-level
    ap.add_argument("--plot", action="store_true", help="also score plot skeletons")
    ap.add_argument("--plot-backend", default="hf", choices=["hf", "spacy"])
    ap.add_argument("--plot-model", default="Qwen/Qwen2.5-7B-Instruct",
                    help="extraction model for --plot-backend hf")
    ap.add_argument("--plot-spacy-model", default="en_core_web_sm")
    ap.add_argument("--plot-batch-size", type=int, default=8)
    ap.add_argument("--plot-cache", help="default: <out-dir>/plot_skeletons.json")
    args = ap.parse_args()

    # A notebook runs this through a pipe, and a piped Python block-buffers
    # stdout, so a long run looks like it has hung until it finishes. Line
    # buffering makes progress visible without needing `python -u`.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:  # pragma: no cover
        pass

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

    labels = _load_labels(paths)
    runs, seen, order = [], {}, {}
    for path in paths:
        stories, pidx = read_run_csv(path)
        if len(stories) < args.min_group:
            continue
        name = labels.get(path.stem, path.stem)
        if name in seen:  # two conditions sharing a label: keep them apart
            name = f"{name} [{path.stem[-12:]}]"
        seen[name] = path.stem
        order[name] = label_run(path.stem).sort_key
        runs.append((name, stories, pidx))
    if not runs:
        raise SystemExit("no run CSVs with enough stories")
    runs.sort(key=lambda r: order[r[0]])

    print(rule())
    print(f"  {len(runs)} conditions, {sum(len(s) for _, s, _ in runs)} stories")
    for line in plan_summary([seen[n] for n, _, _ in runs]):
        print(f"  {line}")
    print(f"  embeddings: {resolve_embedding_model(args.embedding_model)}")
    print(rule() + "\n")

    # Checked before anything downloads a model: this run takes minutes to reach
    # the point where the reference is used, and failing there wastes all of it.
    if args.coherence_reference:
        matched = [n for n, _, _ in runs if args.coherence_reference.lower() in n.lower()]
        if not matched:
            raise SystemExit(
                f"--coherence-reference {args.coherence_reference!r} matches no "
                "condition. Available:\n  "
                + "\n  ".join(n for n, _, _ in runs)
            )
        print(f"coherence reference will be: {', '.join(matched)}\n")

    coherence: dict = {}

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
        print(f"Length matching: every story cut to its first {budget} words for the "
              f"Vendi@{budget} column.")
        print(f"  {budget} is the mean length of the shortest condition, {short}.\n")

    scorer = DiversityScorer(
        args.embedding_model, truncate_to=budget, device=args.device, batch_size=args.batch_size
    )

    if args.coherence or args.drop_incoherent:
        runs, coherence = _apply_coherence(
            runs, args, scorer=None if args.no_semantic_coherence else scorer
        )
        if not runs:
            raise SystemExit(
                "every condition lost all of its stories to the coherence checks. "
                "That is a threshold problem, not a result: look at the reason "
                "columns above and relax the check that fired, or run without "
                "--drop-incoherent to see the pass rates alone."
            )
        # Trimming changes word counts, so the length budget is recomputed.
        if budget and str(args.truncate_words).lower() == "auto":
            after = int(round(min(statistics.mean(word_count(s) for s in st)
                                  for _, st, _ in runs)))
            if after != budget:
                budget = after
                scorer.truncate_to = budget
                print(f"\n  Trimming changed the shortest condition, so the length-"
                      f"matched budget is now {budget} words.")
            print()

    # ---- distinct-k threshold, calibrated once and shared ------------------- #
    if str(args.distinct_threshold).lower() == "auto":
        threshold, basis = _calibrate(scorer, runs, args.distinct_percentile, budget)
        if math.isnan(threshold):
            print("cannot calibrate a distinct-k threshold from this input; "
                  "pass --distinct-threshold explicitly")
        else:
            print(rule(" when are two stories the same story "))
            print(f"  Two stories count as the same above cosine similarity "
                  f"{threshold:.4f}.")
            print(f"  That cut-off is the {args.distinct_percentile:.0f}th percentile of "
                  f"{basis} -- pairs\n  that are known to be different stories, so a pair "
                  "only merges if it is more\n  alike than almost every known-different "
                  "pair.")
            _threshold_diagnostics(scorer, runs, budget, threshold)
    else:
        threshold = float(args.distinct_threshold)
        print(f"Two stories count as the same above cosine similarity {threshold:.4f} "
              "(fixed).")
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
        print(rule(" plot skeletons "))
        print(f"  Reducing each story to setting, protagonist, goal, obstacle, turning "
              f"point\n  and resolution with {extractor.tag}, then measuring diversity "
              "over those.")
        for name, st, _ in runs:
            skels = extract_skeletons(st, extractor, cache)
            skeletons[name] = [skeleton_text(s) for s in skels]
            print(f"  {len(skels):>4} skeletons: {name}", flush=True)
        # Skeletons are short, uniform and share a field scaffold, so they sit in a
        # much narrower similarity range than prose. Reusing the prose cut-off here
        # would merge everything, so calibrate a second one on the skeletons.
        if str(args.distinct_threshold).lower() == "auto":
            plot_runs = [(n, skeletons[n], pi) for n, _, pi in runs]
            plot_threshold, plot_basis = _calibrate(
                scorer, plot_runs, args.distinct_percentile, None
            )
            if not math.isnan(plot_threshold):
                print(f"  Two skeletons count as the same above similarity "
                      f"{plot_threshold:.4f}, calibrated on {plot_basis}.")
                _threshold_diagnostics(scorer, plot_runs, None, plot_threshold,
                                       indent="  plot ")
        else:
            plot_threshold = threshold
        print()

    # ---- rarefaction size, shared by every condition ------------------------ #
    sizes = [len(ix) for _, st, pi in runs
             for ix in _group_by_prompt(st, pi, args.min_group)]
    if str(args.rarefy).lower() in ("off", "none", "0"):
        rarefy_to = None
    elif str(args.rarefy).lower() == "auto":
        rarefy_to = max(2, min(sizes)) if sizes else None
    else:
        rarefy_to = int(args.rarefy)
    if rarefy_to:
        print(f"distinct@m compares every condition at m = {rarefy_to} stories per "
              "prompt,\n  the smallest any condition has left, averaged over "
              f"{args.rarefy_draws} subsamples.\n")

    # ---- score every condition ---------------------------------------------- #
    print(rule(" scoring "))
    rows = []
    for name, stories, pidx in runs:
        res = scorer.score(stories, pidx, threshold=threshold, min_group=args.min_group,
                           rarefy_to=rarefy_to, draws=args.rarefy_draws)
        row = {
            "run": name, "n": res.n, "groups": res.n_groups,
            "coherent": coherence.get(name, {}).get("pass_rate", float("nan")),
            "mean_words": res.mean_words,
            "vendi_raw": res.vendi_raw, "vendi_matched": res.vendi_matched,
            "distinct_mean": res.distinct_mean, "distinct_frac": res.distinct_frac,
            "distinct_rarefied": res.distinct_rarefied,
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
        print(f"  scored {name}", flush=True)

    # already sorted by family and magnitude when the runs were read

    # ---- length-sensitivity check ------------------------------------------- #
    w = [r["mean_words"] for r in rows]
    checks = [(label, pearson(w, [r[k] for r in rows])) for label, k in [
        ("Vendi", "vendi_raw"), (f"Vendi@{budget}" if budget else "Vendi@N", "vendi_matched"),
        ("distinct", "distinct_mean"), ("of group", "distinct_frac"),
        (f"distinct@{rarefy_to}" if rarefy_to else "distinct@m", "distinct_rarefied"),
        ("plot Vendi", "plot_vendi"), ("plot distinct", "plot_distinct"),
    ]]
    checks = [(l, v) for l, v in checks if not math.isnan(v)]

    keys = [k for k, _, _, _ in COLUMNS
            if not (k in ("plot_vendi", "plot_distinct") and not skeletons)
            and not (k == "coherent" and not coherence)
            and not (k == "distinct_rarefied" and not rarefy_to)]
    headers = {}
    for k, h, _, _ in COLUMNS:
        if k == "vendi_matched" and budget:
            h = f"Vendi@{budget}"
        elif k == "distinct_rarefied" and rarefy_to:
            h = f"distinct@{rarefy_to}"
        headers[k] = h
    fmts = {k: f for k, _, f, _ in COLUMNS}
    notes = {k: n for k, _, _, n in COLUMNS}

    table = render_table(rows, keys, headers, fmts)
    legend = [f"  {headers[k]:<14}{notes[k]}" for k in keys if notes[k]]

    print("\n" + rule(" results "))
    print("\n".join(table))
    print("\nwhat the columns mean")
    print("\n".join(legend))

    print("\nDoes each score still just track story length?")
    print("  Correlation across conditions between mean word count and each column.")
    print("  A column that still tracks length is still reporting length.")
    for label, v in checks:
        print(f"    {label:<14}{v:+.3f}")

    # ---- files -------------------------------------------------------------- #
    def md_row(vals):
        return "| " + " | ".join(vals) + " |"

    md_lines = [md_row(headers[k] for k in keys),
                "|" + "|".join("---" for _ in keys) + "|"]
    for r in rows:
        md_lines.append(md_row(
            "--" if isinstance(r[k], float) and r[k] != r[k] else fmts[k].format(r[k])
            for k in keys))

    header_lines = plan_summary([seen[n] for n, _, _ in runs])
    md = out_dir / "Diversity_Table.md"
    md.write_text(
        "# Diversity, length-matched\n\n"
        + "".join(f"- {line}\n" for line in header_lines)
        + f"- embeddings: {describe(args.embedding_model)}\n"
        + (f"- every story cut to its first {budget} words for the Vendi@{budget} "
           "column\n" if budget else "")
        + f"- two stories count as the same when their similarity exceeds "
          f"{threshold:.4f}\n"
        + "- every score is computed within a prompt group and then averaged over "
          "groups\n\n"
        + "\n".join(md_lines)
        + "\n\n## What the columns mean\n\n"
        + "".join(f"- **{headers[k]}** {notes[k]}\n" for k in keys if notes[k])
        + "\n## Does each score still just track story length?\n\n"
          "Correlation across conditions between mean word count and each column. "
          "A column that still tracks length is still reporting length.\n\n"
        + "| column | r with mean word count |\n|---|---:|\n"
        + "".join(f"| {l} | {v:+.3f} |\n" for l, v in checks),
        encoding="utf-8",
    )
    with (out_dir / "Diversity_Table.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r[k] for k in keys})

    print(f"\nwrote {md}")


if __name__ == "__main__":
    main()
