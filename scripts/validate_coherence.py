#!/usr/bin/env python
"""Check the coherence filter against the paper's own modal-collapse labels.

``noiseegra/data/paper_modal_collapse_indices.json`` records, per published run,
which stories were judged totally collapsed. That is a hand-labelled ground truth
for "incoherent", so it can be used to measure the filter rather than assert it.

    python scripts/validate_coherence.py                       # heuristics only
    python scripts/validate_coherence.py --coherence-model Qwen/Qwen2.5-0.5B

Reports, per run and pooled: how often the filter rejects a story the paper
labelled collapsed (recall), how often it rejects one it did not (false-positive
rate), and which check fired. The perplexity reference is built from the clean
baseline conditions, so a run that is mostly broken is still judged against
something that is not.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from noiseegra.coherence import (  # noqa: E402
    CoherenceFilter, CoherenceThresholds, PerplexityScorer,
    apply_sentence_coherence, nll_reference, sentence_coherence,
)

RESULTS = Path(__file__).resolve().parents[1] / "experiment_results"
LABELS = Path(__file__).resolve().parents[1] / "noiseegra" / "data" / "paper_modal_collapse_indices.json"
STORY_DIRS = ["baseline", "ResidNoise", "AttnNoise", "EmbedNoise", "AENIMaxW",
              "TwoStage", "DoubleResid"]


def detokenise(text: str) -> str:
    """The published CSVs store SentencePiece output, not plain text."""
    if "▁" in text:
        return text.replace("<0x0A>", "\n").replace(" ", "").replace("▁", " ")
    return text


def find_run(stem: str) -> Path | None:
    for d in STORY_DIRS:
        p = RESULTS / d / f"{stem}.csv"
        if p.is_file():
            return p
    return None


def load(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as fh:
        return [detokenise(r[0]) for r in csv.reader(fh) if r and r[0].strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coherence-model", default=None,
                    help="small LM for the perplexity checks, e.g. Qwen/Qwen2.5-0.5B")
    ap.add_argument("--embedding-model", default=None,
                    help="enable the sentence-to-sentence coherence check with this "
                         "embedding model (registry key or HF id; bge-m3 for Arabic)")
    ap.add_argument("--ppl-z", type=float, default=3.5)
    ap.add_argument("--tail-gap", type=float, default=1.5)
    ap.add_argument("--min-words", type=int, default=15)
    ap.add_argument("--no-trim-tail", dest="trim", action="store_false")
    args = ap.parse_args()

    if not LABELS.is_file():
        raise SystemExit(f"no label file at {LABELS}")
    labels = {k: set(v) for k, v in json.loads(LABELS.read_text(encoding="utf-8")).items()}

    runs = []
    for stem, collapsed in sorted(labels.items()):
        path = find_run(stem)
        if path is None:
            continue
        stories = load(path)
        if len(stories) >= 10:
            runs.append((stem, stories, collapsed))
    if not runs:
        raise SystemExit("found no story files matching the label file")

    thresholds = CoherenceThresholds(min_words=args.min_words, max_ppl_z=args.ppl_z,
                                     max_tail_nll_gap=args.tail_gap)
    ppl = PerplexityScorer(args.coherence_model) if args.coherence_model else None
    filt = CoherenceFilter(thresholds, trim=args.trim, ppl_scorer=ppl)

    reports = {stem: [filt.check(t) for t in stories] for stem, stories, _ in runs}

    def clean_pool(values_by_run):
        """Reference built from runs with no labelled collapse at all."""
        pool = [v for stem, _, coll in runs if not coll for v in values_by_run[stem]
                if v == v]
        if not pool:
            print("no fully clean run to calibrate on; pooling everything instead")
            pool = [v for vs in values_by_run.values() for v in vs if v == v]
        return nll_reference(pool)

    if args.embedding_model:
        from noiseegra.diversity import DiversityScorer

        scorer = DiversityScorer(args.embedding_model)
        sims = {}
        for stem, _, _ in runs:
            print(f"sentence coherence: {stem}", flush=True)
            sims[stem] = sentence_coherence([r.text or " " for r in reports[stem]],
                                            lambda t: scorer.encode(t, truncate=None))
        ref = clean_pool(sims)
        print(f"coherence reference: median {ref[0]:.3f}, scale {ref[1]:.3f}\n")
        for stem, _, _ in runs:
            apply_sentence_coherence(reports[stem], sims[stem], thresholds, ref)

    if ppl is not None:
        triples = {}
        for stem, _, _ in runs:
            print(f"perplexity: {stem}", flush=True)
            triples[stem] = ppl.score([r.text or " " for r in reports[stem]],
                                      thresholds.tail_fraction)
        reference = clean_pool({n: [o for o, _, _ in v] for n, v in triples.items()})
        print(f"perplexity reference: median NLL {reference[0]:.3f}, scale {reference[1]:.3f}\n")
        for stem, _, _ in runs:
            filt.apply_perplexity(reports[stem], triples[stem], reference)

    tp = fp = tn = fn = 0
    fired: dict[str, int] = {}
    print(f"{'run':<52}{'n':>4}{'lab':>5}{'recall':>8}{'FPR':>7}")
    for stem, stories, collapsed in runs:
        reps = reports[stem]
        pos = [i for i in range(len(reps)) if i in collapsed]
        neg = [i for i in range(len(reps)) if i not in collapsed]
        r_tp = sum(1 for i in pos if not reps[i].ok)
        r_fp = sum(1 for i in neg if not reps[i].ok)
        tp += r_tp; fn += len(pos) - r_tp; fp += r_fp; tn += len(neg) - r_fp
        for i in pos:
            for reason in reps[i].reasons:
                fired[reason] = fired.get(reason, 0) + 1
        rec = f"{r_tp / len(pos):>7.0%}" if pos else "      -"
        fpr = f"{r_fp / len(neg):>6.0%}" if neg else "     -"
        print(f"{stem[:51]:<52}{len(reps):>4}{len(pos):>5}{rec}{fpr}")

    print(f"\npooled: {tp + fn} labelled collapsed, {tn + fp} not")
    if tp + fn:
        print(f"  recall (labelled collapsed that the filter rejects) {tp / (tp + fn):.0%}")
    if fp + tn:
        print(f"  false-positive rate (clean stories rejected)        {fp / (fp + tn):.0%}")
    if tp + fp:
        print(f"  precision (rejections that were labelled collapsed) {tp / (tp + fp):.0%}")
    if fired:
        print("\n  which check fired on the labelled-collapsed stories:")
        for reason, n in sorted(fired.items(), key=lambda kv: -kv[1]):
            print(f"    {reason:<20} {n}")
    if ppl is None and not args.embedding_model:
        print("\nSurface heuristics only. The published collapse cases are largely "
              "fluent-looking word salad -- correct characters, real words, ordinary "
              "punctuation -- which surface checks cannot see. Add --embedding-model "
              "for the sentence-coherence check, --coherence-model for the perplexity "
              "checks, or both; those are the parts aimed at that failure.")


if __name__ == "__main__":
    main()
