#!/usr/bin/env python
"""Diversity measured the honest way, in one pass, with every step shown.

    python scripts/clean_vendi.py --input-dir <dir of run CSVs> [--truncate-words 40]

Four things have to happen before an embedding Vendi score means what it says,
and this runs all four in order so no combination of flags can leave one out:

1. **Reject broken stories.** A looping, collapsed or off-task story embeds far
   from everything else, so degeneration reads as variety. The coherence checks
   drop them; the table prints how many went and why.
2. **Cut every story to the same length.** Longer stories embed further apart
   whatever they say, so an unmatched comparison partly measures length.
3. **Trim isolated stories.** After the broken ones are gone, a few survivors can
   still sit far from the whole set and carry a visible slice of the score by
   themselves. These are cut by a robust z-score on each story's mean similarity
   to the rest (see `noiseegra.diversity.similarity_outliers`), one-sided, so
   only the isolated ones go.
4. **Compare at one group size.** Vendi grows with the number of stories, and
   after the first three steps conditions have kept different numbers, so every
   condition is resampled to the smallest surviving count and averaged.

The headline column is the score after all four. The columns before it are the
same measurement with the later steps switched off, so the effect of each step is
readable rather than taken on trust. The order-2 score is printed beside the
usual one: it weights the dominant modes instead of counting every item equally,
so a large gap between them says the score still rests on a few unusual stories.

Embeddings are saved next to the output so the analysis can be redone locally
without a GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from noiseegra.coherence import CoherenceFilter  # noqa: E402
from noiseegra.diversity import (  # noqa: E402
    DiversityScorer, read_run_csv, similarity_outliers, vendi_from_embeddings,
)
from noiseegra.run_labels import label_run  # noqa: E402


def rarefied(emb: np.ndarray, m: int, draws: int, q: float, seed: int = 0) -> float:
    """Mean Vendi over random subsets of ``m`` rows, so group sizes match."""
    n = emb.shape[0]
    if n < 2 or m < 2:
        return float("nan")
    if m >= n:
        return vendi_from_embeddings(emb, q=q)
    rng = np.random.default_rng(seed)
    return float(np.mean([
        vendi_from_embeddings(emb[rng.choice(n, size=m, replace=False)], q=q)
        for _ in range(draws)
    ]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--truncate-words", type=int, default=40)
    ap.add_argument("--trim-z", type=float, default=3.5,
                    help="how many robust deviations below the median similarity a "
                         "story may sit before it is trimmed; 0 disables trimming")
    ap.add_argument("--draws", type=int, default=40)
    ap.add_argument("--group-size", type=int, default=None,
                    help="default: the smallest surviving count across conditions")
    ap.add_argument("--embedding-model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    d = Path(args.input_dir)
    csvs = sorted(p for p in d.glob("*.csv")
                  if not p.name.startswith(("live_scores", "English_Constraint",
                                            "Diversity")))
    if not csvs:
        raise SystemExit(f"no run CSVs in {d}")

    filt = CoherenceFilter()
    scorer = DiversityScorer(embedding_model=args.embedding_model, device=args.device)
    print(f"embedding model: {scorer.embedding_model}", flush=True)

    conds = []
    for p in csvs:
        texts, _ = read_run_csv(p)
        if len(texts) < 4:
            continue
        reps = [filt.check(t) for t in texts]
        kept = [r.text for r in reps if r.ok]
        why: dict = {}
        for r in reps:
            for reason in r.reasons:
                why[reason] = why.get(reason, 0) + 1
        if len(kept) < 4:
            print(f"  [warn] {p.stem}: only {len(kept)} coherent stories, skipped")
            continue
        emb = np.asarray(scorer.encode(kept, truncate=args.truncate_words))
        keep_mask, sims = (similarity_outliers(emb, args.trim_z) if args.trim_z > 0
                           else (np.ones(len(emb), bool), np.zeros(len(emb))))
        conds.append(dict(name=label_run(p.stem).text, rid=p.stem, n=len(texts), kept=len(kept),
                          emb=emb, keep=keep_mask, sims=sims, why=why))
        print(f"  scored {p.stem[-46:]}", flush=True)

    # Two conditions the describer cannot tell apart -- it does not know every
    # mechanism -- would share a row name and overwrite each other's saved
    # embeddings. Where names collide, the end of the run id goes into the name.
    names = [c["name"] for c in conds]
    for c in conds:
        if names.count(c["name"]) > 1:
            c["name"] = f"{c['name'][:28]} | {c['rid'][-26:]}"

    if not conds:
        raise SystemExit("nothing to score")
    m = args.group_size or min(int(c["keep"].sum()) for c in conds)
    print(f"\nevery story cut to its first {args.truncate_words} words; "
          f"isolated stories trimmed at {args.trim_z} robust deviations; "
          f"every condition resampled to {m} stories, {args.draws} draws\n")

    head = (f"{'condition':<52}{'n':>5}{'coherent':>10}{'trimmed':>9}"
            f"{'all stories':>12}{'coherent':>10}{'+trimmed':>10}"
            f"{'CLEAN':>8}{'order-2':>9}")
    print(head)
    print("-" * len(head))
    for c in sorted(conds, key=lambda c: -rarefied(c["emb"][c["keep"]], m, args.draws, 1.0)):
        raw_all = vendi_from_embeddings(c["emb"])          # coherent, unrarefied
        trimmed = c["emb"][c["keep"]]
        v_trim = vendi_from_embeddings(trimmed)
        clean = rarefied(trimmed, m, args.draws, 1.0)
        clean2 = rarefied(trimmed, m, args.draws, 2.0)
        n_trim = int((~c["keep"]).sum())
        print(f"{c['name'][:50]:<52}{c['n']:>5}{c['kept']:>10}{n_trim:>9}"
              f"{'-':>12}{raw_all:>10.2f}{v_trim:>10.2f}{clean:>8.2f}{clean2:>9.2f}")
    print("\n  coherent     Vendi over the stories that passed the checks, as they are")
    print("  +trimmed     the same after isolated stories are removed")
    print("  CLEAN        the same again with every condition at one group size <- the number to quote")
    print("  order-2      the clean score weighted toward the bulk rather than the tail")

    print("\nwhy stories were rejected, per condition:")
    for c in conds:
        if c["why"]:
            why = ", ".join(f"{k} {v}" for k, v in
                            sorted(c["why"].items(), key=lambda kv: -kv[1]))
            print(f"  {c['name'][:60]}: {why}")

    out = Path(args.out_dir or (d / "CLEAN_VENDI"))
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "embeddings.npz",
                        **{c["rid"]: c["emb"] for c in conds})
    print(f"\nembeddings saved to {out / 'embeddings.npz'}")


if __name__ == "__main__":
    main()
