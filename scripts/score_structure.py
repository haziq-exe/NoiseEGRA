#!/usr/bin/env python
"""Degeneracy and structural diversity per condition, from saved stories, locally.

    python scripts/score_structure.py --input-dir experiments/<run>/state/<model>
    python scripts/score_structure.py --input-dir ... --truncate-words 40

For every condition: how many stories the coherence checks flag (and why), the
structural diversity of the stories that pass, and their syntactic diversity.
No GPU and no embedding model -- spaCy only -- so it runs on this machine over
any pulled run, unlike the embedding Vendi, which is computed on the kernel.

The two diversity columns separate what one embedding Vendi conflates:

  structural   effective number of stories in which different things happen:
               each story reduced to the set of content lemmas in it (names
               excluded), Vendi over those sets. Phrasing cannot move it.
  syntactic    effective number of sentence shapes: Vendi over part-of-speech
               trigram profiles. Phrasing is all that moves it.

Both are computed only over stories that pass the coherence filter, rarefied to
the same group size in every condition (Vendi grows with group size, and the
filter keeps different numbers per condition). The flagged share is printed
beside them because filtering is not free: a condition that loses a third of its
stories to the filter is being carried by the rest.

Truncation defaults to 40 words to match the embedding Vendi tables; never
compare a number at one truncation with a number at another.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from noiseegra.coherence import CoherenceFilter  # noqa: E402
from noiseegra.diversity import read_run_csv  # noqa: E402
from noiseegra.run_labels import label_run, sorted_labels  # noqa: E402
from noiseegra.structure import (  # noqa: E402
    _set_vectors, content_lemma_sets, pos_trigram_vectors, rarefied_vendi,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, help="directory of run CSVs")
    ap.add_argument("--truncate-words", type=int, default=40,
                    help="cut every story to its first N words before the diversity "
                         "scores, matching the embedding Vendi tables (default 40)")
    ap.add_argument("--group-size", type=int, default=None,
                    help="rarefy every condition to this many kept stories "
                         "(default: the smallest kept count across conditions)")
    ap.add_argument("--vendi-q", type=float, default=1.0,
                    help="order of the Vendi score; 1 is the usual Shannon form, "
                         "2 discounts lone outliers (default 1)")
    ap.add_argument("--draws", type=int, default=12,
                    help="random subsets per condition for the rarefied scores")
    args = ap.parse_args()

    d = Path(args.input_dir)
    csvs = sorted(p for p in d.glob("*.csv")
                  if not p.name.startswith(("live_scores", "English_Constraint")))
    if not csvs:
        raise SystemExit(f"no run CSVs in {d}")

    filt = CoherenceFilter()
    rows = []
    for p in csvs:
        texts, _ = read_run_csv(p)
        if len(texts) < 4:
            continue
        reports = [filt.check(t) for t in texts]
        kept = [r.text for r in reports if r.ok]
        why = Counter(reason for r in reports for reason in r.reasons)
        rows.append(dict(rid=p.stem, n=len(texts), kept=kept, why=why))
    if not rows:
        raise SystemExit("nothing to score")

    m = args.group_size or min(len(r["kept"]) for r in rows)
    if m < 2:
        raise SystemExit("a condition keeps fewer than 2 stories; pass --group-size")

    labs = sorted_labels([r["rid"] for r in rows])
    order = {lab.run_id: i for i, lab in enumerate(labs)}
    text = {lab.run_id: lab.text for lab in labs}
    rows.sort(key=lambda r: order.get(r["rid"], 99))

    print(f"{len(rows)} conditions; diversity over the first {args.truncate_words} "
          f"words of coherent stories, rarefied to {m} per condition "
          f"({args.draws} draws), Vendi order q={args.vendi_q:g}\n")
    hdr = (f"{'condition':<58}{'n':>5}{'flag%':>7}{'struct':>8}{'syntax':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        label = text.get(r["rid"]) or label_run(r["rid"]).text
        flagged = 1.0 - len(r["kept"]) / r["n"]
        if len(r["kept"]) >= 2:
            sv = _set_vectors(content_lemma_sets(r["kept"], args.truncate_words))
            pv = pos_trigram_vectors(r["kept"], args.truncate_words)
            s_mean, _ = rarefied_vendi(sv, m, draws=args.draws, q=args.vendi_q)
            p_mean, _ = rarefied_vendi(pv, m, draws=args.draws, q=args.vendi_q)
            print(f"{label:<58}{r['n']:>5}{flagged:>7.0%}{s_mean:>8.2f}{p_mean:>8.2f}")
        else:
            print(f"{label:<58}{r['n']:>5}{flagged:>7.0%}{'-':>8}{'-':>8}")
    print("\nwhy stories were flagged, per condition:")
    for r in rows:
        if not r["why"]:
            continue
        label = text.get(r["rid"]) or label_run(r["rid"]).text
        why = ", ".join(f"{k} {v}" for k, v in r["why"].most_common())
        print(f"  {label}: {why}")


if __name__ == "__main__":
    main()
