#!/usr/bin/env python
"""Print stories from one condition with the requirements each of them breaks.

    python scripts/read_stories.py --input-dir experiments/qwen-round2/state/Qwen3-8B
    python scripts/read_stories.py --input-dir ... --run "at the prompt" -n 8

A table of pass rates says how often a rule is broken and nothing about whether
the text is worth reading. A condition can win on Vendi by writing forty
different kinds of broken sentence. This is the check that catches that, and it
is a script rather than an ad-hoc look so the same sample can be pulled again.

With no ``--run`` it lists the conditions in the directory and stops.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from noiseegra.constraint_metrics_en import (  # noqa: E402
    CONSTRAINT_NAMES, EnglishConstraintChecker,
)
from noiseegra.diversity import read_run_csv  # noqa: E402
from noiseegra.run_labels import label_run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--run", help="substring of the run id or of its readable label")
    ap.add_argument("-n", type=int, default=6, help="how many stories to print")
    ap.add_argument("--seed", type=int, default=0, help="which sample to take")
    ap.add_argument("--worst", action="store_true",
                    help="take the stories breaking the most requirements instead of "
                         "a random sample")
    ap.add_argument("--constraints", nargs="*", default=list(CONSTRAINT_NAMES))
    ap.add_argument("--backend", default="auto", choices=["auto", "spacy", "regex"])
    args = ap.parse_args()

    d = Path(args.input_dir)
    csvs = sorted(p for p in d.glob("*.csv")
                  if not p.name.startswith(("live_scores", "English_Constraint")))
    if not csvs:
        raise SystemExit(f"no run CSVs in {d}")

    if not args.run:
        print(f"{len(csvs)} conditions in {d}:\n")
        for p in csvs:
            print(f"  {label_run(p.stem).text}")
        print("\nPass a substring of one to --run.")
        return

    needle = args.run.lower()
    hits = [p for p in csvs
            if needle in p.stem.lower() or needle in label_run(p.stem).text.lower()]
    if not hits:
        raise SystemExit(f"nothing matches {args.run!r}; run without --run to list them")
    if len(hits) > 1:
        print(f"{args.run!r} matches {len(hits)} conditions:")
        for p in hits:
            print(f"  {label_run(p.stem).text}")
        raise SystemExit("be more specific")

    path = hits[0]
    stories, _ = read_run_csv(path)
    checker = EnglishConstraintChecker(backend=args.backend,
                                       constraints=list(args.constraints))
    short = checker.requirements_short()
    scored = [(checker.evaluate(s, i), s) for i, s in enumerate(stories)]

    if args.worst:
        picked = sorted(scored, key=lambda x: -x[0].violations)[: args.n]
    else:
        picked = random.Random(args.seed).sample(scored, min(args.n, len(scored)))

    mean = sum(m.violations for m, _ in scored) / max(len(scored), 1)
    print(f"{label_run(path.stem).text}")
    print(f"{len(stories)} stories, {mean:.2f} of {len(checker.constraints)} "
          f"requirements broken on average")
    print(f"showing {len(picked)} "
          + ("worst" if args.worst else f"at random (seed {args.seed})") + "\n")

    for m, text in picked:
        broken = [short[c] for c in checker.constraints if m.checks[c] is False]
        print("-" * 78)
        print(f"  story {m.story_index}   {m.word_count} words, {m.n_sentences} "
              f"sentences, grade {m.grade_level:.1f}")
        print(f"  breaks {len(broken)}: " + ("; ".join(broken) if broken else "nothing"))
        print()
        for line in text.strip().splitlines():
            print("    " + line)
        print()


if __name__ == "__main__":
    main()
