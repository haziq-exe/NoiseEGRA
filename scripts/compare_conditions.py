#!/usr/bin/env python
"""One table comparing conditions that may live in different runs.

    python scripts/compare_conditions.py --constraint-set middle \
        --condition "untouched=r26-ladder-extracted:BASELINE" \
        --condition "push 3, extracted=r26-ladder-extracted:bud3" \
        --condition "push 3, random=r26-ladder-random:bud3"

`scripts/score_structure.py` scores the conditions inside one run directory.
This one takes conditions from anywhere, which is what a control that has to be
run on a separate Kaggle account needs: three ladders on three accounts are
three run directories, and pooling every condition at a common group size is the
only way their diversity numbers can be compared at all.

Every number here follows the order the rest of the project uses: drop the
stories the coherence checks reject, cut each story to its first 40 words, trim
the ones sitting far from the rest of their own set, and pool every condition at
the size of the smallest. The embedding-based Vendi is not computed -- it needs
a GPU -- so this reports the two local measures, which separate what one
embedding score conflates: what happens in the story, and how it is worded.

Three columns exist because the reading floor turned out to be unreadable
without them. Reading grade has only two inputs, words per sentence and
syllables per word, so a condition that chops the same prose into more sentences
scores easier without changing a word. Words per sentence and sentences per
story are printed beside it, and the share of sentences begun by the commonest
opening word is printed beside the rule that counts them, because that rule
penalises writing more sentences as much as it penalises repeating an opening.

`--per-rule` adds the pass rate of every requirement, which is where a change in
the mean number broken is actually explained.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from noiseegra.coherence import (  # noqa: E402
    CoherenceFilter, opens_with_a_title, trim_title,
)
from noiseegra.constraint_metrics_en import (  # noqa: E402
    MIDDLE_CONSTRAINTS, MONOTONE_CONSTRAINTS, EnglishConstraintChecker,
    opens_in_the_wrong_tense,
)
from noiseegra.readability import uncommon_word_share  # noqa: E402
from noiseegra.structure import (  # noqa: E402
    _set_vectors, content_lemma_sets, pos_trigram_vectors, rarefied_vendi,
    trim_isolated,
)

SENTENCE = re.compile(r"[^.!?]+[.!?]")
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def stories_from_run(run: str, fragment: str):
    """Every story for one run id, read from the checkpoint.

    The checkpoint rather than the CSVs because a resumed sharded run's merged
    CSV repeats its history once per shard.
    """
    found: dict = {}
    for path in glob.glob(f"experiments/{run}/state/**/state.json", recursive=True):
        for rid, cells in json.load(open(path)).get("runs", {}).items():
            if fragment in rid:
                found.setdefault(rid, {}).update(cells)
    if not found:
        raise SystemExit(f"no run id in {run} contains {fragment!r}")
    if len(found) > 1:
        # An exact name wins over the ids that merely contain it, so the
        # untouched model can be asked for by name in a run that also holds
        # the raised-temperature conditions built on top of that name.
        if fragment in found:
            found = {fragment: found[fragment]}
        else:
            names = "\n    ".join(sorted(found))
            raise SystemExit(
                f"{fragment!r} matches {len(found)} run ids in {run}:\n    {names}\n"
                "  Give more of the id, or its exact name.")
    rid, cells = next(iter(found.items()))
    return rid, [cells[k] for k in sorted(cells, key=lambda x: int(x.split(":")[1]))]


def opener_share(text: str) -> float:
    """Share of sentences begun by the commonest opening word."""
    heads = [(s.strip().split() or [""])[0].strip('"“”‘’,').lower()
             for s in SENTENCE.findall(text)]
    if not heads:
        return 0.0
    return max(Counter(heads).values()) / len(heads)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", action="append", default=[], metavar="NAME=RUN:FRAG",
                    help="repeatable; FRAG is any substring picking out one run id")
    ap.add_argument("--constraint-set", choices=("monotone", "middle"), default="middle")
    ap.add_argument("--truncate-words", type=int, default=40)
    ap.add_argument("--min-grade", type=float, default=3.0)
    ap.add_argument("--max-word-uses", type=int, default=5)
    ap.add_argument("--max-opener-uses", type=int, default=5)
    ap.add_argument("--max-words", type=int, default=200)
    ap.add_argument("--max-adverbs", type=int, default=5)
    ap.add_argument("--min-sensory", type=int, default=6)
    ap.add_argument("--present-ratio", type=float, default=0.9)
    ap.add_argument("--group-size", type=int,
                    help="rarefy every condition to this many kept stories "
                         "(default: the smallest kept count)")
    ap.add_argument("--draws", type=int, default=40)
    ap.add_argument("--per-rule", action="store_true",
                    help="add a pass-rate table, one row per requirement")
    args = ap.parse_args()

    rules = (MIDDLE_CONSTRAINTS if args.constraint_set == "middle"
             else MONOTONE_CONSTRAINTS)
    checker = EnglishConstraintChecker(
        backend="spacy", constraints=rules, max_opener_uses=args.max_opener_uses,
        max_word_uses=args.max_word_uses, min_grade_level=args.min_grade,
        max_words=args.max_words, max_adverbs=args.max_adverbs,
        min_sensory=args.min_sensory, present_ratio_threshold=args.present_ratio,
    )
    coherence = CoherenceFilter()

    scored = {}
    order = []
    for spec in args.condition:
        name, rest = spec.split("=", 1)
        run, fragment = rest.split(":", 1)
        rid, texts = stories_from_run(run, fragment)
        order.append(name)

        # The instruction forbids a title, and a title is also distinctive
        # content sitting in exactly the opening words every diversity number
        # here is computed over. Counted, then removed, so the scores below
        # describe the story rather than its heading.
        # A heading is scored as a broken requirement, not as a broken story:
        # the prose underneath it is usually fine, and so is the prose in a
        # story that stopped capitalising. Both cost one rule. The rate is still
        # reported, and the text is still measured with the heading removed,
        # because a title is distinctive content sitting in exactly the opening
        # words the diversity scores are computed over.
        title_flags = [opens_with_a_title(t) for t in texts]
        titled = float(np.mean(title_flags))
        scored_text = [trim_title(t)[0] for t in texts]

        reports = [coherence.check(t) for t in scored_text]
        kept = [r.text for r in reports if r.ok]
        # What a reader would actually accept: coherent, and without a heading
        # the instruction forbids. The two failures trade against each other --
        # sparing the prompt's boundary takes titles from 29% to 4% and loses
        # capitalisation in some stories instead -- so neither column alone says
        # how much of a condition's output is usable.
        usable = float(np.mean([r.ok for r in reports]))
        rejected = Counter(r.reason for r in reports if not r.ok)
        if not kept:
            raise SystemExit(f"{name}: the coherence checks rejected all {len(texts)} stories")

        # Requirements are scored on the text as written, heading and all, so
        # the formatting rule can see it; the diversity measures below use the
        # trimmed text, for the reason above.
        kept_raw = [t for t, r in zip(texts, reports) if r.ok]
        agg = checker.evaluate_all(kept_raw)
        per_story = agg["stories"]
        sentences = [len(SENTENCE.findall(t)) or 1 for t in kept]
        words = [len(WORD.findall(t)) for t in kept]

        scored[name] = dict(
            run=run, rid=rid, n=len(texts), kept=len(kept), rejected=rejected,
            broken=agg["mean_violations"], grade=agg["mean_grade_level"],
            uncommon=float(np.mean(uncommon_word_share(kept))),
            sentences=float(np.mean(sentences)),
            words_per_sentence=float(np.mean(words) / np.mean(sentences)),
            words=float(np.mean(words)),
            opener=float(np.mean([opener_share(t) for t in kept])),
            present=float(np.mean([s.present_ratio for s in per_story
                                   if s.present_ratio is not None] or [float("nan")])),
            wrong_open=float(np.mean([opens_in_the_wrong_tense(t, checker) for t in kept])),
            titled=titled, usable=usable,
            pass_rate={r: float(np.mean([s.checks.get(r, True) for s in per_story]))
                       for r in rules},
            happens_vectors=trim_isolated(
                _set_vectors(content_lemma_sets(kept, args.truncate_words))),
            wording_vectors=trim_isolated(
                pos_trigram_vectors(kept, args.truncate_words)),
        )

    if not order:
        raise SystemExit("give at least one --condition")

    pool = args.group_size or min(v["kept"] for v in scored.values())
    for v in scored.values():
        v["happens"] = rarefied_vendi(v["happens_vectors"], pool, draws=args.draws)[0]
        v["wording"] = rarefied_vendi(v["wording_vectors"], pool, draws=args.draws)[0]

    width = max(len(n) for n in order)
    print(f"\n{len(order)} conditions, {len(rules)} requirements, variety over the first "
          f"{args.truncate_words} words of coherent stories,\nisolated stories trimmed, "
          f"every condition pooled at {pool}.\n")
    head = (f"{'condition':<{width}}  {'coherent':>9}  {'broken':>7}  {'happens':>8}  "
            f"{'wording':>8}  {'grade':>6}  {'w/sent':>7}  {'sents':>6}  "
            f"{'uncommon':>9}  {'opener':>7}  {'present':>8}  {'past open':>10}  {'titled':>7}  {'usable':>7}")
    print(head)
    print("-" * len(head))
    for name in order:
        v = scored[name]
        print(f"{name:<{width}}  {v['kept']:>4}/{v['n']:<4}  {v['broken']:>7.2f}  "
              f"{v['happens']:>8.1f}  {v['wording']:>8.1f}  {v['grade']:>6.2f}  "
              f"{v['words_per_sentence']:>7.1f}  {v['sentences']:>6.1f}  "
              f"{v['uncommon']:>8.1%}  {v['opener']:>6.0%}  {v['present']:>7.0%}  "
              f"{v['wrong_open']:>9.0%}  {v['titled']:>6.0%}  {v['usable']:>6.0%}")

    print("\n  broken    mean requirements broken per story, out of "
          f"{len(rules)}; lower is better")
    print("  happens   effective number of stories in which different things happen")
    print("  wording   effective number of sentence shapes")
    print("  grade     Flesch-Kincaid, from words per sentence and syllables per word only")
    print("  w/sent    mean words per sentence;  sents  mean sentences per story")
    print("  uncommon  share of words outside the 3000 commonest in the Brown corpus")
    print("  opener    share of sentences begun by the story's commonest opening word")
    print("  present   share of finite verbs in the present tense, before any threshold")
    print("  past open share of stories opening in the past tense then narrating in the")
    print("            present -- a flaw the whole-story share above cannot see")
    print("  titled    share opening with a title or heading, which the instruction")
    print("            forbids. Removed before every other number in the row")
    print("  usable    share whose prose the coherence checks accept. A heading or")
    print("            lost capitals is a broken requirement, counted in `broken`,")
    print("            not a broken story: the prose underneath is fine either way")

    rejections = {n: scored[n]["rejected"] for n in order if scored[n]["rejected"]}
    if rejections:
        print("\nwhy stories were rejected")
        for name, counts in rejections.items():
            print(f"  {name:<{width}}  " +
                  ", ".join(f"{reason} {count}" for reason, count in counts.most_common()))

    if args.per_rule:
        print("\npass rate per requirement")
        label = max(len(r) for r in rules)
        print(f"{'requirement':<{label}}  " + "  ".join(f"{n:>{max(8, len(n))}}" for n in order))
        for rule in rules:
            cells = "  ".join(f"{scored[n]['pass_rate'][rule]:>{max(8, len(n))}.0%}"
                              for n in order)
            print(f"{rule:<{label}}  {cells}")

    print("\nrun ids")
    for name in order:
        print(f"  {name:<{width}}  {scored[name]['run']} :: {scored[name]['rid'][-60:]}")


if __name__ == "__main__":
    main()
