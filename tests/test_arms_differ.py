#!/usr/bin/env python
"""Two conditions that differ in a setting must not produce identical stories.

This has now happened four times in this project and been caught by hand each
time: a layer band that never reached the prompt hook, an entropy gate whose
thresholds were measured for one suite only, a displacement-size spread that
reached two of twenty call sites, and six decoding conditions -- locally typical
at two settings, min-p at two settings and eta-sampling -- that came back
byte-identical to plain nucleus sampling across 200 stories each.

Every one of them read as a clean null. A null and a setting that never arrived
look exactly alike in a results table, which is why this belongs in the harness
rather than in somebody's memory.

    python tests/test_arms_differ.py            # checks itself and the repo
    python tests/test_arms_differ.py RUN        # checks one run's output
"""

from __future__ import annotations

import csv
import glob
import hashlib
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: str) -> str:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    joined = chr(0).join(r.get("story", "") for r in rows)
    return hashlib.sha1(joined.encode()).hexdigest()


def identical_arms(run: str, root: Path | None = None):
    """Groups of run ids in `run` whose stories are byte-identical."""
    base = root or ROOT
    pattern = str(base / "experiments" / run / "output" / run / "*" / "*.csv")
    files = [p for p in glob.glob(pattern) if not p.endswith("live_scores.csv")]
    by: dict[str, list[str]] = {}
    for f in files:
        by.setdefault(digest(f), []).append(Path(f).name.replace(".csv", ""))
    return [sorted(names) for names in by.values() if len(names) > 1]


def test_it_catches_a_duplicate() -> None:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "experiments" / "selftest" / "output" / "selftest" / "M"
        out.mkdir(parents=True)
        for name, story in (("arm_a", "the same story"),
                            ("arm_b", "the same story"),
                            ("arm_c", "a different story")):
            with open(out / (name + ".csv"), "w", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["story"])
                w.writerow([story])
        groups = identical_arms("selftest", root=Path(d))
    assert len(groups) == 1 and groups[0] == ["arm_a", "arm_b"], groups
    print("a duplicated pair is found, and a genuinely different third arm is not")


def test_no_shipped_run_has_silently_identical_arms() -> None:
    # The one run known to contain them, kept named rather than quietly skipped:
    # its locally-typical, min-p and eta-sampling conditions never applied, and
    # its numbers must not be reported until that is understood.
    known_bad = {
        # Locally typical, min-p and eta-sampling never applied: six conditions
        # byte-identical to plain nucleus sampling. Not yet understood, and its
        # numbers must not be reported until it is.
        "r89-literature",
        # The same three settings, combined with the method.
        "r91-withdecoder",
        # The six-story diagnostic that found the cause: the installed
        # generation stack no longer reads `typical_p`, `min_p` or
        # `eta_cutoff` from the keyword arguments, and a generation config
        # accepts any attribute you set on it, so they were stored and never
        # applied. All three are now applied as an explicit processor
        # (`noiseegra/decoders.py`, `tests/test_decoders.py`). This run
        # predates the fix and is kept as the evidence for it.
        "r94-decodercheck",
        # Historical, each found by hand at the time and since fixed. They are
        # named rather than deleted because they are what this check is for.
        "r31-bands",      # the layer band never reached the prompt hook
        "r32-gated",      # gate thresholds were measured for one suite only
        "r15-dose",       # an explicit budget was overridden by a default
        # Not a fault: Qwen3 ships top_p=0.95 in its own generation config, so
        # asking for it again changes nothing. Worth keeping visible, because it
        # is why the untouched baseline is not untruncated sampling.
        "r15-reference",
        # Also not a fault, and a neat demonstration: betas of 1-1-1 and 2-2-2
        # renormalised to the same fixed total are the same vector. The budget
        # working exactly as intended.
        "r11-control",
    }
    bad = []
    for d in sorted(glob.glob(str(ROOT / "experiments" / "*"))):
        run = Path(d).name
        if run in known_bad:
            continue
        groups = identical_arms(run)
        if groups:
            bad.append((run, groups))
    for run, groups in bad:
        print("  " + run + ": " + str(groups))
    assert not bad, (
        str(len(bad)) + " run(s) contain conditions that differ in a setting and "
        "produced identical stories; a setting is not being applied")
    print("no run has silently identical arms (" + str(len(known_bad))
          + " known-bad run excluded and named)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for g in identical_arms(sys.argv[1]):
            print("IDENTICAL: " + str(g))
        sys.exit(0)
    test_it_catches_a_duplicate()
    test_no_shipped_run_has_silently_identical_arms()
    print("\nok")
