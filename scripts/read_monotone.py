#!/usr/bin/env python3
"""Score a monotone-set run: compliance, degeneration and diversity per condition.

    python scripts/read_monotone.py experiments/r15-reference [--full]

Prints one row per condition sorted by requirements broken, so the reference
curve and the method arms can be read against each other directly. `--full`
adds the per-requirement pass rates.
"""
import argparse, glob, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from noiseegra.constraint_metrics_en import (  # noqa: E402
    CONSTRAINT_SHORT, MONOTONE_CONSTRAINTS, EnglishConstraintChecker,
)


def load(root):
    runs = {}
    for f in sorted(glob.glob(os.path.join(root, "**", "state.json"), recursive=True)):
        runs.update(json.load(open(f)).get("runs", {}))
    out = {}
    for rid, cells in runs.items():
        out[rid] = [cells[k] for k in
                    sorted(cells, key=lambda x: tuple(int(i) for i in x.split(":")))]
    return out


def label(rid):
    """A readable name, from the run id alone."""
    if "BASELINE" in rid:
        bits = []
        for tag, fmt in (("temp", "temperature {}"), ("topp", "top-p {}"), ("topk", "top-k {}")):
            m = re.search(rf"__{tag}([0-9p]+)", rid)
            if m:
                bits.append(fmt.format(m.group(1).replace("p", ".")))
        return "sampling: " + (", ".join(bits) if bits else "plain (T=1.0)")
    out = []
    m = re.search(r"__b([0-9pm.-]+)__", rid)
    if m:
        first = m.group(1).split("-")[0]
        out.append("coeff " + (("-" + first[1:]) if first.startswith("m") else first).replace("p", "."))
    for tag, fmt in (("jperp", "sideways step {}"), ("jrotate", "turned {}"),
                     ("jgain", "re-weighted {}"), ("jframe", "frame jitter {}")):
        m = re.search(rf"__{tag}([0-9p]+)", rid)
        if m:
            out.append(fmt.format(m.group(1).replace("p", ".")))
    m = re.search(r"__g([0-9p]+)orth", rid)
    if m:
        out.append(f"orthogonal offset {m.group(1).replace('p', '.')}")
    m = re.search(r"__a([0-9p]+)", rid)
    if m and m.group(1) not in ("0",):
        out.append(f"noise {m.group(1).replace('p', '.')}")
    if "__smerror" in rid:
        out.append("error-driven")
    return ", ".join(out) or rid[:40]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--min-stories", type=int, default=10,
                    help="skip conditions with fewer than this, so a run can be read "
                         "while it is still going")
    a = ap.parse_args()

    ck = EnglishConstraintChecker(backend="spacy", constraints=MONOTONE_CONSTRAINTS,
                                  max_opener_uses=3)
    rows = []
    for rid, stories in load(a.root).items():
        if len(stories) < a.min_stories:
            print(f"  (skipping {label(rid)}: {len(stories)} stories)")
            continue
        rows.append((label(rid), len(stories), ck.evaluate_all(stories)))
    if not rows:
        print("nothing scored yet")
        return
    rows.sort(key=lambda r: r[2]["mean_violations"])
    ref = next((r for r in rows if r[0].startswith("sampling: plain")), rows[-1])

    print(f"\n{len(rows)} conditions, {MONOTONE_CONSTRAINTS and 13} monotone requirements\n")
    print(f"{'condition':<42}{'n':>5}{'broken/13':>11}{'vs ref':>8}{'loops':>7}{'words':>7}")
    print("-" * 80)
    for name, n, r in rows:
        print(f"{name[:40]:<42}{n:>5}{r['mean_violations']:>11.2f}"
              f"{r['mean_violations'] - ref[2]['mean_violations']:>+8.2f}"
              f"{1 - r['pass_rate']['no_repetition']:>7.0%}{r['mean_word_count']:>7.0f}")
    if a.full:
        print("\nper requirement:")
        print(f"{'condition':<42}" + "".join(f"{CONSTRAINT_SHORT[c]:>8}" for c in MONOTONE_CONSTRAINTS))
        for name, n, r in rows:
            print(f"{name[:40]:<42}" + "".join(f"{r['pass_rate'][c]:>7.0%} "
                                               for c in MONOTONE_CONSTRAINTS))


if __name__ == "__main__":
    main()
