"""Run ids translated back into readable condition names."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noiseegra.run_labels import (  # noqa: E402
    label_from_run_id, label_run, plan_summary, sorted_labels, untag_float,
)

failures = []


def check(name, cond, detail=""):
    print(f"{'[ok] ' if cond else '[FAIL] '}{name}" + (f"  {detail}" if detail else ""))
    if not cond:
        failures.append(name)


ORTHO = "Granite-3.1-8B__ORTHO__L17-25__Cclo-pre-sim-dia__b1-1-1-1__lowdin"
SWEEP = ([ORTHO + "__nznone__a0__k36__schr-c-c-c"]
         + [ORTHO + f"__nzorth__a{m}__k36__schr-c-c-c"
            for m in ("0p05", "0p175", "0p4", "0p8")]
         + [ORTHO + f"__nznone__a0__k36__g{m}orth__schr-c-c-c"
            for m in ("0p05", "0p175", "0p4", "0p8")])

print("== float tags ==")
check("0p4 -> 0.4", untag_float("0p4") == 0.4)
check("0p175 -> 0.175", untag_float("0p175") == 0.175)
check("m1p5 -> -1.5", untag_float("m1p5") == -1.5)
check("garbage -> 0.0", untag_float("nope") == 0.0)

print("\n== the alpha sweep ==")
expected = [
    "steering only, no perturbation",
    "per-token noise a=0.05 (orthogonal)",
    "per-token noise a=0.175 (orthogonal)",
    "per-token noise a=0.4 (orthogonal)",
    "per-token noise a=0.8 (orthogonal)",
    "per-story offset g=0.05 (orthogonal)",
    "per-story offset g=0.175 (orthogonal)",
    "per-story offset g=0.4 (orthogonal)",
    "per-story offset g=0.8 (orthogonal)",
]
got = [lab.text for lab in sorted_labels(SWEEP)]
check("every condition is named", got == expected, "\n     " + "\n     ".join(got))
check("sorted by family then magnitude, not alphabetically", got == expected)

print("\n== the subspace mode is spelled out ==")
for tag, word in [("orth", "orthogonal"), ("iso", "unrestricted"), ("para", "in-subspace")]:
    lab = label_run(ORTHO + f"__nz{tag}__a0p4__k36")
    check(f"{tag} -> {word}", word in lab.text, lab.text)

print("\n== the published Arabic run ids ==")
cases = {
    "ALLam__BASELINE": "baseline",
    "ALLam__BASELINE__temp1p8__topk40": "baseline (temperature 1.8, top-k 40)",
    "Jais__BASELINE__temp1p8__topp0p95": "baseline (temperature 1.8, top-p 0.95)",
    "ALLam__L12-20__std0p036__decay0":
        "residual-stream noise (layers 12-20; sigma 0.036)",
    "ALLam__ATTN__L12-20__std0p105__maxtok200":
        "attention-logit noise (layers 12-20; sigma 0.105; first 200 tokens)",
    "ALLam__EMBED__std0p015": "embedding noise (sigma 0.015)",
    "Fanar__ENTROPY__L18-26__std4p095":
        "AENI entropy-scaled noise (layers 18-26; sigma 4.095)",
    "Jais__TWOSTAGE_ZERO": "two-stage plan then write, no noise",
    "Jais__TWOSTAGE_RESID__L12-20__std5p25__decay0":
        "two-stage plan then write, residual noise (layers 12-20; sigma 5.25)",
    "Fanar__DOUBLE_RESID__L18-26__std10p625__std20p416667__decay0":
        "residual noise at two sites (layers 18-26; sigma 0.625 then 0.416667)",
}
for rid, want in cases.items():
    got = label_run(rid).text
    check(rid[:44], got == want, f"got {got!r}")

print("\n== families ==")
check("baseline sorts before the sweep",
      label_run("ALLam__BASELINE").sort_key < label_run(SWEEP[1]).sort_key)
check("steer only sorts before per-token noise",
      label_run(SWEEP[0]).sort_key < label_run(SWEEP[1]).sort_key)
check("per-token sorts before per-story",
      label_run(SWEEP[4]).sort_key < label_run(SWEEP[5]).sort_key)

print("\n== duplicate labels stay distinguishable ==")
dupes = sorted_labels([ORTHO + "__nzorth__a0p4__k36__schr-c-c-c",
                       ORTHO + "__nzorth__a0p4__k36__prefill"])
check("a suffix is appended when two runs share a name",
      dupes[0].text != dupes[1].text, f"{dupes[0].text} / {dupes[1].text}")

print("\n== the shared settings header ==")
summary = plan_summary(SWEEP)
joined = " | ".join(summary)
for want in ["model: Granite-3.1-8B", "steering layers: 17-25", "closure",
             "present tense", "simple register", "dialogue", "beta 1", "Lowdin",
             "36 dimensions", "orthogonal to the constraint directions", "no decay"]:
    check(f"header mentions {want!r}", want in joined, "" if want in joined else joined)

print("\n== the back-compatible shim ==")
check("known ids resolve", label_from_run_id(SWEEP[0]) == "steering only, no perturbation")
check("unknown ids give None", label_from_run_id("some_other_file") is None)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("all run-label tests passed")
