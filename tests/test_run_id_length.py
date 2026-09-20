#!/usr/bin/env python
"""A run id has to fit in a file name, at any number of directions.

The stories of a condition go in `<run_id>.csv`, and a file name cannot exceed
255 bytes. Spelling out every direction and every weight is readable at five
directions and impossible at thirteen: measured weights are fractions like
0.28125 rather than ones and zeros, so thirteen of them run to eighty
characters and the name reaches 282.

The way it failed is the reason this test exists. The arm whose weights were
ones and zeros generated fine; the arm whose weights were fractions could not
open its file, and only after two hundred stories had been generated. An
eight-arm run spent an hour and returned one arm.

Offline, no GPU, no model.

    python tests/test_run_id_length.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from noiseegra.setup_experiment import _MAX_RUN_ID, _spec_to_run_id, make_specs  # noqa: E402
from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAIL.append(name)


TWELVE = ["present_tense", "mature_register", "dialogue", "varied_openers",
          "plain_words", "sensory", "fresh_words", "named_character",
          "no_repetition", "distinct_sentences", "fresh_openings", "no_heading",
          "something_happens"]
# The shares this project's own allocation produced from 32 calibration
# stories. Thirds and thirty-seconds, not round numbers.
MEASURED = [1.0, 0.3125, 0.5, 0.8125, 0.5625, 0.375, 0.28125, 0.34375, 0.0625,
            0.21875, 0.28125, 0.0, 0.395833]


def run_id(names, weights, **kw):
    torch.manual_seed(0)
    plan = SteeringPlan.build(
        vectors={n: {0: torch.randn(16)} for n in names}, layers=[0],
        specs=[ConstraintSpec(n, beta=w) for n, w in zip(names, weights)],
        rms_scale=1.0, steer_budget=2.5, offset_gamma=1.5, offset_mode="orth",
        offset_basis={0: torch.linalg.qr(torch.randn(16, 6))[0]},
        offset_prefill=True, offset_norm="story", offset_draw_shape="manifold",
        prompt_tail_clear=8, **kw)
    return _spec_to_run_id("Qwen3-1.7B", make_specs({"plan": plan})[0])


measured = run_id(TWELVE, MEASURED, noise_beta=2.0, offset_envelope="rise")
check("thirteen directions at measured weights fit in a file name",
      len(measured) + len(".csv") <= 255, f"{len(measured)} characters")
check("and stay inside the margin this code keeps",
      len(measured) <= _MAX_RUN_ID, f"{len(measured)} of {_MAX_RUN_ID}")

# The compression must not lose the distinction between arms. If it did, two
# conditions would write to one file and the second would replace the first --
# which is worse than the crash, because it looks like a result.
nudged = run_id(TWELVE, [w if i != 3 else 0.8124 for i, w in enumerate(MEASURED)],
                noise_beta=2.0, offset_envelope="rise")
check("changing one weight in the fourth decimal still changes the id",
      measured != nudged)
check("changing the noise colour still changes the id",
      measured != run_id(TWELVE, MEASURED, noise_beta=1.0, offset_envelope="rise"))
check("changing the envelope still changes the id",
      measured != run_id(TWELVE, MEASURED, noise_beta=2.0))

# A handful of directions must stay readable: the compression is a fallback,
# not the normal case, and a run id nobody can read is its own problem.
five = run_id(TWELVE[:5], [1.0] * 5)
check("five directions are still spelled out, not digested",
      "pre-mat-dia-var-pla" in five, five[:70])
check("and thirteen are not", "13dir" in measured, measured[:70])

# Every arm of the suite that motivated this, at the weights that broke it.
ids = []
for weights in (MEASURED, [1.0 if n in ("present_tense", "sensory",
                                        "named_character", "no_heading",
                                        "something_happens") else 0.0
                           for n in TWELVE]):
    for kw in ({}, {"noise_beta": 2.0}, {"noise_beta": 1.0},
               {"noise_beta": 0.0}, {"noise_beta": 2.0, "offset_envelope": "rise"}):
        ids.append(run_id(TWELVE, weights, **kw))
check("every arm of a thirteen-direction suite fits",
      all(len(i) + 4 <= 255 for i in ids), f"longest {max(len(i) for i in ids)}")
check("and every arm is distinct",
      len(set(ids)) == len(ids), f"{len(set(ids))} unique of {len(ids)}")

# The file has to be creatable, not merely short. Names that differ only past
# a filesystem's limit collide silently.
import tempfile  # noqa: E402

with tempfile.TemporaryDirectory() as d:
    made = []
    for i in ids:
        try:
            p = Path(d) / (i + ".csv")
            p.write_text("story\n", encoding="utf-8")
            made.append(p.name)
        except OSError as exc:
            check(f"the file for {i[:40]}... can be created", False, str(exc))
    check("every id can actually be written to disk under its own name",
          len(set(made)) == len(ids), f"{len(set(made))} files for {len(ids)} arms")

print()
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
    sys.exit(1)
print("ALL TESTS PASS")
