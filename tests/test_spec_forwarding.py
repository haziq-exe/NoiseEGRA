#!/usr/bin/env python
"""A decoding setting on the spec has to reach the model.

Locally typical sampling, eta-sampling, min-p and contrastive search were added
to `ExperimentSpec`, to the suite that sweeps them, to the run id that names
them and to `setup_experiment` -- and never to `generate_one`, which is the
function the English runner actually calls. So five conditions generated plain
nucleus sampling under run ids naming a decoder, and the contrastive search
baseline was really top-k 4 sampling, because `top_k` happened to be the one
part of it that was forwarded.

Nothing failed. The run ids were right, the suite was right, the numbers looked
like numbers. Three runs were spent finding it, two of them on real but
secondary faults on the same road.

This checks the wiring directly: every decoding field the spec can carry is
given a value nothing else would produce, and the model records what it was
handed.

Offline, no GPU, no model.

    python tests/test_spec_forwarding.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from kaggle_orthosteer import _DECODING_FIELDS, generate_one  # noqa: E402
from noiseegra.setup_experiment import ExperimentSpec  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAIL.append(name)


class Spy:
    """Records what it was handed, for each entry point generate_one can use."""

    def __init__(self):
        self.seen = {}

    def _record(self, which, kw):
        self.seen[which] = kw
        return "a story"

    def generate(self, prompt, **kw):
        return self._record("baseline", kw)

    def generate_with_orthogonal_steering(self, prompt, plan, **kw):
        return self._record("orthogonal_steering", kw)


# Values nothing else in the codebase would produce, so a default cannot be
# mistaken for the thing having arrived.
MARKERS = {"temperature": 1.77, "top_p": 0.911, "top_k": 41, "typical_p": 0.92,
           "min_p": 0.051, "eta_cutoff": 0.0021, "penalty_alpha": 0.61}


def a_spec():
    # The spec is frozen, so it is built with the markers rather than mutated.
    return ExperimentSpec(do_sample=True, steering_plan=object(), **MARKERS)


check("the list of decoding fields is not empty",
      len(_DECODING_FIELDS) >= 7, str(_DECODING_FIELDS))
check("every named field exists on the spec",
      all(hasattr(a_spec(), f) for f in _DECODING_FIELDS))

# --- the baseline path: every one of them must arrive -----------------------
spy = Spy()
generate_one(spy, a_spec(), "baseline", [{"role": "user", "content": "x"}],
             seed=1, max_new_tokens=8)
got = spy.seen["baseline"]
missing = [f for f in _DECODING_FIELDS if got.get(f) != MARKERS[f]]
check("a baseline condition forwards every decoding setting it carries",
      not missing, "missing or wrong: " + ", ".join(missing) if missing else "")

# The specific one that was wrong for three runs, named so a regression says so.
check("contrastive search forwards its own strength, not only its top-k",
      got.get("penalty_alpha") == MARKERS["penalty_alpha"],
      f"penalty_alpha arrived as {got.get('penalty_alpha')!r}")
check("locally typical sampling arrives",
      got.get("typical_p") == MARKERS["typical_p"])
check("min-p arrives", got.get("min_p") == MARKERS["min_p"])
check("eta-sampling arrives", got.get("eta_cutoff") == MARKERS["eta_cutoff"])

# --- the steered path: the same, minus what it cannot use -------------------
spy2 = Spy()
generate_one(spy2, a_spec(), "orthogonal_steering",
             [{"role": "user", "content": "x"}], seed=1, max_new_tokens=8)
got2 = spy2.seen["orthogonal_steering"]
# Contrastive search is not sampling and does not compose with a steering plan,
# so it is the one field this path does not take.
expect2 = [f for f in _DECODING_FIELDS if f != "penalty_alpha"]
missing2 = [f for f in expect2 if got2.get(f) != MARKERS[f]]
check("a steered condition forwards every decoding setting it can use",
      not missing2, "missing or wrong: " + ", ".join(missing2) if missing2 else "")

# --- and the model must be willing to receive them --------------------------
import inspect  # noqa: E402

from noiseegra.EGRA_functions import EGRA  # noqa: E402

sig = set(inspect.signature(EGRA.generate).parameters)
check("the model's own generate accepts every field forwarded to it",
      set(_DECODING_FIELDS) <= sig,
      "not accepted: " + ", ".join(sorted(set(_DECODING_FIELDS) - sig)))
sig2 = set(inspect.signature(EGRA.generate_with_orthogonal_steering).parameters)
check("and so does the steered one, for the fields it takes",
      set(expect2) <= sig2,
      "not accepted: " + ", ".join(sorted(set(expect2) - sig2)))

print()
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
    sys.exit(1)
print("ALL TESTS PASS")
