#!/usr/bin/env python
"""1/f^beta noise along the token axis, and the split that decides if it can work.

This project's structural finding is that a perturbation varying only *within* a
story cannot make a set of stories more varied. So the quantity that matters for
a noise process here is not its variance but how that variance splits: the part
carried by a story's whole-trajectory mean, which differs between stories, and
the part that only wobbles inside one.

These checks fix both ends of the axis against the two mechanisms already known
-- beta=0 is per-token noise, large beta with a sub-cycle cut-off is the fixed
per-story displacement -- and check that the size of the perturbation does not
change with beta, or an exponent sweep would secretly be a magnitude sweep.

Offline, no GPU, no model.

    python tests/test_colored_noise.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from noiseegra.colored_noise import (  # noqa: E402
    between_story_share, colored_sequence,
)

FAIL = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAIL.append(name)


def ensemble(beta, cycles=0.25, length=256, width=4, stories=128, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.stack([colored_sequence(length, width, beta=beta,
                                         fmin_cycles=cycles, generator=g)
                        for _ in range(stories)])


# --- the spectrum is the thing it claims to be ------------------------------
# Fit the slope of log power against log frequency, away from the cut-off and
# away from Nyquist. It should come out at -beta.
for beta in (0.0, 1.0, 2.0):
    tr = ensemble(beta, cycles=1.0, length=1024, width=1, stories=64)
    spec = torch.fft.rfft(tr[:, :, 0].double(), dim=1).abs() ** 2
    power = spec.mean(dim=0).numpy()
    freq = np.fft.rfftfreq(1024)
    lo, hi = 8, 400                       # clear of the cut-off and of Nyquist
    slope = np.polyfit(np.log(freq[lo:hi]), np.log(power[lo:hi]), 1)[0]
    check(f"the power spectrum falls as 1/f^{beta:g}",
          abs(slope + beta) < 0.15, f"fitted slope {slope:+.3f}")

# --- beta must not double as a magnitude knob -------------------------------
sds = {b: float(ensemble(b).std()) for b in (0.0, 1.0, 2.0, 4.0)}
check("the perturbation is the same size at every exponent",
      all(abs(v - 1.0) < 0.05 for v in sds.values()),
      ", ".join(f"beta={k:g}: {v:.3f}" for k, v in sds.items()))

# --- the two ends are the two mechanisms already known ----------------------
white = between_story_share(ensemble(0.0))
check("beta=0 is per-token noise: nothing differs between stories",
      white < 0.02, f"share {white:.4f}")

const = between_story_share(ensemble(6.0, cycles=0.1))
check("a large exponent under a sub-cycle cut-off is one displacement per story",
      const > 0.98, f"share {const:.4f}")

# --- and the axis between them is monotone ----------------------------------
shares = [between_story_share(ensemble(b)) for b in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)]
check("the between-story share rises with the exponent",
      all(a < b for a, b in zip(shares, shares[1:])),
      " < ".join(f"{s:.3f}" for s in shares))

# A cut-off of one cycle per story cannot produce a per-story displacement
# however large the exponent: the slowest component completes a whole cycle
# inside the story and averages to nothing. This is why the cut-off exists.
capped = between_story_share(ensemble(6.0, cycles=1.0))
check("at one cycle per story the share stays low whatever the exponent",
      capped < 0.35, f"share {capped:.3f}")

# --- the middle of the axis is genuinely mixed, which is the point ----------
mid = between_story_share(ensemble(1.5, cycles=0.25))
check("an intermediate setting is neither end: it varies both within and between",
      0.2 < mid < 0.8, f"share {mid:.3f}")

# --- reproducibility and refusals -------------------------------------------
a = colored_sequence(64, 2, beta=1.0, generator=torch.Generator().manual_seed(7))
b = colored_sequence(64, 2, beta=1.0, generator=torch.Generator().manual_seed(7))
check("the same seed gives the same trajectory", torch.equal(a, b))
c = colored_sequence(64, 2, beta=1.0, generator=torch.Generator().manual_seed(8))
check("a different seed gives a different one", not torch.equal(a, c))
check("the shape is length x width", tuple(a.shape) == (64, 2), str(tuple(a.shape)))


def refuses(fn):
    try:
        fn()
        return False
    except ValueError:
        return True


check("a negative exponent is refused",
      refuses(lambda: colored_sequence(32, 1, beta=-1.0)))
check("a cut-off the sequence cannot carry is refused",
      refuses(lambda: colored_sequence(4, 1, beta=1.0, fmin_cycles=8.0)))
check("a non-positive cut-off is refused",
      refuses(lambda: colored_sequence(32, 1, beta=1.0, fmin_cycles=0.0)))
check("one story cannot have its variance split",
      refuses(lambda: between_story_share(torch.zeros(1, 8, 2))))

# An odd length exercises the branch with no Nyquist bin.
odd = ensemble(1.0, length=255, stories=32)
check("an odd-length sequence is still unit size", abs(float(odd.std()) - 1.0) < 0.08,
      f"{float(odd.std()):.3f}")

print()
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
    sys.exit(1)
print("ALL TESTS PASS")
