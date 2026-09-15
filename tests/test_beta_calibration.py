"""Signing each steering direction by the side its requirement fails on.

python tests/test_beta_calibration.py
"""

import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noiseegra.beta_calibration import calibrate, describe  # noqa: E402
from noiseegra.constraint_metrics_en import EnglishConstraintChecker  # noqa: E402

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


CK = EnglishConstraintChecker(backend="regex")
NAMES = ["present_tense", "simple_register", "dialogue", "terse", "varied_openers"]

# What Qwen3-8B actually does unsteered: about ten sentences of four words each,
# so the four-word floor breaks and the ten-word ceiling never does.
CHOPPY = ('Mira runs. She stops. Rain falls. Birds fly. Sam waits. Dogs bark. '
          'Cars pass. Wind blows. Leaves turn. Night comes.')
# The opposite failure: sentences well over the ceiling.
RAMBLING = ('Mira runs down the long wet road past the shuttered market and the '
            'broken fence while the rain keeps falling on her thin coat. '
            'She stops beside the old blue door and waits there for a while '
            'because the wind has picked up and the light is going.')

print("== the side that is failing sets the sign ==")
b, notes = calibrate([CHOPPY] * 20, CK, NAMES)
for line in notes:
    print("   " + line.strip())
check("sentences under the floor ask for less terseness", b["terse"] == -1.0,
      f"{b['terse']:+g}")

b2, _ = calibrate([RAMBLING] * 20, CK, NAMES)
check("sentences over the ceiling ask for more of it", b2["terse"] == +1.0,
      f"{b2['terse']:+g}")

print("\n== a requirement that passes is not steered ==")
COMPLIANT = (
    'Mira feeds the hens. Two of them peck at her boot. She laughs and steps back '
    'very fast. "Come here," Mira says to them. The grey hen is on a log. Mira lifts '
    'it down with care. "Now you stay here," she says. Soon the hens run to the grass.'
)
b3, notes3 = calibrate([COMPLIANT] * 20, CK, NAMES)
for line in notes3:
    print("   " + line.strip())
check("a fully compliant sample steers nothing",
      all(v == 0.0 for v in b3.values()), describe(b3))

print("\n== the two-sided rules go both ways ==")
NO_QUOTES = COMPLIANT.replace('"Come here," Mira says to them.', 'Mira calls to them.') \
                     .replace('"Now you stay here," she says.', 'She tells them to stay.')
b4, _ = calibrate([NO_QUOTES] * 20, CK, NAMES)
check("too few quoted lines ask for more dialogue", b4["dialogue"] == +1.0,
      f"{b4['dialogue']:+g}")
TOO_MANY = COMPLIANT + ' "One more," she says. "And another," he says.'
b5, _ = calibrate([TOO_MANY] * 20, CK, NAMES)
check("too many ask for less", b5["dialogue"] == -1.0, f"{b5['dialogue']:+g}")

print("\n== details ==")
check("the magnitude passed in is respected",
      calibrate([CHOPPY] * 20, CK, NAMES, beta=2.5)[0]["terse"] == -2.5)
check("a direction with no probe keeps its nominal sign",
      calibrate([CHOPPY] * 20, CK, ["closure"], beta=1.0)[0]["closure"] == 1.0)
mixed = [CHOPPY] * 19 + [COMPLIANT]
check("one story in twenty is inside the tolerance and does not flip a sign",
      calibrate([COMPLIANT] * 19 + [CHOPPY], CK, NAMES)[0]["terse"] == 0.0,
      describe(calibrate([COMPLIANT] * 19 + [CHOPPY], CK, NAMES)[0]))
check("every direction asked for gets a coefficient",
      set(calibrate(mixed, CK, NAMES)[0]) == set(NAMES))
check("there is a line of explanation for each",
      len(calibrate(mixed, CK, NAMES)[1]) == len(NAMES))

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("all beta-calibration tests passed")
