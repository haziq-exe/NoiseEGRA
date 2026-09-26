"""The activation-scale calibrator on a model that returns no cache.

    python tests/test_rms_no_cache.py

Granite 4.0's hybrid class in transformers 5.0.0 returns no cache unless it is
handed one, and decoding token by token without it ran every token with no
context: the scale came out at 462 instead of 0.3, and the rule steering sized
from it wrote nonsense. The calibrator now reruns the sequence each step when no
cache comes back. This checks, on a tiny random Llama, that doing so measures
what the cached decode measures. Skipped when the model cannot be downloaded.
"""
import sys, warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    MID = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    model = AutoModelForCausalLM.from_pretrained(MID).eval()
    tok = AutoTokenizer.from_pretrained(MID)
except Exception as exc:  # no network
    print(f"skipped: {exc.__class__.__name__}")
    sys.exit(0)

from noiseegra.EGRA_functions import EGRA  # noqa: E402
from noiseegra.RMS_std import RMSCalibrator  # noqa: E402

e = EGRA.__new__(EGRA)
e.device = "cpu"
e.model, e.tokenizer = model, tok
prompt = "Once upon a time there was a small town by the sea"
a = RMSCalibrator(e).collect_block_rms(prompt, layers=[0, 1], max_new_tokens=6)
real = model.forward


def no_cache(*args, **kw):
    out = real(*args, **kw)
    out.past_key_values = None
    return out


model.forward = no_cache
b = RMSCalibrator(e).collect_block_rms(prompt, layers=[0, 1], max_new_tokens=6)
model.forward = real
check("a scale is measured with no cache returned", set(b) == set(a) and all(v > 0 for v in b.values()))
check("and it is the scale the cached decode measures",
      max(abs(a[k] - b[k]) for k in a) < 1e-4 * max(a.values()),
      f"{a} vs {b}")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all passed")
