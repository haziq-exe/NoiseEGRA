#!/usr/bin/env python
"""The published truncation schemes must actually remove something.

Six conditions differing only in `typical_p`, `min_p` and `eta_cutoff` once came
back byte-identical to plain nucleus sampling across 200 stories each: the
settings were passed to `generate` by name, stored on a generation config that
accepts any attribute, and never read. Nothing failed and nothing was logged.

These checks work on synthetic scores, so they run anywhere and do not need a
model. Each one states the published rule and checks the set that survives.

    python tests/test_decoders.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from noiseegra.decoders import TruncationWarper, truncation_warper  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAIL.append(name)


def kept(scores, **kw):
    out = TruncationWarper(**kw)(None, scores.clone())
    return [i for i, v in enumerate(out[0].tolist()) if v != -float("inf")]


# A deliberately skewed row: one dominant token, a shoulder, and a long tail.
# No entry is allowed to sit exactly on a threshold tested below, or the check
# would be measuring floating-point rounding rather than the rule.
probs = torch.tensor([[0.50, 0.25, 0.15, 0.06, 0.021, 0.011, 0.005, 0.003]])
scores = probs.log()

# --- min-p (Nguyen et al., ICLR 2025) ---------------------------------------
# Keep p >= min_p * max_p. With max_p = 0.50 and min_p = 0.1 the floor is 0.05,
# so the first four survive and the four-token tail goes.
check("min-p keeps exactly the tokens at or above min_p times the top probability",
      kept(scores, min_p=0.1) == [0, 1, 2, 3],
      str(kept(scores, min_p=0.1)))
check("a stricter min-p keeps fewer", kept(scores, min_p=0.4) == [0, 1],
      str(kept(scores, min_p=0.4)))
check("min-p at 0 keeps everything", kept(scores, min_p=0.0) == list(range(8)))

# --- eta-sampling (Hewitt et al., EMNLP Findings 2022) ----------------------
# Floor is min(eta, sqrt(eta) * exp(-H)). Check the survivors against the rule
# computed independently here, rather than against a memorised list.
for eta in (2e-3, 1e-2, 5e-2):
    logp = torch.log_softmax(scores, dim=-1)
    p = logp.exp()
    H = float(-(p * logp).sum())
    floor = min(eta, (eta ** 0.5) * torch.tensor(-H).exp().item())
    want = [i for i, v in enumerate(p[0].tolist()) if v >= floor]
    if not want:
        want = [int(p.argmax())]
    check(f"eta-sampling at {eta} keeps the tokens above min(eta, sqrt(eta)exp(-H))",
          kept(scores, eta_cutoff=eta) == want,
          f"floor {floor:.4g}")

# --- locally typical (Meister et al., TACL 2023) ----------------------------
# Order by |-log p - H| and keep the smallest prefix reaching typical_p. The
# kept set is NOT a prefix of the probability order: a token of middling
# probability can be more typical than the most likely one.
typ = kept(scores, typical_p=0.6)
logp = torch.log_softmax(scores, dim=-1)
p = logp.exp()
H = float(-(p * logp).sum())
dev = (-logp - H).abs()[0]
order = sorted(range(8), key=lambda i: float(dev[i]))
want, total = [], 0.0
for i in order:
    want.append(i)
    total += float(p[0][i])
    if total >= 0.6:
        break
check("locally typical keeps the smallest most-typical set reaching typical_p",
      sorted(typ) == sorted(want), f"{sorted(typ)} vs {sorted(want)}")
check("a larger typical_p keeps at least as many",
      len(kept(scores, typical_p=0.95)) >= len(kept(scores, typical_p=0.2)))

# --- the properties that make it safe to run -------------------------------
check("nothing ever removes the most likely token",
      all(int(scores.argmax()) in kept(scores, **kw)
          for kw in ({"min_p": 0.99}, {"typical_p": 0.0}, {"eta_cutoff": 0.9})))

flat = torch.zeros(1, 8)
check("a flat row survives every scheme",
      all(len(kept(flat, **kw)) >= 1
          for kw in ({"min_p": 0.5}, {"typical_p": 0.5}, {"eta_cutoff": 1e-2})))

# Temperature is applied inside the processor, so it changes what is cut.
hot = kept(scores, min_p=0.1, temperature=2.0)
cold = kept(scores, min_p=0.1, temperature=0.5)
check("temperature is applied by the processor and changes what survives",
      len(hot) > len(cold), f"T=2.0 keeps {len(hot)}, T=0.5 keeps {len(cold)}")

# --- the wiring, which is what actually broke -------------------------------
check("no scheme asked for means no processor",
      truncation_warper(temperature=1.8) is None)
check("one scheme asked for means a processor",
      truncation_warper(temperature=1.8, min_p=0.05) is not None)

try:
    TruncationWarper(min_p=0.1, typical_p=0.5)
    two = False
except ValueError:
    two = True
check("asking for two schemes at once is refused rather than silently ignored", two)

# And the keyword arguments must no longer carry the three unread settings,
# because the processor applies them -- including the temperature.
from noiseegra.EGRA_functions import EGRA  # noqa: E402

kw = EGRA._sampling_kwargs(None, do_sample=True, temperature=1.8, min_p=0.05)
check("min-p is not passed to generate by name", "min_p" not in kw, str(kw))
check("temperature is left at 1.0 when the processor applies it",
      kw["temperature"] == 1.0, str(kw))
kw2 = EGRA._sampling_kwargs(None, do_sample=True, temperature=1.8, top_p=0.95)
check("an ordinary nucleus arm still asks for its own temperature",
      kw2["temperature"] == 1.8 and kw2["top_p"] == 0.95, str(kw2))

# --- end to end: the processor must reach an actual generate() call ----------
# The unit checks above are on synthetic scores. This one runs a real forward
# pass, because what broke was never the arithmetic -- it was that the setting
# never arrived. A randomly initialised model is nearly uniform over its
# vocabulary, where min-p and eta correctly remove nothing, so the temperature
# is dropped to sharpen it first.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiny_model import Tiny  # noqa: E402
from transformers import LogitsProcessorList  # noqa: E402

_m = Tiny()
_prompt = [{"role": "user", "content": "write a story"}]


def _generate(temperature, **kw):
    torch.manual_seed(7)
    chat = _m.apply_chat_template(_prompt, tokenize=False, add_generation_prompt=True)
    inp = _m.tokenizer(chat, return_tensors="pt")
    gen = _m._sampling_kwargs(do_sample=True, temperature=temperature, **kw)
    warper = truncation_warper(temperature=temperature, **kw)
    extra = {"logits_processor": LogitsProcessorList([warper])} if warper else {}
    return _m.model.generate(**inp, max_new_tokens=25, **extra, **gen)[0].tolist()


_plain_hot = _generate(1.8)
check("locally typical changes what a real generate() draws",
      _generate(1.8, typical_p=0.2) != _plain_hot)
_plain_cold = _generate(0.3)
check("min-p changes what a real generate() draws, on a distribution it can bite",
      _generate(0.3, min_p=0.5) != _plain_cold)
check("two different schemes do not draw the same tokens",
      _generate(0.3, min_p=0.5) != _generate(0.3, typical_p=0.2))

# --- the log must name every condition, not just the first ------------------
# One process runs every arm of a suite in turn. A diagnostic that reports once
# per process names the arm nobody is worried about and stays silent for the
# seven that are.
import io, contextlib  # noqa: E402

_m2 = Tiny()
_out = io.StringIO()
with contextlib.redirect_stdout(_out):
    for kw in ({}, {"min_p": 0.05}, {"typical_p": 0.2}, {"min_p": 0.05}):
        _generate.__globals__["_m"] = _m2
        torch.manual_seed(1)
        chat = _m2.apply_chat_template(_prompt, tokenize=False, add_generation_prompt=True)
        inp = _m2.tokenizer(chat, return_tensors="pt")
        gen = _m2._sampling_kwargs(do_sample=True, temperature=1.8, **kw)
        w = truncation_warper(temperature=1.8, **kw)
        extra = {"logits_processor": LogitsProcessorList([w])} if w else {}
        _m2.generate(_prompt, 5, True, temperature=1.8, **kw)
lines = [ln for ln in _out.getvalue().splitlines() if ln.startswith("decoding:")]
check("each distinct decoding setting is reported once",
      len(lines) == 3, f"{len(lines)} lines for 3 distinct settings of 4 calls")
check("and the reports say which scheme was applied",
      sum("min_p=0.05" in ln for ln in lines) == 1
      and sum("typical_p=0.2" in ln for ln in lines) == 1
      and sum("applied here: none" in ln for ln in lines) == 1,
      " | ".join(ln.split("applied here: ")[-1] for ln in lines))

print()
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
    sys.exit(1)
print("ALL TESTS PASS")
