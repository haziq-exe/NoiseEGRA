"""The noise's target set from the model before any story.

    python tests/test_target_rule.py
"""
import math, sys, warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))

from noiseegra.online_calibration import measure_for_rule, target_from_top_share  # noqa: E402
from noiseegra.setup_experiment import ExperimentSpec, _ortho_tag  # noqa: E402
from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


print("== the rule ==")
q = target_from_top_share(0.6854, 0.762)
l = target_from_top_share(1.3450, 0.678)
check("it gives Qwen3-1.7B's best target", abs(q - 1.0) < 0.03, f"{q:.3f}")
check("and Llama-3.2-3B's", abs(l - 0.43) < 0.02, f"{l:.3f}")
check("a less certain model gets a smaller target",
      target_from_top_share(1.0, 0.5) < target_from_top_share(1.0, 0.8))
check("so does one top-p moves further", target_from_top_share(2.0, 0.7) < target_from_top_share(1.0, 0.7))
d = q * 0.6854
check("the mass the noise may move is k of the top word's probability",
      abs(math.sin(d / 2) - 0.43 * 0.762) < 1e-9)

print("\n== in a run ==")
from tiny_model import Tiny  # noqa: E402
egra = Tiny()
torch.manual_seed(0)
V = {n: {l: torch.randn(64) for l in [2, 3]} for n in ["a", "b"]}


def plan(k):
    return SteeringPlan.build(V, [2, 3], [ConstraintSpec(n, beta=1.0) for n in V], rms_scale=1.0,
        noise_mode="none", noise_alpha=0.0, steer_budget=1.0, offset_gamma=0.1, offset_mode="orth",
        offset_norm="energy", offset_basis_kind="random", offset_random_rank=8, noise_beta=2.0,
        offset_prefill=True, steer_prefill=True, prompt_tail_clear=2, offset_online=1.0,
        online_max_gain=10.0, online_rule_k=k)


p = plan(0.43)
rid = _ortho_tag("M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=p))
check("the run id records the rule", "rule0p43" in rid, rid[-50:])
PROMPT = [{"role": "user", "content": "write a story"}]
chat = egra.apply_chat_template(PROMPT, tokenize=False, add_generation_prompt=True)
ids = egra.tokenizer(chat, return_tensors="pt")["input_ids"]
ref = measure_for_rule(egra, plan(0.43), ids)
want = target_from_top_share(ref["unit"], ref["top_prob"], 0.43)
calls = []
import noiseegra.online_calibration as OC
real = OC.measure_for_rule
def counting(*a, **k):
    calls.append(1); return real(*a, **k)
OC.measure_for_rule = counting
outs = [egra.generate_with_orthogonal_steering(PROMPT, p, max_new_tokens=8, seed=s) for s in range(3)]
OC.measure_for_rule = real
check("stories generate with the rule", all(isinstance(o, str) and o for o in outs))
check("the model is measured once per prompt, not per story", len(calls) == 1, str(len(calls)))
check("with no hooks on, so it matches a measurement made outside a run",
      abs(p.offset_online - want) < 1e-6, f"{p.offset_online:.5f} vs {want:.5f}")
off = plan(0.0)
egra.generate_with_orthogonal_steering(PROMPT, off, max_new_tokens=6, seed=1)
check("without the rule the target stays as set", off.offset_online == 1.0
      and getattr(off, "_rule_cache", None) is None)

print("\n== the starting length ==")
from noiseegra.online_calibration import start_for_target  # noqa: E402

ps = plan(0.43)
ps.online_rule_start = True
rid = _ortho_tag("M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=ps))
check("the run id records the measured start", "rule0p43start" in rid, rid[-50:])
st = start_for_target(egra, plan(0.43), ids, want, ref["unit"])
check("the start it finds moves the predictions by the target",
      st["reached"] == 1.0 and abs(st["moves"] - want) < 0.05 * want,
      f"{st['moves']:.4f} vs {want:.4f} at {st['fraction']:.4f} of the norm")
st2 = start_for_target(egra, plan(0.43), ids, 2 * want, ref["unit"])
check("and a larger target needs a longer start", st2["start"] > st["start"])


def first_draw(pl, seed):
    """The story's own noise basis and its length when writing starts."""
    got = {}
    real_gen = egra.model.generate
    def spy(*a, **k):
        lp = next(iter(pl.layer_plans.values()))
        got["basis"] = lp.offset_basis.detach().clone()
        got["length"] = float(lp.offset_length)
        got["target"] = float(pl.offset_online)
        return real_gen(*a, **k)
    egra.model.generate = spy
    try:
        egra.generate_with_orthogonal_steering(PROMPT, pl, max_new_tokens=4, seed=seed)
    finally:
        egra.model.generate = real_gen
    return got


a = first_draw(plan(0.43), 5)
ps = plan(0.43); ps.online_rule_start = True
b = first_draw(ps, 5)
check("measuring the start leaves the story's own noise draw as it was",
      torch.allclose(a["basis"], b["basis"]))
check("the story starts at the measured length", abs(b["length"] - ps._rule_cache[key0]["start"]) < 1e-4
      if (key0 := next(iter(ps._rule_cache))) else False,
      f"{b['length']:.4f} vs {ps._rule_cache[key0]['start']:.4f}")
check("and aims at the rule's target", abs(b["target"] - want) < 1e-6)
calls.clear()
OC.measure_for_rule = counting
first_draw(ps, 6)
OC.measure_for_rule = real
check("a second story reuses the measurement", calls == [])

print("\n== a fixed measured length, no controller ==")


def simple(decode=True):
    return SteeringPlan.build(V, [2, 3], [ConstraintSpec(n, beta=1.0) for n in V], rms_scale=1.0,
        noise_mode="none", noise_alpha=0.0, steer_budget=1.0, offset_gamma=0.1, offset_mode="iso",
        offset_norm="energy", offset_basis_kind="random", offset_random_rank=0, noise_beta=None,
        offset_prefill=True, offset_decode=decode, steer_prefill=True, prompt_tail_clear=2,
        online_rule_k=0.43, offset_measured=True)


sp = simple()
rid_s = _ortho_tag("M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=sp))
rid_p = _ortho_tag("M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=simple(False)))
check("the run id records the measured length", "__meas0p43" in rid_s and "__online" not in rid_s, rid_s[-40:])
check("and a prompt-only arm differs from it", rid_p != rid_s)
rows = []
real_gen = egra.model.generate
def spy_rows(*a, **k):
    ids_ = k.get("input_ids", a[0] if a else None)
    rows.append(int(ids_.shape[0]))
    lp = next(iter(sp.layer_plans.values()))
    rows.append(float(lp.offset_length))
    return real_gen(*a, **k)
egra.model.generate = spy_rows
try:
    out = egra.generate_with_orthogonal_steering(PROMPT, sp, max_new_tokens=6, seed=3)
finally:
    egra.model.generate = real_gen
m = next(iter(sp._rule_cache.values()))
check("it writes one row: no shadow copy", rows[0] == 1, str(rows[0]))
check("at the measured length", abs(rows[1] - m["start"]) < 1e-4, f"{rows[1]:.4f} vs {m['start']:.4f}")
check("with no controller in the log", not getattr(sp, "online_log", None))
check("and the target is the rule's", abs(m["target"] - want) < 1e-6)
lp0 = next(iter(sp.layer_plans.values()))
check("the noise is one constant vector", lp0.offset_traj is None and lp0.offset_basis is None)
pp = simple(False)
egra.generate_with_orthogonal_steering(PROMPT, pp, max_new_tokens=6, seed=3)
mp = next(iter(pp._rule_cache.values()))
check("a prompt-only arm is given the same length", abs(mp["start"] - m["start"]) < 1e-6,
      f"{mp['start']:.4f} vs {m['start']:.4f}")
check("and still writes with the noise off", pp.offset_decode is False)

print("\n== the absolute target ==")
import types  # noqa: E402
from noiseegra.fisher import fisher_rao_distance  # noqa: E402
from noiseegra.online_calibration import OnlineSizer  # noqa: E402
torch.manual_seed(0)
clean = torch.randn(1, 50)
noisy = clean + 0.3 * torch.randn(1, 50)
scores = torch.cat([noisy, clean], 0)
d_now = float(fisher_rao_distance(torch.softmax(clean, -1), torch.softmax(noisy, -1)))
rel, ab = types.SimpleNamespace(online_gain=1.0), types.SimpleNamespace(online_gain=1.0)
s_rel = OnlineSizer(rel, target=6.0, bounds=(0.25, 10.0))
s_abs = OnlineSizer(ab, target=6.0, bounds=(0.25, 10.0), absolute=d_now)
for _ in range(8):
    s_rel(None, scores.clone()); s_abs(None, scores.clone())
check("holding the absolute distance it already has, the gain stays put",
      abs(ab.online_gain - 1.0) < 1e-6, f"{ab.online_gain:.3f}")
check("while a relative target far above it keeps raising the gain", rel.online_gain > 2.0,
      f"{rel.online_gain:.3f}")
pa = plan(0.43); pa.online_rule_start = True; pa.online_absolute = True
rid = _ortho_tag("M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=pa))
check("the run id records the absolute target", "rule0p43startabs" in rid, rid[-40:])
out = egra.generate_with_orthogonal_steering(PROMPT, pa, max_new_tokens=6, seed=2)
check("and a story generates with it", isinstance(out, str))

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all passed")
