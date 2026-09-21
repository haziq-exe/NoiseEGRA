"""The random perturbation, sized by how far it moves the model's predictions.

    python tests/test_random_fisher.py

Checks that the per-story subspace is random, reproducible from the seed and
clear of the rule directions; that the calibration hits the distance it is asked
for and asks a longer offset for a larger distance; that measuring leaves the
plan's state as it found it; and that a story generates end to end with it.
"""
import math, sys, warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))

from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402
from noiseegra.fisher_calibration import (  # noqa: E402
    _logits, calibrate_offset, offset_distance, reference_passage, set_offset_length)
from noiseegra.setup_experiment import ExperimentSpec, _ortho_tag  # noqa: E402

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


DIM, LAYERS, RANK = 64, [2, 3], 8
NAMES = ["present_tense", "mature_register", "dialogue"]
torch.manual_seed(0)
VECS = {n: {l: torch.randn(DIM) for l in LAYERS} for n in NAMES}


def plan(**kw):
    kw.setdefault("noise_mode", "none")
    kw.setdefault("noise_alpha", 0.0)
    return SteeringPlan.build(
        VECS, LAYERS, [ConstraintSpec(n, beta=1.0) for n in NAMES], rms_scale=1.0,
        offset_gamma=kw.pop("size", 1.0), offset_mode="orth", offset_norm="fisher",
        offset_basis_kind="random", offset_random_rank=RANK, noise_beta=2.0,
        offset_prefill=True, steer_prefill=True, prompt_tail_clear=2,
        steer_budget=1.0, **kw)


print("== the per-story subspace ==")
p = plan()
torch.manual_seed(11); p.resample_offset()
lp = p.layer_plans[LAYERS[0]]
q1 = lp.offset_basis.clone()
check("a subspace is drawn", q1 is not None and q1.shape[0] == DIM and q1.shape[1] >= RANK - 3,
      f"shape {tuple(q1.shape)}")
check("its columns are orthonormal",
      torch.allclose(q1.t() @ q1, torch.eye(q1.shape[1]), atol=1e-4))
check("it is clear of the rule directions",
      float((lp.protect.t() @ q1).abs().max()) < 1e-4)
check("the offset lies in it",
      float((lp.offset - q1 @ (q1.t() @ lp.offset)).norm()) < 1e-3 * float(lp.offset.norm()))
torch.manual_seed(11); p.resample_offset()
check("the same seed draws the same subspace",
      torch.allclose(p.layer_plans[LAYERS[0]].offset_basis, q1))
torch.manual_seed(12); p.resample_offset()
q2 = p.layer_plans[LAYERS[0]].offset_basis
check("another seed draws another", float((q1.t() @ q2).norm()) < 0.95 * math.sqrt(q1.shape[1]))
check("it wanders over the story", lp.offset_traj is not None)

tag = _ortho_tag("M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=p))
sizes = {_ortho_tag("M", ExperimentSpec(use_orthogonal_steering=True,
                                          steering_plan=plan(size=x)))
         for x in (0.5, 1.0, 2.0)}
check("three sizes get three names", len(sizes) == 3)
check("the run name records the random subspace and the unit",
      "__rr8" in tag and "__onfisher" in tag and "__obrandom" in tag, tag[-80:])

print("\n== the calibration, on a 64-wide model ==")
from tiny_model import Tiny  # noqa: E402

egra = Tiny()
ids = torch.tensor([[1] + [2 + (b % 250) for b in b"<s>write a story<a>"]])
p = plan(size=0.5)
torch.manual_seed(3); p.resample_offset()
passage = reference_passage(egra, p, ids, n_tokens=6)
check("the reference passage extends the prompt", passage.shape[-1] == ids.shape[-1] + 6)
again = reference_passage(egra, p, ids, n_tokens=6)
check("and is the same every time", torch.equal(passage, again))

p._jitter_step = 5
a = _logits(egra, p, passage, ids.shape[-1], with_offset=False)
b = _logits(egra, p, passage, ids.shape[-1], with_offset=False)
check("a measurement is deterministic", torch.allclose(a, b))
check("and leaves the plan's step counter alone", p._jitter_step == 5)

set_offset_length(p, 0.0)
check("no offset moves nothing",
      offset_distance(egra, p, passage, ids.shape[-1]) < 1e-6)

maxlen = 1.0 * math.sqrt(DIM)
cache = {}
got = calibrate_offset(egra, p, ids, 0.5, max_length=maxlen, n_tokens=6, cache=cache)
check("the unit is positive", got["unit"] > 0, f"{got['unit']:.4f}")
if got["reached"]:
    check("the distance asked for is hit", abs(got["distance"] - 0.5) < 0.1,
          f"asked 0.5, got {got['distance']:.3f}")
else:
    check("an unreachable size is reported, not hidden", got["length"] == maxlen)
check("the offset is left at the length found",
      abs(float(p.layer_plans[LAYERS[0]].offset.norm()) - got["length"]) < 1e-3)
check("the passage is cached per prompt", len(cache) == 1)
small = got["length"]
big = calibrate_offset(egra, p, ids, 1.5, max_length=maxlen, n_tokens=6, cache=cache)
check("a larger distance asks for a longer offset", big["length"] >= small,
      f"{small:.3f} then {big['length']:.3f}")

print("\n== a story, end to end ==")
egra.tokenizer.decode = lambda ids, skip_special_tokens=True: " ".join(str(int(i)) for i in ids)
PROMPT = [{"role": "system", "content": "s"}, {"role": "user", "content": "write a story"}]
p = plan(size=0.5)
outs = {egra.generate_with_orthogonal_steering(PROMPT, p, max_new_tokens=8, seed=s)
        for s in range(3)}
check("it generates", all(isinstance(o, str) and o for o in outs))
check("every story is calibrated", len(getattr(p, "fisher_log", []) or []) == 3)

print("\n== the variants ==")
pt = plan(size=1.0)
pt.noise_beta = 0.0
torch.manual_seed(5); pt.resample_offset()
lpt = pt.layer_plans[LAYERS[0]]
d0, d1, d2 = (lpt.offset_at(t) for t in (0, 1, 2))
cos = lambda a, b: float(a @ b / (a.norm() * b.norm()))
check("per-token noise points somewhere new at every step",
      abs(cos(d0, d1)) < 0.9 and abs(cos(d1, d2)) < 0.9,
      f"cosines {cos(d0, d1):.2f}, {cos(d1, d2):.2f}")
check("and keeps its length", abs(float(d0.norm()) - float(d2.norm())) < 1e-4)
dr = plan(size=1.0)
torch.manual_seed(5); dr.resample_offset()
ldr = dr.layer_plans[LAYERS[0]]
check("per-story noise barely moves from one step to the next",
      cos(ldr.offset_at(0), ldr.offset_at(1)) > 0.95,
      f"cosine {cos(ldr.offset_at(0), ldr.offset_at(1)):.3f}")

fx = SteeringPlan.build(
    VECS, LAYERS, [ConstraintSpec(n, beta=1.0) for n in NAMES], rms_scale=2.0,
    offset_gamma=0.4, offset_mode="orth", offset_norm="energy",
    offset_basis_kind="random", offset_random_rank=RANK, noise_beta=2.0,
    offset_envelope="decay", offset_envelope_steps=260,
    noise_mode="none", noise_alpha=0.0, steer_budget=1.0)
torch.manual_seed(5); fx.resample_offset()
got = float(fx.layer_plans[LAYERS[0]].offset.norm())
check("a fixed size is the RMS multiple per coordinate",
      abs(got - 0.4 * 2.0 * math.sqrt(DIM)) < 1e-3, f"{got:.3f}")
check("the fade starts at full strength", abs(fx.envelope_at(0) - 1.0) < 1e-9)
check("is at half by the middle", abs(fx.envelope_at(130) - 0.5) < 1e-6)
check("and is gone at the length limit", fx.envelope_at(260) < 1e-9 and fx.envelope_at(400) < 1e-9)
tags = {_ortho_tag("M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=x))
        for x in (fx,)}
check("the fade's length is in the run name", any("__envdecay260" in t for t in tags))

print("\n== a rise does not inflate the size ==")
def calibrated(envelope):
    q = plan(size=0.5)
    q.offset_envelope, q.offset_envelope_steps = envelope, 260
    torch.manual_seed(3); q.resample_offset()
    out = calibrate_offset(egra, q, ids, 0.15, max_length=maxlen, n_tokens=6)
    check(f"the envelope is restored after measuring ({envelope})",
          q.offset_envelope == envelope)
    check(f"the size is reachable here ({envelope})", out["reached"] == 1.0,
          f"length {out['length']:.3f}")
    return out["length"]
flat_len, rise_len = calibrated("flat"), calibrated("rise")
check("a rise is sized as if at full strength", abs(flat_len - rise_len) < 1e-6,
      f"{flat_len:.3f} vs {rise_len:.3f}")

print("\n== where the noise lands ==")
from noiseegra.leakage import rule_leakage  # noqa: E402
q = plan(size=0.5)
torch.manual_seed(3); q.resample_offset()
pas = reference_passage(egra, q, ids, n_tokens=6)
set_offset_length(q, 2.0)
lk = rule_leakage(egra, q, pas, ids.shape[-1], NAMES)
check("every steered layer is measured", sorted(lk) == LAYERS, str(sorted(lk)))
first = lk[LAYERS[0]]
check("at the first layer the noise has no part along the rules",
      first["share"] < 1e-4, f"{first['share']:.2e}")
later = lk[LAYERS[-1]]
check("downstream some of it can", 0.0 <= later["share"] <= 1.0 and later["share"] > 1e-4,
      f"{later['share']:.3f} against {later['chance']:.3f} by chance")
check("each rule is reported", set(later["per_rule"]) == set(NAMES))
set_offset_length(q, 0.0)
z = rule_leakage(egra, q, pas, ids.shape[-1], NAMES)
check("no noise, no leak", all(v["share"] == 0.0 or v["vs_push"] < 1e-6 for v in z.values()))

print("\n== the shadow copy ==")
def shadow_plan(on, size=0.5):
    q = plan(size=size)
    q.shadow_protect = on
    return q
for on in (False, True):
    q = shadow_plan(on)
    torch.manual_seed(3); q.resample_offset()
    pas = reference_passage(egra, q, ids, n_tokens=6)
    set_offset_length(q, 2.0)
    lk = rule_leakage(egra, q, pas, ids.shape[-1], NAMES)
    worst = max(v["share"] for v in lk.values())
    if on:
        check("with the shadow nothing reaches the protected directions at any steered layer",
              worst < 1e-3, f"largest share {worst:.2e}")
        moved = offset_distance(egra, q, pas, ids.shape[-1])
        check("and the noise still changes the predictions", moved > 1e-3, f"{moved:.3f}")
    else:
        check("without it some does (the leak being fixed)", worst > 1e-3, f"{worst:.3f}")
q = shadow_plan(True)
from noiseegra.setup_experiment import ExperimentSpec as _ES  # noqa: E402
check("the run name records the shadow",
      "__shadow" in _ortho_tag("M", _ES(use_orthogonal_steering=True, steering_plan=q)))
outs = [egra.generate_with_orthogonal_steering(PROMPT, q, max_new_tokens=10, seed=s)
        for s in range(3)]
check("a story generates with the shadow", all(isinstance(o, str) and o for o in outs))
check("the shadow never leaves the story's words",
      getattr(q, "shadow_drift", None) == 0, f"drift {getattr(q, 'shadow_drift', None)}")
check("each story is still sized", len(q.fisher_log) == 3)
plain = shadow_plan(False)
a = egra.generate_with_orthogonal_steering(PROMPT, plain, max_new_tokens=10, seed=4)
b = egra.generate_with_orthogonal_steering(PROMPT, shadow_plan(True), max_new_tokens=10, seed=4)
check("a plan without the shadow is untouched by it", getattr(plain, "shadow_drift", None) is None)
steer_only = plan(size=0.0)
steer_only.shadow_protect = True
c = egra.generate_with_orthogonal_steering(PROMPT, steer_only, max_new_tokens=10, seed=4)
check("with nothing to protect against, no shadow runs",
      getattr(steer_only, "shadow_drift", None) is None)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all passed")
