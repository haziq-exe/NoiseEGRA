"""The per-story noise sized while the story is written.

    python tests/test_online_sizer.py

Checks that the controller grows the noise when it moves the predictions less
than the decoder would and shrinks it when it moves them more, within its step
and range limits, and holds still when the two agree; that its decoder
distribution is the Fisher calibration's; that the plan applies the gain it sets
to the noise while writing and resets it for every story; and that stories
generate end to end with it, with no calibration before them.
"""
import math, sys, types, warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))

from noiseegra.fisher import fisher_rao_distance  # noqa: E402
from noiseegra.fisher_calibration import _probs  # noqa: E402
from noiseegra.online_calibration import OnlineSizer, sampler_probs  # noqa: E402
from noiseegra.run_labels import label_run  # noqa: E402
from noiseegra.setup_experiment import ExperimentSpec, _ortho_tag  # noqa: E402
from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


print("== the controller ==")
torch.manual_seed(0)
V = 50
clean = torch.randn(V) * 2.0


def run(story, steps, **kw):
    plan = types.SimpleNamespace(online_gain=1.0)
    s = OnlineSizer(plan, kw.pop("target", 1.0), **kw)
    gains = []
    for _ in range(steps):
        out = s(None, torch.stack([story, clean]))
        gains.append(plan.online_gain)
    return s, gains, out


near = clean + 0.01 * torch.randn(V)       # barely moved
s, g, out = run(near, 12, warmup=2)
check("a noise that moves things too little is grown", g[-1] > 1.0, f"gain {g[-1]:.3f}")
check("not before the warm-up is over", g[0] == 1.0 and g[1] == 1.0)
check("by at most the step limit each time",
      all(b / a <= 1.25 + 1e-9 for a, b in zip(g, g[1:])))
s, g, _ = run(near, 200, warmup=2)
check("and never past the top of its range", abs(g[-1] - 2.5) < 1e-9, f"gain {g[-1]:.3f}")

far = torch.randn(V) * 2.0                 # a different distribution entirely
s, g, _ = run(far, 200, warmup=2)
check("a noise that moves things too far is shrunk to the bottom of its range",
      abs(g[-1] - 0.25) < 1e-9, f"gain {g[-1]:.3f}")

pa = torch.softmax(clean, -1)
d = float(fisher_rao_distance(pa, torch.softmax(near, -1)))
u = float(fisher_rao_distance(pa, sampler_probs(clean.view(1, -1), 1.8, 0.95)[0]))
s, g, _ = run(near, 20, target=d / u)
check("when the two agree the size holds", max(abs(x - 1.0) for x in g) < 1e-9)
sm = s.summary()
check("and the story is reported as reaching the target",
      abs(sm["achieved"] - d / u) < 1e-9, f"{sm['achieved']:.4f} vs {d / u:.4f}")
x = torch.stack([near, clean])
check("the scores come back as they went in", OnlineSizer(
    types.SimpleNamespace(online_gain=1.0), 1.0)(None, x) is x)
one = types.SimpleNamespace(online_gain=1.0)
t1 = OnlineSizer(one, 1.0)
t1(None, clean.view(1, -1))
check("with no shadow row it does nothing", one.online_gain == 1.0 and not t1.history)

logits = torch.randn(3, V)
check("its decoder is the one the calibration measures against",
      torch.allclose(sampler_probs(logits[1:2], 1.8, 0.95)[0],
                     _probs(logits, 2, temperature=1.8, top_p=0.95)[0], atol=1e-6))

print("\n== the plan ==")
DIM, LAYERS, RANK = 64, [2, 3], 8
NAMES = ["present_tense", "mature_register", "dialogue"]
torch.manual_seed(0)
VECS = {n: {l: torch.randn(DIM) for l in LAYERS} for n in NAMES}


def plan(online=1.0, gamma=0.15):
    return SteeringPlan.build(
        VECS, LAYERS, [ConstraintSpec(n, beta=1.0) for n in NAMES], rms_scale=1.0,
        noise_mode="none", noise_alpha=0.0, steer_budget=1.0,
        offset_gamma=gamma, offset_mode="orth", offset_norm="energy",
        offset_basis_kind="random", offset_random_rank=RANK, noise_beta=2.0,
        offset_prefill=True, steer_prefill=True, prompt_tail_clear=2,
        offset_prefill_gain=1.5, offset_online=online)


p = plan()
torch.manual_seed(5); p.resample_offset()
L = LAYERS[0]
base = p.delta_for(L, 3, with_offset=False)
one = p.delta_for(L, 3) - base
p.online_gain = 2.0
two = p.delta_for(L, 3) - base
check("the noise while writing is written at the gain the controller set",
      torch.allclose(two, 2.0 * one, atol=1e-5))
check("the steering is not", torch.allclose(p.delta_for(L, 3, with_offset=False), base))
p.resample_offset()
check("every story starts again from the starting length", p.online_gain == 1.0)
rid = _ortho_tag("M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=p))
check("the run name records it", "__online1" in rid, rid[-60:])
check("and the label says so", "sized while writing" in label_run(rid).text,
      label_run(rid).text)

print("\n== a story ==")
from tiny_model import Tiny  # noqa: E402
egra = Tiny()
PROMPT = [{"role": "user", "content": "write a story"}]
q = plan()
outs = [egra.generate_with_orthogonal_steering(PROMPT, q, max_new_tokens=16, seed=s)
        for s in range(3)]
check("stories generate", all(isinstance(o, str) and o for o in outs))
check("the noise-free copy writes the story's words", getattr(q, "shadow_drift", None) == 0,
      f"drift {getattr(q, 'shadow_drift', None)}")
logs = getattr(q, "online_log", None) or []
check("each story reports its sizing", len(logs) == 3 and all(x.get("steps", 0) >= 10 for x in logs))
check("and says how far the noise moved things",
      all(math.isfinite(x["achieved"]) and x["achieved"] > 0 for x in logs),
      ", ".join(f"{x['achieved']:.2f}" for x in logs))
check("the controller changed the size", any(abs(x["final_gain"] - 1.0) > 1e-6 for x in logs),
      ", ".join(f"{x['final_gain']:.2f}" for x in logs))
check("with nothing sized before the story", getattr(q, "fisher_log", None) is None)
check("from the starting length", all(abs(x["start_length"] - 0.15 * math.sqrt(DIM)) < 1e-4
                                      for x in logs), f"{logs[0]['start_length']:.4f}")
off = plan(online=0.0)
egra.generate_with_orthogonal_steering(PROMPT, off, max_new_tokens=8, seed=1)
check("a fixed-length plan runs no copy and no controller",
      getattr(off, "shadow_drift", None) is None and getattr(off, "online_log", None) is None)
still = plan(gamma=0.0)
egra.generate_with_orthogonal_steering(PROMPT, still, max_new_tokens=8, seed=1)
check("with no noise to size, none runs", getattr(still, "online_log", None) is None)

print("\n== a wider range, and the size carried between stories ==")
def plan2(**kw):
    return SteeringPlan.build(
        VECS, LAYERS, [ConstraintSpec(n, beta=1.0) for n in NAMES], rms_scale=1.0,
        noise_mode="none", noise_alpha=0.0, steer_budget=1.0,
        offset_gamma=0.15, offset_mode="orth", offset_norm="energy",
        offset_basis_kind="random", offset_random_rank=RANK, noise_beta=2.0,
        offset_prefill=True, steer_prefill=True, prompt_tail_clear=2,
        offset_prefill_gain=1.5, offset_online=1.0, **kw)
d = plan2()
rid0 = _ortho_tag("M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=d))
check("the default keeps the run id every earlier arm has", "__online1__" in rid0
      and "max" not in rid0.split("__online1")[1][:6] and "carry" not in rid0, rid0[-50:])
w = plan2(online_max_gain=10.0, online_carry=True)
rid1 = _ortho_tag("M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=w))
check("the wider range and the carry are in the run id", "__online1max10carry" in rid1, rid1[-60:])
seen_bounds = {}
import noiseegra.online_calibration as OC
orig_init = OC.OnlineSizer.__init__
def spy_init(self, plan, target, **kw):
    seen_bounds["b"] = kw.get("bounds"); orig_init(self, plan, target, **kw)
OC.OnlineSizer.__init__ = spy_init
egra.generate_with_orthogonal_steering(PROMPT, w, max_new_tokens=12, seed=1)
OC.OnlineSizer.__init__ = orig_init
check("the controller is given the wider range", seen_bounds.get("b") == (0.25, 10.0), str(seen_bounds))
first = w.online_log[-1]
check("a story records the size it hands on", abs(first["carried_into_next"] - first["late_gain"]) < 1e-9)
torch.manual_seed(7); w.resample_offset()
L0 = next(lp.offset_length for lp in w.layer_plans.values() if lp.offset is not None)
base = 0.15 * math.sqrt(DIM)
check("the next story starts at the carried size, prompt included",
      abs(L0 - base * first["carried_into_next"]) < 1e-3 and w.online_gain == 1.0,
      f"{L0:.3f} vs {base * first['carried_into_next']:.3f}")
nc = plan2(online_max_gain=10.0)
egra.generate_with_orthogonal_steering(PROMPT, nc, max_new_tokens=12, seed=1)
torch.manual_seed(7); nc.resample_offset()
L1 = next(lp.offset_length for lp in nc.layer_plans.values() if lp.offset is not None)
check("without the carry every story starts at the nominal length", abs(L1 - base) < 1e-3)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all passed")
