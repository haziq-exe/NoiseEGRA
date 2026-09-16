"""f(S_c): perturbing the constraint vector itself instead of adding noise beside it.

Everything here runs on CPU against a randomly initialised 64-wide model.
No downloads, no GPU.  python tests/test_jitter.py
"""

import math, sys, warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from noiseegra.run_labels import label_run  # noqa: E402
from noiseegra.setup_experiment import _spec_to_run_id, make_specs  # noqa: E402
from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


DIM, LAYERS = 64, [2, 3, 4]
NAMES = ("present_tense", "simple_register", "dialogue", "terse", "varied_openers")
torch.manual_seed(0)
VECS = {n: {l: torch.randn(DIM) for l in LAYERS} for n in NAMES}
SPECS = [ConstraintSpec(n) for n in NAMES]
RMS = 2.0
L = LAYERS[0]


def plan(**kw):
    kw.setdefault("noise_mode", "none")
    kw.setdefault("noise_alpha", 0.0)
    return SteeringPlan.build(VECS, LAYERS, SPECS, rms_scale=RMS, **kw)


plain = plan()
plain.resample_offset()
S = plain.delta_for(L, 0)

print("== the unjittered constraint vector ==")
check("steering alone is the sum of the constraint directions", S is not None and S.shape == (DIM,))
check("kappa=0 is a no-op",
      torch.allclose(plan(jitter_mode="perp", jitter_kappa=0.0).delta_for(L, 0), S, atol=1e-6))
check("an unknown mode is rejected", _raises(lambda: plan(jitter_mode="sideways", jitter_kappa=1.0)))
check("an unknown draw is rejected", _raises(lambda: plan(jitter_draw="everywhere", jitter_kappa=1.0)))

print("\n== perp: the constraint push survives exactly ==")
KAPPA = 0.15
p = plan(jitter_mode="perp", jitter_kappa=KAPPA)
torch.manual_seed(3); p.resample_offset()
d = p.delta_for(L, 0)
u = S / S.norm()
check("the component along S_c is unchanged",
      abs(float(d @ u) - float(S.norm())) < 1e-4,
      f"{float(d @ u):.6f} vs {float(S.norm()):.6f}")
step = d - S
check("the added step is perpendicular to S_c",
      abs(float(step @ u)) < 1e-4, f"{float(step @ u):.3g}")
want = KAPPA * RMS * math.sqrt(DIM)
check("the step's length is kappa * rms * sqrt(dim), the same scale gamma uses",
      abs(float(step.norm()) - want) < 1e-3, f"{float(step.norm()):.4f} vs {want:.4f}")

# The whole point: unlike the protected-subspace offset, the step is allowed to
# lie inside the constraint subspace. It only has to leave the *summed* push alone.
protect = p.layer_plans[L].protect
inside = float(((protect.t() @ step) ** 2).sum() / (step @ step))
check("the step is free to lie inside the constraint subspace",
      inside > 1.0 / DIM, f"{inside:.4f} of its energy is in the protected span")

print("\n== rotate: the dose is fixed, only the aim moves ==")
for k in (0.5, 1.0, 2.0):
    r = plan(jitter_mode="rotate", jitter_kappa=k)
    torch.manual_seed(5); r.resample_offset()
    d = r.delta_for(L, 0)
    cos = float((d @ S) / (d.norm() * S.norm()))
    check(f"kappa={k}: the length is unchanged",
          abs(float(d.norm()) - float(S.norm())) < 1e-4,
          f"{float(d.norm()):.5f} vs {float(S.norm()):.5f}")
    check(f"kappa={k}: the turn is atan(kappa) = {math.degrees(math.atan(k)):.0f} degrees",
          abs(cos - 1.0 / math.sqrt(1 + k * k)) < 1e-4,
          f"cos {cos:.5f} vs {1 / math.sqrt(1 + k * k):.5f}")

print("\n== gain: nothing leaves the constraint span ==")
g = plan(jitter_mode="gain", jitter_kappa=0.5)
basis = g.layer_plans[L].basis
seen = []
for sd in range(200):
    torch.manual_seed(sd); g.resample_offset()
    seen.append(list(g.gains))
d = g.delta_for(L, 0)
resid = d - basis @ (basis.t() @ d)
check("the jittered vector still lies in the span of the constraint directions",
      float(resid.norm()) < 1e-4, f"residual {float(resid.norm()):.3g}")
flat = [x for row in seen for x in row]
check("every gain is positive, so no constraint is ever pushed backwards",
      min(flat) > 0, f"min {min(flat):.4f}")
mean = sum(flat) / len(flat)
check("the gains average 1, so the expected constraint push is unchanged",
      abs(mean - 1.0) < 0.1, f"mean {mean:.4f}")
check("the gains actually vary", max(flat) / min(flat) > 2, f"{min(flat):.3f}-{max(flat):.3f}")

print("\n== perturbing the directions before they are made orthogonal ==")
# Everything above perturbs the summed constraint vector after the five
# directions have been made mutually orthogonal, so the frame is identical for
# every story. This perturbs each direction first and orthogonalises the
# perturbed set, giving every story its own frame.
f = plan(jitter_mode="frame", jitter_kappa=0.3)
torch.manual_seed(21); f.resample_offset()
b1 = f.layer_plans[L].basis.clone()
torch.manual_seed(22); f.resample_offset()
b2 = f.layer_plans[L].basis.clone()
check("every story gets a different frame", not torch.allclose(b1, b2, atol=1e-5))
check("the frame is still orthonormal",
      float((b1.t() @ b1 - torch.eye(len(NAMES))).abs().max()) < 1e-4,
      f"{float((b1.t() @ b1 - torch.eye(len(NAMES))).abs().max()):.3g}")
check("the summed push is turned away from the unjittered one",
      abs(float((f.delta_for(L, 0) @ S) / (f.delta_for(L, 0).norm() * S.norm()))) < 0.999)
check("and is constant within a story",
      torch.allclose(f.delta_for(L, 0), f.delta_for(L, 9), atol=1e-6))
check("kappa=0 leaves the frame alone",
      torch.allclose(plan(jitter_mode="frame", jitter_kappa=0.0).layer_plans[L].basis,
                     plain.layer_plans[L].basis, atol=1e-6))
check("the raw directions are kept so the frame can be rebuilt each story",
      f.layer_plans[L].raw_basis is not None
      and f.layer_plans[L].raw_basis.shape == (DIM, len(NAMES)))

print("\n== one draw per story, held for the whole story ==")
p = plan(jitter_mode="perp", jitter_kappa=0.3)
torch.manual_seed(11); p.resample_offset()
steps = [p.delta_for(L, t) for t in (0, 1, 7, 99)]
check("the perturbation is the same at every decode step",
      all(torch.allclose(steps[0], s, atol=1e-6) for s in steps[1:]))
torch.manual_seed(12); p.resample_offset()
check("a different story gets a different one",
      not torch.allclose(steps[0], p.delta_for(L, 0), atol=1e-5))
check("every layer draws its own",
      not torch.allclose(p.layer_plans[LAYERS[0]].jitter, p.layer_plans[LAYERS[1]].jitter))

print("\n== drawing the step inside the activation subspace ==")
sub = {l: torch.linalg.qr(torch.randn(DIM, 8))[0] for l in LAYERS}
b = plan(jitter_mode="perp", jitter_kappa=0.3, jitter_draw="basis", offset_basis=sub)
torch.manual_seed(13); b.resample_offset()
j = b.layer_plans[L].jitter
q = sub[L]
check("the draw lies in the activation subspace",
      float((j - q @ (q.t() @ j)).norm()) < 1e-5)
check("a basis draw with no basis supplied is refused",
      _raises(lambda: plan(jitter_mode="perp", jitter_kappa=0.3, jitter_draw="basis")))
# The basis is handed over unprojected: the constraint subspace is deliberately
# left in, because f(S_c) keeps the push intact by being perpendicular to S_c,
# not by avoiding the subspace.
check("the constraint directions are not stripped out of it",
      float((b.layer_plans[L].protect.t() @ q).abs().max()) > 1e-3)

print("\n== where the constraint vector is applied ==")
d_only = plan(jitter_mode="perp", jitter_kappa=0.3, steer_decode=False)
torch.manual_seed(17); d_only.resample_offset()
check("steer_decode=False adds nothing at a decode step", d_only.delta_for(L, 0) is None)
check("but the prompt still gets the jittered vector",
      d_only.steering_only(L, 0) is not None)
both = plan(jitter_mode="perp", jitter_kappa=0.3)
torch.manual_seed(17); both.resample_offset()
check("and it is the same vector the decode arm would have used",
      torch.allclose(d_only.steering_only(L, 0), both.delta_for(L, 0), atol=1e-6))

print("\n== noise and the offset still work alongside it ==")
n = plan(jitter_mode="rotate", jitter_kappa=1.0, noise_mode="orth", noise_alpha=0.4)
torch.manual_seed(19); n.resample_offset()
check("per-token noise is still added on top", n.delta_for(L, 0) is not None)
a, bb = n.delta_for(L, 0), n.delta_for(L, 1)   # consecutive draws, no reseeding
check("and it is redrawn every step", not torch.allclose(a, bb, atol=1e-5))
jit = n.layer_plans[L].jitter.clone()
n.delta_for(L, 2)
check("while f(S_c) stays fixed underneath it",
      torch.allclose(jit, n.layer_plans[L].jitter, atol=1e-8))

print("\n== run ids and labels ==")
arms = [
    plan(),
    plan(steer_decode=False, steer_prefill=True),
    plan(noise_mode="orth", noise_alpha=0.4),
    plan(noise_mode="orth", noise_alpha=0.4, noise_schedule="cosine_decay", noise_horizon=64),
    plan(offset_gamma=0.15, offset_mode="orth", offset_basis=sub, offset_basis_kind="story"),
    plan(offset_gamma=0.15, offset_mode="orth", offset_basis=sub, offset_basis_kind="story",
         offset_prefill=True, offset_decode=False),
]
for mode, k in (("perp", 0.15), ("rotate", 1.0), ("gain", 0.5)):
    arms.append(plan(jitter_mode=mode, jitter_kappa=k))
    arms.append(plan(jitter_mode=mode, jitter_kappa=k, steer_decode=False, steer_prefill=True))
ids = [_spec_to_run_id("Tiny", s) for s in make_specs(*[{"plan": a} for a in arms])]
check("every arm of the main suite gets its own run id",
      len(set(ids)) == len(ids), f"{len(set(ids))} of {len(ids)}")
labels = [label_run(i).text for i in ids]
check("and its own readable label", len(set(labels)) == len(labels), "\n     ".join(labels))
for txt in labels:
    print(f"     {txt}")

print("\n== end to end on a 64-wide model ==")
from tiny_model import Tiny  # noqa: E402

egra = Tiny()
# The stub decoder returns a fixed sentence whose length is the token count, so two
# different token streams of the same length decode to the same string. Report the
# ids instead, otherwise this only tests whether the perturbation changed where the
# model stopped.
egra.tokenizer.decode = lambda ids, skip_special_tokens=True: " ".join(str(int(i)) for i in ids)
PROMPT = [{"role": "system", "content": "s"}, {"role": "user", "content": "write a story"}]
vecs = {n: {l: torch.randn(DIM) for l in LAYERS} for n in NAMES}


def tiny(**kw):
    kw.setdefault("noise_mode", "none")
    kw.setdefault("noise_alpha", 0.0)
    return SteeringPlan.build(vecs, LAYERS, [ConstraintSpec(n, beta=8.0) for n in NAMES],
                              rms_scale=1.0, **kw)


base = egra.generate_with_orthogonal_steering(
    PROMPT, tiny(), max_new_tokens=12, seed=7)
rot = egra.generate_with_orthogonal_steering(
    PROMPT, tiny(jitter_mode="rotate", jitter_kappa=2.0), max_new_tokens=12, seed=7)
pro = egra.generate_with_orthogonal_steering(
    PROMPT, tiny(jitter_mode="rotate", jitter_kappa=2.0, steer_decode=False,
                  steer_prefill=True), max_new_tokens=12, seed=7)
check("turning the constraint vector changes what is written", rot != base, f"{base!r} vs {rot!r}")
check("applying it at the prompt only changes it differently", pro not in (base, rot),
      f"{pro!r}")
outs = {egra.generate_with_orthogonal_steering(
    PROMPT, tiny(jitter_mode="rotate", jitter_kappa=2.0), max_new_tokens=12, seed=s)
    for s in range(6)}
check("different seeds give different stories", len(outs) > 1, f"{len(outs)} of 6 distinct")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("all f(S_c) tests passed")
