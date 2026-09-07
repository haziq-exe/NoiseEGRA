"""Geometry and metric tests for the orthogonal-steering ablation.

Runs on CPU without loading a language model:

    python tests/test_subspace.py
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from noiseegra.subspace import (  # noqa: E402
    ConstraintSpec,
    SteeringPlan,
    constrained_noise,
    cosine_gram,
    orthonormalize,
    schedule_factor,
    split_noise,
    unit_columns,
)

FAILURES = []


def check(name, cond, extra=""):
    status = "PASS" if cond else "FAIL"
    if not cond:
        FAILURES.append(name)
    print(f"  [{status}] {name}{('  ' + extra) if extra else ''}")


def entangled_directions(dim=512, n=3, seed=0):
    """Three directions with deliberately high mutual cosine, like real constraints."""
    g = torch.Generator().manual_seed(seed)
    shared = torch.randn(dim, generator=g)
    cols = [shared + 0.8 * torch.randn(dim, generator=g) for _ in range(n)]
    return torch.stack(cols, dim=1)


# --------------------------------------------------------------------------- #
print("\n== orthonormalisation ==")
D = entangled_directions()
gram = cosine_gram(D)
off = (gram - torch.eye(3)).abs().max().item()
check("test directions are genuinely entangled", off > 0.3, f"max|cos|={off:.3f}")

for method in ("none", "gram_schmidt", "lowdin"):
    B, rep = orthonormalize(D, method=method)
    norms = B.norm(dim=0)
    check(f"{method}: columns are unit norm", torch.allclose(norms, torch.ones(3), atol=1e-5))
    check(
        f"{method}: sign-aligned with the raw direction",
        bool((rep["retained"] > 0).all()),
        f"retained={[round(float(v), 3) for v in rep['retained']]}",
    )
    g2 = B.t() @ B
    max_off = (g2 - torch.eye(3)).abs().max().item()
    if method == "none":
        check("none: columns stay non-orthogonal (naive-sum baseline)", max_off > 0.3,
              f"max|cos|={max_off:.3f}")
    else:
        check(f"{method}: columns are orthonormal", max_off < 1e-4, f"max|cos|={max_off:.2e}")

Bgs, _ = orthonormalize(D, method="gram_schmidt")
check(
    "gram_schmidt: first column is untouched (order dependence)",
    torch.allclose(Bgs[:, 0], unit_columns(D)[:, 0], atol=1e-5),
)

perm = [2, 0, 1]
Blo, _ = orthonormalize(D, method="lowdin")
Blo_perm, _ = orthonormalize(D[:, perm], method="lowdin")
check(
    "lowdin: order-invariant (permuting inputs permutes outputs)",
    torch.allclose(Blo[:, perm], Blo_perm, atol=1e-4),
)
Bgs_perm, _ = orthonormalize(D[:, perm], method="gram_schmidt")
check(
    "gram_schmidt: NOT order-invariant (documented caveat)",
    not torch.allclose(Bgs[:, perm], Bgs_perm, atol=1e-3),
)

_, rep_lo = orthonormalize(D, method="lowdin")
_, rep_gs = orthonormalize(D, method="gram_schmidt")
check(
    "lowdin retains more of every direction than gram_schmidt on average",
    rep_lo["retained"].mean().item() > rep_gs["retained"].mean().item(),
    f"lowdin={rep_lo['retained'].mean():.3f} vs gs={rep_gs['retained'].mean():.3f}",
)

try:
    dup = torch.stack([D[:, 0], D[:, 0], D[:, 1]], dim=1)
    orthonormalize(dup, method="lowdin")
    check("lowdin rejects linearly dependent directions", False)
except ValueError:
    check("lowdin rejects linearly dependent directions", True)


# --------------------------------------------------------------------------- #
print("\n== constrained noise ==")
dim, k = 4096, 24
torch.manual_seed(7)
Q = torch.linalg.qr(torch.randn(dim, k))[0]
g = torch.randn(dim)
sigma = 0.3

iso = constrained_noise(g, sigma, Q, mode="iso")
orth = constrained_noise(g, sigma, Q, mode="orth")
para = constrained_noise(g, sigma, Q, mode="para")

check("orth has ~zero projection onto the protected subspace",
      (orth @ Q).norm().item() < 1e-3, f"|Q^T eps|={(orth @ Q).norm().item():.2e}")
para_res = para - (para @ Q) @ Q.t()
check("para lies inside the protected subspace",
      para_res.norm().item() / para.norm().item() < 1e-5)

iso_e, orth_e, para_e = iso.pow(2).sum(), orth.pow(2).sum(), para.pow(2).sum()
check("norm_match='energy' equalises total energy across arms",
      abs(orth_e / iso_e - 1) < 0.05 and abs(para_e / iso_e - 1) < 0.35,
      f"orth/iso={orth_e/iso_e:.3f}  para/iso={para_e/iso_e:.3f}")

para_raw = constrained_noise(g, sigma, Q, mode="para", norm_match="none")
frac = (para_raw.pow(2).sum() / iso_e).item()
check("without norm matching, para carries only ~k/dim of the energy",
      abs(frac - k / dim) < 3 * (k / dim), f"frac={frac:.5f}  k/dim={k/dim:.5f}")

# The headline dimensionality caveat, measured.
draws = torch.randn(400, dim)
par, _ = split_noise(draws, Q)
share = (par.pow(2).sum(dim=1) / draws.pow(2).sum(dim=1)).mean().item()
check("a random draw already puts only ~k/dim of its energy in the subspace",
      abs(share - k / dim) < 0.2 * (k / dim), f"measured={share:.5f}  k/dim={k/dim:.5f}")

rank1 = Q[:, :1]
o1 = constrained_noise(g, sigma, rank1, mode="orth")
cos = torch.nn.functional.cosine_similarity(o1, iso, dim=0).item()
check("with a rank-1 protected subspace, 'orth' is nearly identical to 'iso'",
      cos > 0.9995, f"cos(orth, iso)={cos:.6f}  <- why protect_rank matters")

check("iso is unaffected by the basis", torch.allclose(iso, g * sigma))
check("mode='none' returns nothing", constrained_noise(g, sigma, Q, mode="none") is None)
check("sigma=0 returns nothing", constrained_noise(g, 0.0, Q, mode="orth") is None)


# --------------------------------------------------------------------------- #
print("\n== schedules ==")
check("constant is flat", all(schedule_factor("constant", t, 200) == 1.0 for t in (0, 50, 500)))
check("cosine_decay: 1 -> 0",
      schedule_factor("cosine_decay", 0, 200) == 1.0
      and abs(schedule_factor("cosine_decay", 200, 200)) < 1e-12)
check("ramp: 0 -> 1",
      schedule_factor("ramp", 0, 200) == 0.0 and schedule_factor("ramp", 200, 200) == 1.0)
check("ramp is monotone increasing",
      all(schedule_factor("ramp", t, 200) <= schedule_factor("ramp", t + 1, 200) for t in range(200)))
check("linear_decay: 1 -> 0",
      schedule_factor("linear_decay", 0, 200) == 1.0
      and schedule_factor("linear_decay", 200, 200) == 0.0)


# --------------------------------------------------------------------------- #
print("\n== SteeringPlan ==")
dim = 256
layers = [3, 4, 5]
names = ["closure", "present_tense", "simple_register"]
gg = torch.Generator().manual_seed(11)
vectors = {
    n: {l: 2.5 * torch.randn(dim, generator=gg) for l in layers} for n in names
}
specs = [
    ConstraintSpec("closure", beta=1.5, schedule="ramp"),
    ConstraintSpec("present_tense", beta=1.0),
    ConstraintSpec("simple_register", beta=0.5),
]
plan = SteeringPlan.build(
    vectors, layers, specs, rms_scale=2.0, orthogonalize="lowdin",
    noise_mode="orth", noise_alpha=0.175, horizon=200,
)
check("plan sees the right dimension and layers", plan.dim == dim and plan.layers == layers)
check("sigma follows the paper's alpha * median(RMS)", abs(plan.sigma - 0.175 * 2.0) < 1e-9)
check("protected rank equals the number of constraints by default", plan.protect_rank == 3)

torch.manual_seed(0)
d1 = plan.delta_for(4, 10)
torch.manual_seed(0)
d2 = plan.delta_for(4, 10)
check("delta is reproducible under a fixed seed", torch.allclose(d1, d2))
check("delta has the residual-stream shape", tuple(d1.shape) == (dim,))

steer_only = plan.delta_for(4, 0, with_noise=False)
lp = plan.layer_plans[4]
coeff = steer_only @ lp.basis
check("at t=0 the ramped closure term is off and the others are on",
      abs(float(coeff[0])) < 1e-4
      and abs(float(coeff[1]) - 1.0 * 2.0) < 1e-3
      and abs(float(coeff[2]) - 0.5 * 2.0) < 1e-3,
      f"coeffs={[round(float(c), 3) for c in coeff]}")

steer_end = plan.delta_for(4, 200, with_noise=False)
coeff_end = steer_end @ lp.basis
check("at t=horizon the ramped closure term reaches beta*rms",
      abs(float(coeff_end[0]) - 1.5 * 2.0) < 1e-3, f"closure coeff={float(coeff_end[0]):.3f}")

plan_extra = SteeringPlan.build(
    vectors, layers, specs, rms_scale=2.0, orthogonalize="lowdin", noise_mode="orth",
    protect_extra={l: torch.randn(dim, 8, generator=gg) for l in layers},
)
check("protect_extra enlarges the protected subspace", plan_extra.protect_rank == 11,
      f"rank={plan_extra.protect_rank}")

plan_none = SteeringPlan.build(vectors, layers, specs, rms_scale=2.0,
                               noise_mode="none", noise_alpha=0.0)
dn = plan_none.delta_for(4, 5)
check("noise_mode='none' leaves a pure steering delta",
      dn is not None and abs(float((dn - plan_none.delta_for(4, 5, with_noise=False)).norm())) < 1e-6)

try:
    SteeringPlan.build(vectors, layers, [ConstraintSpec("nope")], rms_scale=1.0)
    check("missing constraint raises", False)
except KeyError:
    check("missing constraint raises", True)

try:
    SteeringPlan.build(vectors, [99], specs, rms_scale=1.0)
    check("missing layer raises", False)
except KeyError:
    check("missing layer raises", True)

# Every constructor parameter must actually reach the object. A build() argument
# that is quietly dropped falls back to its default and is invisible until a run
# behaves wrong.
import inspect  # noqa: E402
sig = inspect.signature(SteeringPlan.build).parameters
probe = SteeringPlan.build(
    vectors, layers, specs, rms_scale=3.0, orthogonalize="gram_schmidt",
    noise_mode="para", noise_alpha=0.4, noise_norm_match="none",
    noise_schedule="cosine_decay", horizon=77, steer_prefill=True,
    offset_gamma=0.2, offset_mode="free",
)
for attr, want in [("rms_scale", 3.0), ("orthogonalize", "gram_schmidt"),
                   ("noise_mode", "para"), ("noise_alpha", 0.4),
                   ("noise_norm_match", "none"), ("noise_schedule", "cosine_decay"),
                   ("horizon", 77), ("steer_prefill", True),
                   ("offset_gamma", 0.2), ("offset_mode", "free")]:
    check(f"build() forwards {attr}", getattr(probe, attr) == want,
          f"got {getattr(probe, attr)!r}, wanted {want!r}")

print("\n== per-story offsets ==")
flat = [ConstraintSpec(n, beta=1.0, schedule="constant") for n in names]
op = SteeringPlan.build(vectors, layers, flat, rms_scale=2.0, noise_mode="none",
                        noise_alpha=0.0, offset_gamma=0.15, offset_mode="orth")
torch.manual_seed(0); op.resample_offset()
d_early, d_late = op.delta_for(4, 0), op.delta_for(4, 60)
off_a = op.layer_plans[4].offset.clone()
torch.manual_seed(1); op.resample_offset()
off_b = op.layer_plans[4].offset.clone()
check("offset is constant across steps within a story", torch.allclose(d_early, d_late))
check("offset differs between stories", not torch.allclose(off_a, off_b))
check("orth offset has no component in the constraint subspace",
      float((off_a @ op.layer_plans[4].protect).norm()) < 1e-4)

fp = SteeringPlan.build(vectors, layers, flat, rms_scale=2.0, noise_mode="none",
                        noise_alpha=0.0, offset_gamma=0.15, offset_mode="free")
torch.manual_seed(0); fp.resample_offset()
free_off = fp.layer_plans[4].offset
check("free offset does leak into the constraint subspace",
      float((free_off @ fp.layer_plans[4].protect).norm()) > 1e-3,
      f"leak={float((free_off @ fp.layer_plans[4].protect).norm()):.4f}")

# With a supplied activation basis the constraint directions sit inside it, which
# is the realistic case and the one where the projection has to be exact.
pcs = {l: torch.linalg.qr(torch.cat(
    [torch.stack([vectors[n][l] for n in names], 1), torch.randn(dim, 40, generator=gg)], 1))[0]
    for l in layers}
bp = SteeringPlan.build(vectors, layers, flat, rms_scale=2.0, noise_mode="none",
                        noise_alpha=0.0, offset_gamma=0.15, offset_mode="orth",
                        offset_basis=pcs)
lp_b = bp.layer_plans[4]
check("offset basis drops exactly the constraint directions",
      lp_b.offset_basis.shape[1] == pcs[4].shape[1] - len(names),
      f"{lp_b.offset_basis.shape[1]} of {pcs[4].shape[1]}")
worst = 0.0
for sd in range(50):
    torch.manual_seed(sd); bp.resample_offset()
    worst = max(worst, float((lp_b.offset @ lp_b.basis).abs().max()))
check("offset from that basis never touches a constraint direction", worst < 1e-4,
      f"worst leak {worst:.2e}")

none_p = SteeringPlan.build(vectors, layers, flat, rms_scale=2.0, noise_mode="none",
                            noise_alpha=0.0)
none_p.resample_offset()
check("offset_mode='none' leaves no offset", none_p.layer_plans[4].offset is None)

plan.print_report()


# --------------------------------------------------------------------------- #
print("\n== exact constraint metrics ==")
from noiseegra.constraint_metrics import ExactConstraintChecker, tokenize_words  # noqa: E402
from noiseegra.egra_constraint_checker import EGRAConstraintChecker  # noqa: E402

base = EGRAConstraintChecker()
sample = "يذهب سامي إلى المدرسة. يقابل صديقه عند الباب، ويدخلان معاً."
check("word tokenisation matches the published checker",
      tokenize_words(sample) == base._tokenize_words(sample))

ck = ExactConstraintChecker(backend="regex")
short_present = "تلعب مريم في الحديقة. تجد كرة صغيرة. تعيدها إلى صديقتها."
m = ck.evaluate(short_present)
check("a short present-tense story passes all three",
      m.length_ok and m.present_tense_ok and m.simple_register_ok and m.violations == 0,
      f"words={m.word_count} ratio={m.present_ratio} mean_sent={m.mean_sentence_words}")

past = "ذهبت مريم إلى الحديقة. وجدت كرة صغيرة. أعادتها إلى صديقتها. ابتسمت وضحكت."
m2 = ck.evaluate(past)
check("a past-tense story fails present_tense", m2.present_tense_ok is False,
      f"present={m2.present_verbs} past={m2.past_verbs} ratio={m2.present_ratio}")

long_story = " ".join(["تلعب مريم في الحديقة مع صديقتها الصغيرة"] * 12) + "."
m3 = ck.evaluate(long_story)
check("an over-long story fails length and register",
      (not m3.length_ok) and (not m3.simple_register_ok), f"words={m3.word_count}")

agg = ck.evaluate_all([short_present, past, long_story])
check("aggregate pass rates are computed", abs(agg["pass_rate"]["length"] - 2 / 3) < 1e-9)
check("tense coverage is reported", agg["tense_coverage"] == 1.0)

for w, expect in [("يكتبون", "P"), ("تذكرت", "Q"), ("كانت", "Q"), ("يوم", "-"),
                  ("بصوت", "-"), ("تفاحة", "-"), ("البيت", "-"), ("أرنب", "-")]:
    from noiseegra.constraint_metrics import _regex_tense_counts
    c = _regex_tense_counts([w])
    got = "P" if c["present"] else ("Q" if c["past"] else "-")
    check(f"tense probe {w!r} -> {expect}", got == expect, f"got {got}")


# --------------------------------------------------------------------------- #
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("all tests passed")
