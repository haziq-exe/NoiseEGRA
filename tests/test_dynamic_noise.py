"""Noise that changes direction while written, the clean-model guard, and B-Trans.

    python tests/test_dynamic_noise.py

B-Trans adds one fixed Gaussian offset per hidden-size norm layer, reproducible
from the story's seed and gone when the hooks are removed. The guard samples the
story only among tokens its noise-free shadow allows and counts the steps where
the noise pushed a token the shadow rules out; the correction removes the
noise's component along that token's exact gradient. The feedback turns the
noise toward, away from, or against the displacement it caused downstream.
Every new arm gets its own run id.
"""
import math, sys, types, warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

from noiseegra.dynamic_noise import (CleanGuard, attach_btrans, correct_offsets,  # noqa: E402
                                     feedback_update)
from noiseegra.setup_experiment import _spec_to_run_id, make_specs  # noqa: E402
from noiseegra.steering_vectors import SteeringVectorSet  # noqa: E402
from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402
from tiny_model import Tiny  # noqa: E402

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


egra = Tiny()
ids = torch.tensor([[1] + [2 + (b % 250) for b in b"<user>write a story<a>"]])

print("== B-Trans ==")
with torch.no_grad():
    clean = egra.model(ids).logits
    torch.manual_seed(3)
    hs = attach_btrans(egra.model, 0.02)
    a = egra.model(ids).logits
    for h in hs:
        h.remove()
    torch.manual_seed(3)
    hs = attach_btrans(egra.model, 0.02)
    b = egra.model(ids).logits
    for h in hs:
        h.remove()
    torch.manual_seed(4)
    hs = attach_btrans(egra.model, 0.02)
    c = egra.model(ids).logits
    for h in hs:
        h.remove()
    back = egra.model(ids).logits
n_layers = egra.model.config.num_hidden_layers
check("one offset per hidden-size norm: two per layer and the final one",
      len(hs) == 2 * n_layers + 1, f"{len(hs)}")
check("it changes the output", float((a - clean).abs().max()) > 1e-5)
check("the same seed draws the same offsets", torch.allclose(a, b))
check("another seed draws others", float((c - a).abs().max()) > 1e-6)
check("removing it restores the model", torch.allclose(back, clean))

print("\n== the guard ==")
V = 10
clean_s = torch.tensor([5.0, 4.9, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
story = clean_s.clone(); story[7] = 9.0          # the noise made a ruled-out token top
seen = []
g = CleanGuard(0.05, lambda ids_, tok: seen.append(tok) or True)
out = g(None, torch.stack([story, clean_s]))
pc = torch.softmax(clean_s, -1)
allowed = pc >= 0.05 * pc.max()
check("the story keeps only tokens the shadow allows",
      bool(torch.isinf(out[0][~allowed]).all()) and bool(torch.isfinite(out[0][allowed]).all()))
check("the shadow row is left alone", torch.equal(out[1], clean_s))
check("a ruled-out top token is a violation, handed to the correction",
      g.violations == 1 and seen == [7] and g.corrections == 1)
g(None, torch.stack([clean_s + 0.1, clean_s]))
check("an allowed top token is not", g.violations == 1 and g.steps == 2)
one = CleanGuard(0.05)
x = clean_s.view(1, -1)
check("with no shadow row it does nothing", one(None, x) is x and one.steps == 0)

print("\n== the correction and the feedback, on a plan ==")
DIM, LAYERS = 64, [2, 3]
NAMES = ["present_tense", "mature_register", "dialogue"]
torch.manual_seed(0)
VECS = {n: {l: torch.randn(DIM) for l in LAYERS} for n in NAMES}


def plan(beta=None, **kw):
    return SteeringPlan.build(
        VECS, LAYERS, [ConstraintSpec(n, beta=1.0) for n in NAMES], rms_scale=1.0,
        noise_mode="none", noise_alpha=0.0, steer_budget=1.0,
        offset_gamma=0.4, offset_mode="orth", offset_norm="energy",
        offset_basis_kind="random", offset_random_rank=8, noise_beta=beta,
        offset_prefill=True, steer_prefill=True, prompt_tail_clear=2,
        offset_prefill_gain=1.5, offset_online=1.0, **kw)


p = plan()
torch.manual_seed(1); p.resample_offset()
lp = p.layer_plans[LAYERS[0]]
v0 = lp.offset.clone()
grad = v0 / v0.norm() + 0.3 * torch.randn(DIM)
before = float(lp.offset @ grad)
correct_offsets(p, {LAYERS[0]: grad}, 1.0, 0)
gp = grad - lp.protect @ (lp.protect.t() @ grad)
check("the correction removes the noise's push on the token",
      abs(float(lp.offset @ gp)) < 1e-4 * float(v0.norm() * gp.norm()) and before > 0,
      f"{before:.3f} -> {float(lp.offset @ gp):.2e}")
check("and keeps the length", abs(float(lp.offset.norm() - v0.norm())) < 1e-4)
w = lp.offset.clone()
correct_offsets(p, {LAYERS[0]: -grad}, 1.0, 0)
check("a noise pushing the token down is left alone", torch.allclose(lp.offset, w))

pd = plan(beta=2.0)
torch.manual_seed(1); pd.resample_offset()
lq = pd.layer_plans[LAYERS[0]]
cur = lq.offset_at(5).float()
gv = cur + 0.1 * torch.randn(DIM)
correct_offsets(pd, {LAYERS[0]: gv}, 1.0, 5)
gq = gv - lq.protect @ (lq.protect.t() @ gv)
check("with a drifting direction every later step loses the component too",
      abs(float(lq.offset_at(50) @ (gq / gq.norm()))) < 1e-3 * float(lq.offset_at(50).norm()))

for mode, sign in (("toward", 1), ("away", -1)):
    q = plan()
    torch.manual_seed(2); q.resample_offset()
    l0 = q.layer_plans[LAYERS[0]]
    u = l0.offset / l0.offset.norm()
    delta = torch.randn(DIM)
    r = delta - (delta @ u) * u
    before = float(u @ r)
    feedback_update(q, delta, mode, 0.1)
    un = l0.offset / l0.offset.norm()
    rr = r - l0.protect @ (l0.protect.t() @ r)
    moved = float(un @ rr / rr.norm())
    check(f"{mode}: the direction turns {'toward' if sign > 0 else 'away from'} the response",
          sign * moved > 0.05, f"{moved:+.3f}")
    check(f"{mode}: by about the step asked", abs(float(un @ u) - math.cos(0.1)) < 0.02,
          f"cos {float(un @ u):.4f}")
    check(f"{mode}: the length is kept", abs(float(l0.offset.norm() - 0.4 * 8)) < 1e-3)
q = plan()
torch.manual_seed(2); q.resample_offset()
l0 = q.layer_plans[LAYERS[0]]
delta = torch.randn(DIM)
feedback_update(q, delta, "cancel", 1.0)
dp = delta - l0.protect @ (l0.protect.t() @ delta)
check("cancel at 1: the next direction is opposite the displacement",
      float((l0.offset / l0.offset.norm()) @ (dp / dp.norm())) < -0.999)

print("\n== stories ==")
PROMPT = [{"role": "user", "content": "write a story"}]
fx = plan(guard_alpha=0.9, correct_eta=1.0)
# Record, at every violation, the story's own log-probability of the token and
# the replay's, which must agree for the gradient to be the step's.
pairs = []
_orig_call = CleanGuard.__call__


def _spy(self, input_ids, scores):
    if scores.dim() == 2 and scores.shape[0] >= 2:
        self._row0 = torch.log_softmax(scores[0].float(), -1)
        inner = self.on_violation
        if inner is not None and not getattr(self, "_wrapped", False):
            def wrapped(ids_, tok, self=self, inner=inner):
                egra._last_replay_logprob = None
                got = inner(ids_, tok)
                if egra._last_replay_logprob is not None:
                    pairs.append((float(self._row0[tok]), egra._last_replay_logprob))
                return got
            self.on_violation = wrapped
            self._wrapped = True
    return _orig_call(self, input_ids, scores)


CleanGuard.__call__ = _spy
outs = [egra.generate_with_orthogonal_steering(PROMPT, fx, max_new_tokens=16, seed=s)
        for s in range(3)]
CleanGuard.__call__ = _orig_call
check("the replay reproduces the step it corrects",
      len(pairs) > 0 and max(abs(a - b) for a, b in pairs) < 1e-3,
      f"{len(pairs)} replays, worst gap {max((abs(a - b) for a, b in pairs), default=float('nan')):.1e}")
check("guarded stories generate", all(isinstance(o, str) and o for o in outs))
gl = fx.guard_log
check("each reports its guard", len(gl) == 3 and all(x["steps"] >= 10 for x in gl),
      str(gl[0]))
check("violations are corrected by replaying the step",
      sum(x["violations"] for x in gl) > 0 and sum(x["corrections"] for x in gl) > 0,
      f"{sum(x['violations'] for x in gl):.0f} violations, "
      f"{sum(x['corrections'] for x in gl):.0f} corrections")
check("the replay leaves the story and its copy in step", fx.shadow_drift == 0,
      f"drift {fx.shadow_drift}")
check("and the controller saw every step once",
      all(abs(o["steps"] - g_["steps"]) < 1 for o, g_ in zip(fx.online_log, gl)),
      f"{[o['steps'] for o in fx.online_log]} vs {[x['steps'] for x in gl]}")

for mode in ("toward", "away", "cancel"):
    fb = plan(feedback_mode=mode, feedback_eta=(1.0 if mode == "cancel" else 0.05),
              feedback_layer=5)
    egra.generate_with_orthogonal_steering(PROMPT, fb, max_new_tokens=16, seed=1)
    f = fb.feedback_log[-1]
    check(f"{mode}: turned the direction every step while writing",
          f["steps"] >= 10 and f["start_end_cos"] < 0.999,
          f"{f['steps']:.0f} turns, end cosine {f['start_end_cos']:.3f}")
try:
    egra.generate_with_orthogonal_steering(
        PROMPT, plan(beta=2.0, feedback_mode="toward", feedback_eta=0.05), max_new_tokens=4, seed=1)
    check("feedback with a drifting direction is refused", False)
except ValueError:
    check("feedback with a drifting direction is refused", True)

bt = SteeringPlan.build(
    VECS, LAYERS, [ConstraintSpec(n, beta=1.0) for n in NAMES], rms_scale=1.0,
    noise_mode="none", noise_alpha=0.0, steer_budget=1.0, steer_prefill=True,
    prompt_tail_clear=2, btrans_sigma=0.5)
o1 = egra.generate_with_orthogonal_steering(PROMPT, bt, max_new_tokens=8, seed=1)
check("a B-Trans story generates", isinstance(o1, str) and o1)
with torch.no_grad():
    after = egra.model(ids).logits
check("and leaves the model as it was", torch.allclose(after, clean))

print("\n== the arms and their run ids ==")
from run_orthosteer_experiment import build_suite  # noqa: E402
torch.manual_seed(0)
SV = SteeringVectorSet(
    vectors={n: {l: torch.randn(DIM) for l in LAYERS} for n in NAMES},
    components={n: {l: torch.linalg.qr(torch.randn(DIM, 8))[0] for l in LAYERS} for n in NAMES},
    positives={n: {l: torch.randn(DIM) for l in LAYERS} for n in NAMES},
)
# In the order the suite builds them.
ARMS = ["while", "whilenodrift", "whilenoproj", "whileguard", "whilefix", "whileminp",
        "whiletoward", "whileaway", "whilecancel", "whilefixed", "btrans"]
args = types.SimpleNamespace(
    protect_rank=8, horizon=200, steer_prefill=False, alpha=0.4, beta=1.0,
    noise_horizon=24, steer_budget=2.5, tail_sweep=[8], noise_beta_sweep=[2.0],
    offset_random_rank=16, front_tokens=40, baseline_temperature=1.8,
    baseline_top_p=0.95, baseline_top_k=40, offset_basis=None,
    offset_basis_kind="random", direction_source="extracted", keep_directions={},
    headline_arms=ARMS, headline_prompt_gains=[1.5], online_rule_k=0.43,
    online_rule_start=True, guard_alpha=0.05, correct_eta=1.0, feedback_eta=0.05,
    feedback_cancel_eta=1.0, feedback_layer=20, btrans_sigma=0.02)
items, _ = build_suite("headline", SV, LAYERS, list(NAMES), 1.5, args)
check("every asked arm is built once", len(items) == len(ARMS), f"{len(items)}")
rids = [_spec_to_run_id("M", s) for s in make_specs(*items)]
check("and every one has its own run id", len(set(rids)) == len(rids))
by = dict(zip(ARMS, zip(items, rids)))
check("no drift: no colour in the id", "__cn" not in by["whilenodrift"][1] and "__cn2" in by["while"][1])
check("no projection: isotropic", by["whilenoproj"][0]["plan"].offset_mode == "iso")
pf = by["whilefixed"][0]["plan"]
check("no controller: measured and held, same direction",
      pf.offset_measured and not pf.offset_online and pf.noise_beta == 2.0
      and pf.offset_mode == "orth" and pf.offset_random_rank == 16)
check("the guard and the fix are in the id",
      "__guard0p05" in by["whileguard"][1] and "__guard0p05fix1" in by["whilefix"][1])
check("min-p is on the arm, not the plan",
      by["whileminp"][0].get("min_p") == 0.05 and "minp0p05" in by["whileminp"][1])
check("the feedback arms carry their mode, and no drift",
      all(f"__fb{m}" in by[f"while{m}"][1] and by[f"while{m}"][0]["plan"].noise_beta is None
          for m in ("toward", "away", "cancel")))
pb = by["btrans"][0]["plan"]
check("B-Trans: rule steering, its own noise, no residual offset",
      pb.btrans_sigma == 0.02 and pb.offset_gamma == 0.0 and pb.steer_prefill
      and "__btrans0p02" in by["btrans"][1], by["btrans"][1][-40:])

sweep = types.SimpleNamespace(**{**vars(args), "headline_arms": ["btrans"],
                                 "btrans_sigma": [0.05, 0.1, 0.2]})
sw, _ = build_suite("headline", SV, LAYERS, list(NAMES), 1.5, sweep)
sids = [_spec_to_run_id("M", s) for s in make_specs(*sw)]
check("a B-Trans sweep builds one arm per size, each with its own id",
      [it["plan"].btrans_sigma for it in sw] == [0.05, 0.1, 0.2] and len(set(sids)) == 3
      and all(f"__btrans{t}" in i for t, i in zip(("0p05", "0p1", "0p2"), sids)))

npn = types.SimpleNamespace(**{**vars(args), "headline_arms": ["while", "whiletoward"],
                               "headline_prompt_gains": [1.0], "no_prompt_noise": True})
ni, _ = build_suite("headline", SV, LAYERS, list(NAMES), 1.5, npn)
nids = [_spec_to_run_id("M", s) for s in make_specs(*ni)]
check("no prompt noise: the noise is off on the prompt and the ids say so",
      all(not it["plan"].offset_prefill and it["plan"].offset_decode for it in ni)
      and all("__opre" not in i for i in nids) and len(set(nids)) == 2
      and all(i not in rids for i in nids))

def rule_plan(prefill):
    return SteeringPlan.build(
        VECS, LAYERS, [ConstraintSpec(n, beta=1.0) for n in NAMES], rms_scale=1.0,
        noise_mode="none", noise_alpha=0.0, steer_budget=1.0,
        offset_gamma=0.4, offset_mode="orth", offset_norm="energy",
        offset_basis_kind="random", offset_random_rank=8, noise_beta=2.0,
        offset_prefill=prefill, steer_prefill=True, prompt_tail_clear=2,
        offset_online=1.0, online_rule_k=0.43, online_rule_start=True)


starts = []
for pf in (True, False):
    rp = rule_plan(pf)
    egra.generate_with_orthogonal_steering(PROMPT, rp, max_new_tokens=4, seed=3)
    starts.append(next(iter(rp._rule_cache.values()))["start"])
    check(f"the plan's own prompt setting is restored after measuring ({pf})", rp.offset_prefill is pf)
check("with no prompt noise the starting length is measured as with it",
      abs(starts[0] - starts[1]) < 1e-6 * max(starts[0], 1.0), f"{starts[0]:.4f} vs {starts[1]:.4f}")

lo = types.SimpleNamespace(**{**vars(npn), "online_min_gain": 0.05})
li, _ = build_suite("headline", SV, LAYERS, list(NAMES), 1.5, lo)
lids = [_spec_to_run_id("M", s) for s in make_specs(*li)]
check("a lower controller floor reaches the plan and the id",
      all(it["plan"].online_min_gain == 0.05 for it in li) and all("min0p05" in i for i in lids)
      and all("min" not in i.split("__online")[1][:25] for i in nids))
seen_b = {}
_oi = OC_init = __import__("noiseegra.online_calibration", fromlist=["OnlineSizer"]).OnlineSizer.__init__
import noiseegra.online_calibration as _OC
def _spy_b(self, plan, target, **kw):
    seen_b["b"] = kw.get("bounds"); _oi(self, plan, target, **kw)
_OC.OnlineSizer.__init__ = _spy_b
fp = plan(); fp.online_min_gain = 0.05
egra.generate_with_orthogonal_steering(PROMPT, fp, max_new_tokens=4, seed=1)
_OC.OnlineSizer.__init__ = _oi
check("and the controller is given it", seen_b.get("b") == (0.05, 2.5), str(seen_b))

print("\n== the writing noise faded in ==")
from noiseegra.online_calibration import OnlineSizer  # noqa: E402
hp = types.SimpleNamespace(online_gain=1.0)
hs = OnlineSizer(hp, 1.0, hold=5)
Vn = 30
cl = torch.randn(Vn) * 2.0
near = cl + 0.01 * torch.randn(Vn)
gains = []
for _ in range(12):
    hs(None, torch.stack([near, cl])); gains.append(hp.online_gain)
check("the controller holds still while the noise fades in", all(g == 1.0 for g in gains[:7]),
      str([round(g, 2) for g in gains]))
check("and starts after it", gains[-1] > 1.0)
fz = types.SimpleNamespace(**{**vars(npn), "writing_fade_in": [16, 32]})
fi, _ = build_suite("headline", SV, LAYERS, list(NAMES), 1.5, fz)
fids = [_spec_to_run_id("M", s) for s in make_specs(*fi)]
check("each fade length is its own set of arms, in the plan and the id",
      len(fi) == 4 and sorted(it["plan"].offset_envelope_steps for it in fi) == [16, 16, 32, 32]
      and all(it["plan"].offset_envelope == "rise" for it in fi) and len(set(fids)) == 4
      and all("__envrise" in i for i in fids), str([i[-60:] for i in fids]))
fp2 = plan(); fp2.offset_envelope, fp2.offset_envelope_steps = "rise", 8
seen_h = {}
_oi2 = _OC.OnlineSizer.__init__
def _spy_h(self, plan, target, **kw):
    seen_h["h"] = kw.get("hold"); _oi2(self, plan, target, **kw)
_OC.OnlineSizer.__init__ = _spy_h
egra.generate_with_orthogonal_steering(PROMPT, fp2, max_new_tokens=12, seed=1)
_OC.OnlineSizer.__init__ = _oi2
check("a story faded in holds its controller for the fade", seen_h.get("h") == 8, str(seen_h))
torch.manual_seed(1); fp2.resample_offset()
lpf = fp2.layer_plans[LAYERS[0]]
check("the noise starts at zero and reaches full size at the end of the fade",
      fp2.envelope_at(0) == 0.0 and abs(fp2.envelope_at(8) - 1.0) < 1e-9 and 0 < fp2.envelope_at(4) < 1)

print("\n== the prompt's noise faded along the prompt ==")
tz = types.SimpleNamespace(**{**vars(args), "headline_arms": ["while"], "prompt_taper": [1.0, 0.25, 0.0]})
ti, _ = build_suite("headline", SV, LAYERS, list(NAMES), 1.5, tz)
tids = [_spec_to_run_id("M", s) for s in make_specs(*ti)]
check("one arm per fade, the flat one unchanged",
      [it["plan"].offset_taper for it in ti] == [1.0, 0.25, 0.0] and len(set(tids)) == 3
      and "__tap" not in tids[0] and "__tap0p25" in tids[1] and all(it["plan"].offset_prefill for it in ti),
      str([i[-50:] for i in tids]))
tstarts = []
for tp in (1.0, 0.0):
    rp = rule_plan(True); rp.offset_taper = tp
    egra.generate_with_orthogonal_steering(PROMPT, rp, max_new_tokens=4, seed=3)
    tstarts.append(next(iter(rp._rule_cache.values()))["start"])
    check(f"the fade is restored after measuring ({tp})", rp.offset_taper == tp)
check("a faded prompt measures the same starting length as a flat one",
      abs(tstarts[0] - tstarts[1]) < 1e-6 * max(tstarts[0], 1.0), f"{tstarts[0]:.4f} vs {tstarts[1]:.4f}")

from noiseegra.fisher_calibration import _logits  # noqa: E402
lg = {}
for tp in (1.0, 0.0):
    rp = rule_plan(True); rp.offset_taper = tp
    torch.manual_seed(4); rp.resample_offset()
    lg[tp] = _logits(egra, rp, ids, ids.shape[-1] - 3, with_offset=True)
check("a fade to zero changes what the prompt's noise does (it is not read as flat)",
      float((lg[1.0] - lg[0.0]).abs().max()) > 1e-5)

print("\n== the clean model's word where it is sure ==")
from noiseegra.dynamic_noise import ConfidentAnchor  # noqa: E402
ca = ConfidentAnchor(0.9)
sure = torch.tensor([9.0, 0.0, 0.0, 0.0]); unsure = torch.tensor([1.0, 0.9, 0.8, 0.0])
noisy = torch.tensor([0.0, 5.0, 0.0, 0.0])
o1 = ca(None, torch.stack([noisy, sure])); o2 = ca(None, torch.stack([noisy, unsure]))
check("where the shadow is sure the story takes its scores", torch.equal(o1[0], sure))
check("elsewhere the story keeps its own", torch.equal(o2[0], noisy) and ca.anchored == 1 and ca.steps == 2)
az = types.SimpleNamespace(**{**vars(args), "headline_arms": ["while", "whiletoward"], "anchor_p1": [0.9, 0.7]})
ai, _ = build_suite("headline", SV, LAYERS, list(NAMES), 1.5, az)
aids = [_spec_to_run_id("M", s) for s in make_specs(*ai)]
check("one set of arms per threshold, on the while arms, in the id",
      sorted(it["plan"].anchor_p1 for it in ai) == [0.7, 0.7, 0.9, 0.9] and len(set(aids)) == 4
      and all("__anchor0p" in i for i in aids))
ap_ = plan(anchor_p1=0.5)
egra.generate_with_orthogonal_steering(PROMPT, ap_, max_new_tokens=12, seed=2)
check("a story runs with the anchor and reports it", ap_.anchor_log and ap_.anchor_log[-1]["steps"] >= 10
      and ap_.shadow_drift == 0, str(ap_.anchor_log))

print("\n== the writing noise front-loaded ==")
pp = plan(); pp.offset_envelope, pp.offset_envelope_steps = "plateau", 10
check("full for the plateau, half way down in the middle of the fade, off after",
      pp.envelope_at(5) == 1.0 and abs(pp.envelope_at(15) - 0.5) < 1e-9 and pp.envelope_at(25) == 0.0)
hp2 = types.SimpleNamespace(online_gain=1.0, offset_envelope="plateau", envelope_at=pp.envelope_at)
hs2 = OnlineSizer(hp2, 1.0)
far2 = torch.randn(Vn) * 2.0
for _ in range(30):
    hs2(None, torch.stack([far2, cl]))
g_at_end = hp2.online_gain
check("the controller holds still once the envelope is off", abs(g_at_end - hs2.history[20][2]) < 1e-9,
      f"{hs2.history[20][2]:.3f} -> {g_at_end:.3f}")
pz = types.SimpleNamespace(**{**vars(args), "headline_arms": ["while"], "noise_plateau": [32, 64]})
pi_, _ = build_suite("headline", SV, LAYERS, list(NAMES), 1.5, pz)
pids = [_spec_to_run_id("M", s) for s in make_specs(*pi_)]
check("one arm per plateau length, in the plan and the id",
      [it["plan"].offset_envelope_steps for it in pi_] == [32, 64] and len(set(pids)) == 2
      and all("__envplateau" in i for i in pids))

sp = plan(); sp.offset_envelope, sp.offset_envelope_steps = "plateau", 10
torch.manual_seed(1); sp.resample_offset()
st0 = sp.delta_for(LAYERS[0], 3, with_offset=False)
sp.steer_envelope = True
check("steering that follows the envelope is full early", torch.allclose(sp.delta_for(LAYERS[0], 3, with_offset=False), st0))
check("and gone once the envelope is", sp.delta_for(LAYERS[0], 25, with_offset=False) is None
      or float(sp.delta_for(LAYERS[0], 25, with_offset=False).abs().max()) < 1e-9)
for ws, tagc in (("off", "__sdec0"), ("follow", "__senv")):
    wz = types.SimpleNamespace(**{**vars(args), "headline_arms": ["while"], "noise_plateau": [32], "writing_steer": ws})
    wi, _ = build_suite("headline", SV, LAYERS, list(NAMES), 1.5, wz)
    wid = _spec_to_run_id("M", make_specs(*wi)[0])
    check(f"writing steer '{ws}' reaches the plan and the id", tagc in wid, wid[-70:])

print("\n== rule debt ==")
from noiseegra.dynamic_noise import RuleDebt, debt_token_sets, owed_rules  # noqa: E402
check("owed rules read from the text",
      owed_rules("Mia runs.") == ["dialogue", "she", "he", "simile"]
      and owed_rules('"Hi," he says to her, quick as a fox.') == ["simile"]
      and owed_rules('"Hi," he says to her, like a fox.') == [])


class _Tok:
    vocab = ["<s>", " she", " Her", "her", ' "', "\u201cWe", " he", " His", " like", " as", " cat"]
    vocab_size = len(vocab)

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.vocab[i] for i in ids)


ts = debt_token_sets(_Tok())
check("each rule's words found in the vocabulary, only at a word's start",
      ts == {"dialogue": [4, 5], "she": [1, 2], "he": [6, 7], "simile": [8, 9]}, str(ts))
# Every id in the tiny vocabulary split among the rules, so the owed words carry
# real probability and the noise is bound to lower some of them.
V = int(egra.model.config.vocab_size)
egra._debt_sets = {"dialogue": list(range(0, V, 4)), "she": list(range(1, V, 4)),
                   "he": list(range(2, V, 4)), "simile": list(range(3, V, 4))}
dp_ = plan(debt_eta=1.0, debt_tau=0.0)
pairs.clear()
_orig_debt = RuleDebt.__call__


def _spy_debt(self, input_ids, scores):
    if scores.dim() == 2 and scores.shape[0] >= 2:
        self._row0 = torch.log_softmax(scores[0].float(), -1)
        inner = self.on_suppressed
        if inner is not None and not getattr(self, "_wrapped", False):
            def wrapped(ids_, toks, self=self, inner=inner):
                egra._last_replay_logprob = None
                got = inner(ids_, toks)
                if egra._last_replay_logprob is not None:
                    pairs.append((float(torch.logsumexp(self._row0[toks], 0)),
                                  egra._last_replay_logprob))
                return got
            self.on_suppressed = wrapped
            self._wrapped = True
    return _orig_debt(self, input_ids, scores)


RuleDebt.__call__ = _spy_debt
d_outs = [egra.generate_with_orthogonal_steering(PROMPT, dp_, max_new_tokens=16, seed=s)
          for s in range(3)]
RuleDebt.__call__ = _orig_debt
dl = dp_.debt_log
check("debt stories generate and report", len(dl) == 3 and all(isinstance(o, str) for o in d_outs),
      str(dl[0]))
check("the owed words' shortfall triggers corrections",
      sum(x["corrections"] for x in dl) > 0, f"{sum(x['corrections'] for x in dl):.0f}")
check("the replay reproduces the owed words' log-probability",
      len(pairs) > 0 and max(abs(a - b) for a, b in pairs) < 1e-3,
      f"{len(pairs)} replays, worst gap {max((abs(a - b) for a, b in pairs), default=float('nan')):.1e}")
check("and leaves the story and its copy in step", dp_.shadow_drift == 0)
off = plan(debt_eta=1.0, debt_tau=0.0); off.offset_envelope, off.offset_envelope_steps = "plateau", 2
egra.generate_with_orthogonal_steering(PROMPT, off, max_new_tokens=16, seed=1)
check("no corrections once the noise has faded", off.debt_log[-1]["steps"] <= 5,
      str(off.debt_log[-1]))
egra._debt_sets = None
dz = types.SimpleNamespace(**{**vars(args), "headline_arms": ["while"], "debt_eta": 0.5})
di, _ = build_suite("headline", SV, LAYERS, list(NAMES), 1.5, dz)
did = _spec_to_run_id("M", make_specs(*di)[0])
check("the debt option reaches the plan and the id",
      di[0]["plan"].debt_eta == 0.5 and "__debt0p5t0p3" in did, did[-60:])

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all passed")
