"""Per-story offsets, the story-level activation basis, and the opening window.

Everything here runs on CPU against a randomly initialised 64-wide model.
No downloads, no GPU.  python tests/test_offsets.py
"""

import contextlib, io, json, shutil, sys, warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from noiseegra.activation_basis import collect_prompt_pcs, collect_story_pcs  # noqa: E402
from noiseegra.run_labels import label_run  # noqa: E402
from noiseegra.subspace import SteeringPlan, ConstraintSpec, schedule_factor  # noqa: E402

FAILURES = []


def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


DIM, LAYERS = 64, [2, 3, 4]
torch.manual_seed(0)
VECS = {n: {l: torch.randn(DIM) for l in LAYERS}
        for n in ("closure", "present_tense", "simple_register", "dialogue")}
SPECS = [ConstraintSpec(n) for n in VECS]
RMS = 2.0


def plan(**kw):
    return SteeringPlan.build(VECS, LAYERS, SPECS, rms_scale=RMS, **kw)


print("== the opening-window schedule ==")
check("prefix is on for the first H steps",
      all(schedule_factor("prefix", t, 24) == 1.0 for t in (0, 1, 23)))
check("prefix is off from step H onward",
      all(schedule_factor("prefix", t, 24) == 0.0 for t in (24, 25, 400)))

p = plan(noise_mode="orth", noise_alpha=0.4, noise_schedule="prefix",
         noise_horizon=8, horizon=200)
# Not seed 0: the steering vectors above were drawn from it, so a noise draw
# under the same seed reproduces one of them exactly and its orthogonal part is
# zero -- which would make this test pass for the wrong reason.
torch.manual_seed(7)
inside = p.delta_for(2, 3)
torch.manual_seed(7)
outside = p.delta_for(2, 50)
torch.manual_seed(7)
steer_only = plan(noise_mode="none", noise_alpha=0.0).delta_for(2, 50)
check("noise is present inside the window", not torch.allclose(inside, steer_only, atol=1e-5))
check("noise is gone outside it", torch.allclose(outside, steer_only, atol=1e-5),
      f"max diff {float((outside - steer_only).abs().max()):.3g}")

# The closure direction ramps over `horizon`. A 8-token noise window must not
# drag that ramp in with it.
a = plan(noise_mode="none", noise_alpha=0.0, horizon=200).delta_for(2, 100)
b = plan(noise_mode="orth", noise_alpha=0.4, noise_schedule="prefix",
         noise_horizon=8, horizon=200).delta_for(2, 100)
check("the constraint schedules keep their own horizon", torch.allclose(a, b, atol=1e-5))

print("\n== offset magnitude ==")
basis = {l: torch.linalg.qr(torch.randn(DIM, 16))[0] for l in LAYERS}
for gamma in (0.05, 0.2):
    p = plan(noise_mode="none", noise_alpha=0.0, offset_gamma=gamma,
             offset_mode="free", offset_basis=basis)
    torch.manual_seed(1)
    p.resample_offset()
    got = float(p.layer_plans[2].offset.norm())
    want = gamma * RMS * DIM ** 0.5
    check(f"an offset at gamma={gamma} is {gamma:.0%} of the hidden state's own length",
          abs(got - want) < 1e-3, f"|offset|={got:.4f} want {want:.4f}")

# Same gamma, a subspace a quarter the size: the length must not change.
small = {l: basis[l][:, :4] for l in LAYERS}
p = plan(noise_mode="none", noise_alpha=0.0, offset_gamma=0.2,
         offset_mode="free", offset_basis=small)
torch.manual_seed(1)
p.resample_offset()
check("and does not depend on the rank of the subspace it was drawn from",
      abs(float(p.layer_plans[2].offset.norm()) - 0.2 * RMS * DIM ** 0.5) < 1e-3)

p = plan(noise_mode="none", noise_alpha=0.0, offset_gamma=0.2,
         offset_mode="free", offset_basis=basis, offset_norm="raw")
torch.manual_seed(1)
p.resample_offset()
check("offset_norm='raw' keeps the old, rank-dependent behaviour",
      abs(float(p.layer_plans[2].offset.norm()) - 0.2 * RMS * DIM ** 0.5) > 1e-2)

print("\n== the offset stays off the constraint directions ==")
p = plan(noise_mode="none", noise_alpha=0.0, offset_gamma=0.2,
         offset_mode="orth", offset_basis=basis, protect_extra=None)
torch.manual_seed(2)
p.resample_offset()
lp = p.layer_plans[2]
leak = float((lp.protect.t() @ lp.offset).norm() / lp.offset.norm())
check("none of it lies along a constraint direction", leak < 1e-4, f"leak {leak:.3g}")

p = plan(noise_mode="none", noise_alpha=0.0, offset_gamma=0.2,
         offset_mode="free", offset_basis=basis)
torch.manual_seed(2)
p.resample_offset()
lp = p.layer_plans[2]
free_leak = float((lp.protect.t() @ lp.offset).norm() / lp.offset.norm())
check("the unprojected control does leak onto them", free_leak > 1e-3, f"leak {free_leak:.3g}")

print("\n== the offset is one vector for the whole story ==")
p = plan(noise_mode="none", noise_alpha=0.0, offset_gamma=0.2,
         offset_mode="orth", offset_basis=basis)
torch.manual_seed(3)
p.resample_offset()
d0, d1 = p.delta_for(2, 0), p.delta_for(2, 1)
steer0 = plan(noise_mode="none", noise_alpha=0.0).delta_for(2, 0)
steer1 = plan(noise_mode="none", noise_alpha=0.0).delta_for(2, 1)
check("the same offset is added at every step",
      torch.allclose(d0 - steer0, d1 - steer1, atol=1e-5))
first = p.layer_plans[2].offset.clone()
p.resample_offset()
check("and a fresh one is drawn for the next story",
      not torch.allclose(first, p.layer_plans[2].offset, atol=1e-4))

print("\n== the offset can be confined to the prompt ==")
p = plan(noise_mode="none", noise_alpha=0.0, offset_gamma=0.2, offset_mode="orth",
         offset_basis=basis, offset_prefill=True, offset_decode=False)
torch.manual_seed(3)
p.resample_offset()
check("a prompt-only offset is still drawn", p.layer_plans[2].offset is not None)
d = p.delta_for(2, 5)
steer = plan(noise_mode="none", noise_alpha=0.0).delta_for(2, 5)
check("but adds nothing at a decode step", torch.allclose(d, steer, atol=1e-5),
      f"max diff {float((d - steer).abs().max()):.3g}")

print("\n== amplifying a story's own deviation ==")
torch.manual_seed(5)
amp_basis = {l: torch.linalg.qr(torch.randn(DIM, 12))[0] for l in LAYERS}
amp_mean = {l: torch.randn(DIM) for l in LAYERS}
p = plan(noise_mode="none", noise_alpha=0.0, amplify_lambda=2.0,
         amplify_basis=amp_basis, amplify_mean=amp_mean)
B = p.layer_plans[2].amp_basis
mu = p.layer_plans[2].amp_mean
h = torch.randn(DIM)
d = p.amplify_delta(2, h)
new_h = h + d
before = B.t() @ (h - mu)
after = B.t() @ (new_h - mu)
check("lambda=2 doubles the part of the state that lies in the story subspace",
      torch.allclose(after, 2 * before, atol=1e-4),
      f"|before|={float(before.norm()):.3f} |after|={float(after.norm()):.3f}")
rest = (h - mu) - B @ before
rest_after = (new_h - mu) - B @ after
check("and leaves the rest of the state untouched",
      torch.allclose(rest, rest_after, atol=1e-4))
leak = float((p.layer_plans[2].protect.t() @ d).norm() / d.norm())
check("nothing is added along a constraint direction", leak < 1e-4, f"leak {leak:.3g}")
# Prefill hands the hook every prompt position at once.
block = torch.randn(7, DIM)
db = p.amplify_delta(2, block)
check("a run of positions is amplified position by position",
      db.shape == (7, DIM)
      and torch.allclose(db[3], p.amplify_delta(2, block[3]), atol=1e-4),
      str(tuple(db.shape)))
check("lambda=1 is a no-op",
      plan(noise_mode="none", noise_alpha=0.0, amplify_lambda=1.0,
           amplify_basis=amp_basis, amplify_mean=amp_mean).amplify_delta(2, h) is None)

# Two stories that had started to diverge must be driven apart, not jointly moved.
h1, h2 = torch.randn(DIM), torch.randn(DIM)
gap = float((h1 - h2).norm())
gap_after = float(((h1 + p.amplify_delta(2, h1)) - (h2 + p.amplify_delta(2, h2))).norm())
check("two different states end up further apart than they started",
      gap_after > gap * 1.05, f"{gap:.3f} -> {gap_after:.3f}")

print("\n== run ids and labels ==")
from noiseegra.setup_experiment import _spec_to_run_id, make_specs  # noqa: E402

ids = {}
for tag, kw in [
    ("decode", dict(offset_gamma=0.2, offset_mode="orth", offset_basis=basis,
                    offset_basis_kind="story", offset_prefill=False)),
    ("prefill", dict(offset_gamma=0.2, offset_mode="orth", offset_basis=basis,
                     offset_basis_kind="story", offset_prefill=True)),
    ("stepbasis", dict(offset_gamma=0.2, offset_mode="orth", offset_basis=basis,
                       offset_basis_kind="step", offset_prefill=False)),
    ("window", dict(noise_mode="orth", noise_alpha=0.4, noise_schedule="prefix",
                    noise_horizon=24)),
    ("promptonly", dict(offset_gamma=0.2, offset_mode="orth", offset_basis=basis,
                        offset_basis_kind="story", offset_prefill=True,
                        offset_decode=False)),
    ("amp", dict(amplify_lambda=2.0, amplify_basis=amp_basis, amplify_mean=amp_mean)),
    ("amppre", dict(amplify_lambda=2.0, amplify_prefill=True,
                    amplify_basis=amp_basis, amplify_mean=amp_mean)),
]:
    kw.setdefault("noise_mode", "none"); kw.setdefault("noise_alpha", 0.0)
    spec = list(make_specs({"plan": plan(**kw)}))[0]
    ids[tag] = _spec_to_run_id("M", spec)
check("every setting gets its own run id", len(set(ids.values())) == len(ids),
      "\n    " + "\n    ".join(f"{k}: {v}" for k, v in ids.items()))
check("the prompt-onward offset says so in its name",
      "from the prompt onward" in label_run(ids["prefill"]).text,
      label_run(ids["prefill"]).text)
from noiseegra.run_labels import plan_summary  # noqa: E402
check("the header says which directions the offsets came from",
      any("whole stories differ" in l for l in plan_summary([ids["decode"]])),
      str(plan_summary([ids["decode"]])))
check("and says so differently for the decode-step basis",
      any("one decode step differs" in l for l in plan_summary([ids["stepbasis"]])),
      str(plan_summary([ids["stepbasis"]])))
check("the windowed noise names its window",
      "first 24 tokens" in label_run(ids["window"]).text, label_run(ids["window"]).text)
check("a prompt-only offset says so",
      "at the prompt only" in label_run(ids["promptonly"]).text,
      label_run(ids["promptonly"]).text)
check("amplification is named and its factor read back",
      label_run(ids["amp"]).family == "amplify"
      and label_run(ids["amp"]).magnitude == 2.0,
      label_run(ids["amp"]).text)
check("and where it is applied distinguishes the two arms",
      label_run(ids["amp"]).text != label_run(ids["amppre"]).text,
      label_run(ids["amppre"]).text)

print("\n== the story-level basis ==")
from tiny_model import Tiny  # noqa: E402

egra = Tiny()
msgs = [{"role": "user", "content": "Write a story."}]
axes = collect_story_pcs(egra, msgs, [2, 3], n_stories=6, rank=8, max_new_tokens=12,
                         skip_first=1, verbose=False)
check("a basis per layer", sorted(axes.basis) == [2, 3], str(sorted(axes.basis)))
k = axes.basis[2].shape[1]
check("rank is capped at one less than the number of stories", k <= 5, f"rank {k}")
check("the columns are orthonormal",
      torch.allclose(axes.basis[2].t() @ axes.basis[2], torch.eye(k), atol=1e-4))
check("the mean the directions are measured from comes back too",
      sorted(axes.mean) == [2, 3] and axes.mean[2].shape == (64,),
      str({l: tuple(v.shape) for l, v in axes.mean.items()}))
check("and the share of variation they carry is reported",
      0.0 < axes.explained <= 1.0, f"{axes.explained:.3f}")

print("\n== the basis from the prompt alone ==")
# No generation at all: one forward pass over the instruction.
calls = {"n": 0}
_gen = egra.model.generate
def counting_generate(*a, **k):
    calls["n"] += 1
    return _gen(*a, **k)
egra.model.generate = counting_generate
long_msgs = [{"role": "user", "content": "Write a story. " * 40}]
paxes = collect_prompt_pcs(egra, long_msgs, [2, 3], rank=8, verbose=False)
egra.model.generate = _gen
check("nothing was generated to build it", calls["n"] == 0, f"{calls['n']} generate calls")
check("a basis per layer", sorted(paxes.basis) == [2, 3], str(sorted(paxes.basis)))
pk = paxes.basis[2].shape[1]
check("the columns are orthonormal",
      torch.allclose(paxes.basis[2].t() @ paxes.basis[2], torch.eye(pk), atol=1e-4))
check("it carries the mean too, so amplification can use it",
      sorted(paxes.mean) == [2, 3] and paxes.mean[2].shape == (64,))
check("too few usable positions is an error, not a silent empty basis",
      _raises(lambda: collect_prompt_pcs(egra, long_msgs, [2], rank=8,
                                         skip_first=10_000, verbose=False)))

print("\n== cosine decay with its own horizon ==")
p = plan(noise_mode="orth", noise_alpha=0.8, noise_schedule="cosine_decay",
         noise_horizon=16, horizon=200)
steer = plan(noise_mode="none", noise_alpha=0.0)
sizes = []
for t in (0, 8, 15, 16, 40):
    torch.manual_seed(11)
    d = p.delta_for(2, t)
    torch.manual_seed(11)
    sizes.append(float((d - steer.delta_for(2, t)).norm()))
check("noise is strongest at the first token", sizes[0] == max(sizes), f"{sizes}")
check("it shrinks as the story runs", sizes[0] > sizes[1] > sizes[2], f"{sizes}")
check("and is gone at the horizon", sizes[3] < 1e-5 and sizes[4] < 1e-5, f"{sizes}")

print("\n== end to end on a tiny model ==")
import run_english_experiment as R  # noqa: E402

OUT = Path("/tmp/_offset_test"); shutil.rmtree(OUT, ignore_errors=True)
R.build_model = lambda mid, **kw: Tiny()
sys.argv = ["x", "--model", "Qwen3-8B", "--suite", "story", "window", "prompt",
            "amplify", "decay", "--lambda-sweep", "2.0",
            "--layers", "2", "5", "--task", "generic", "--stories", "2",
            "--out", str(OUT), "--max-new-tokens", "4", "--pca-rank", "2",
            "--protect-rank", "2", "--no-diversity", "--gamma-sweep", "0.1",
            "--alpha-sweep", "0.4", "--noise-horizon", "3",
            "--offset-rank", "6", "--offset-basis-stories", "5",
            "--offset-basis-tokens", "8"]
with contextlib.redirect_stdout(io.StringIO()) as buf:
    R.main()
out = buf.getvalue()
state = json.loads((OUT / "Qwen3-8B" / "state.json").read_text())
runs = sorted(state["runs"])
check("seven conditions: two offset arms, one prompt-only, two amplify, "
      "one window, one decaying",
      len(runs) == 7, "\n    " + "\n    ".join(runs))
check("the window and the decaying arm are not the same condition",
      sum("__nschp" in r for r in runs) == 1 and sum("__nschd" in r for r in runs) == 1,
      "\n    " + "\n    ".join(r for r in runs if "__nsch" in r))
check("every cell is filled",
      all(len(v) == 2 for v in state["runs"].values()),
      str({r: len(v) for r, v in state["runs"].items()}))
check("the basis is cached for the next run, under the kind it came from",
      (OUT / "Qwen3-8B" / "actpcs_prompt_Qwen3-8B.pt").is_file(),
      str(sorted(f.name for f in (OUT / "Qwen3-8B").glob("actpcs_*"))))
check("the table names the arms apart",
      len({l[:52] for l in out.splitlines() if "per-story offset" in l}) >= 2
      and "first 3 generated tokens" in out,
      "\n".join(l for l in out.splitlines() if "offset" in l or "first 3" in l))

before = {r: dict(v) for r, v in state["runs"].items()}
with contextlib.redirect_stdout(io.StringIO()) as buf2:
    R.main()
check("a second run regenerates nothing", "0 of 14 still to generate" in buf2.getvalue())
state2 = json.loads((OUT / "Qwen3-8B" / "state.json").read_text())
check("and leaves the stories alone", state2["runs"] == before)

print()
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all offset tests passed")
