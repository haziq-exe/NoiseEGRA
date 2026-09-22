"""The next token tilted toward the rules, sized in the noise's units.

    python tests/test_rule_tilt.py

Checks that a rule's output profile favours what the rule-following side
predicts more and ignores what neither side predicts; that the extraction reads
one per rule from the same forward passes as the directions, and saves and loads
it; that the tilt moves each step's distribution the asked share of top-p's
distortion, toward the favoured tokens, measuring that distortion on the
noise-free row; and that a story generates with it on top of the noise sized
while writing.
"""
import math, sys, tempfile, types, warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

from noiseegra.fisher import fisher_rao_distance  # noqa: E402
from noiseegra.online_calibration import RuleTilt, sampler_probs  # noqa: E402
from noiseegra.run_labels import label_run  # noqa: E402
from noiseegra.setup_experiment import ExperimentSpec, _ortho_tag  # noqa: E402
from noiseegra.steering_vectors import (  # noqa: E402
    SteeringVectorExtractor, SteeringVectorSet, output_profile)

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


print("== a rule's output profile ==")
pos = torch.tensor([0.40, 0.10, 0.30, 0.2 - 1e-6, 1e-6])
neg = torch.tensor([0.10, 0.40, 0.30, 0.2 - 1e-6, 1e-6])
r = output_profile(pos, neg)
check("favours what the rule-following side predicts more", r[0] > 0 and r[1] < 0,
      f"{r[0]:.2f}, {r[1]:.2f}")
check("is neutral where both sides agree", abs(float(r[2])) < 1e-6)
check("and ignores what neither side predicts", float(r[4]) == 0.0)

print("\n== extraction ==")
from tiny_model import Tiny  # noqa: E402
egra = Tiny()
pairs = {
    "dialogue": {"pairs": [{"prefix": "Once, ", "positive": f"she says \"hi {i}\" to him.",
                            "negative": f"she waves {i} times at him."} for i in range(3)]},
    "present_tense": {"pairs": [{"prefix": "Then ", "positive": f"the dog runs {i} laps.",
                                 "negative": f"the dog ran {i} laps."} for i in range(3)]},
}
vs = SteeringVectorExtractor(egra).extract(pairs, [2, 3], system="write", user="a story",
                                           pca_rank=0, verbose=False)
V = egra.model.config.vocab_size
check("one profile per rule", set(vs.output_profiles) == {"dialogue", "present_tense"})
check("over the whole vocabulary", all(v.shape == (V,) for v in vs.output_profiles.values()))
check("finite", all(bool(torch.isfinite(v).all()) for v in vs.output_profiles.values()))
combo = vs.output_profile(["dialogue", "present_tense"])
check("the rules combine into one vector", combo is not None and combo.shape == (V,))
check("a set with none of them has none", vs.output_profile(["simile"]) is None)
rep = vs.profile_report(egra.tokenizer)
check("and each rule's profile can be read as tokens", len(rep) == 4 and "favours" in rep[0])
with tempfile.TemporaryDirectory() as d:
    vs.save(Path(d) / "v.pt")
    back = SteeringVectorSet.load(Path(d) / "v.pt")
check("and they are saved with the directions",
      set(back.output_profiles) == set(vs.output_profiles)
      and torch.allclose(back.output_profiles["dialogue"], vs.output_profiles["dialogue"]))

print("\n== the tilt ==")
torch.manual_seed(0)
V2 = 200
prof = torch.zeros(V2)
prof[:10] = 1.0             # ten tokens the rules favour
prof[10:20] = -1.0          # ten they disfavour
clean = torch.randn(V2) * 2.0
story = clean + 0.1 * torch.randn(V2)
for share in (0.25, 0.5, 1.0):
    t = RuleTilt(prof, share)
    x = torch.stack([story.clone(), clean.clone()])
    before = torch.softmax(x[0], -1)
    out = t(None, x)
    after = torch.softmax(out[0], -1)
    moved = float(fisher_rao_distance(before, after))
    budget = float(fisher_rao_distance(torch.softmax(clean, -1),
                                       sampler_probs(clean.view(1, -1), 1.8, 0.95)[0]))
    check(f"at {share:g} it moves the step {share:g} of top-p's distortion",
          abs(moved - share * budget) <= 0.035 * share * budget, f"{moved:.4f} vs {share * budget:.4f}")
    check(f"at {share:g} toward the favoured tokens",
          float(after[:10].sum()) > float(before[:10].sum())
          and float(after[10:20].sum()) < float(before[10:20].sum()))
check("the noise-free row is left alone", torch.equal(out[1], clean))
x = torch.stack([story.clone(), clean.clone()])
check("a share of zero changes nothing", torch.equal(RuleTilt(prof, 0.0)(None, x.clone()), x))
one = RuleTilt(prof, 0.5)(None, story.clone().view(1, -1))
check("with no noise-free row it measures top-p on the story itself",
      not torch.equal(one[0], story))
flat = torch.zeros(V2)
flat[0] = 50.0              # nearly certain of a token the profile is neutral about
tprof = torch.zeros(V2)
tprof[5] = 1.0              # favours one it gives no chance
t = RuleTilt(tprof, 1.0, max_beta=20.0)
t(None, flat.clone().view(1, -1))
check("the tilt never goes past its limit", t.history[-1][0] <= 20.0 + 1e-9)
sm = RuleTilt(prof, 0.5)
for _ in range(5):
    sm(None, torch.stack([story.clone(), clean.clone()]))
check("and each story reports how far it tilted", abs(sm.summary()["achieved"] - 0.5) < 0.02,
      f"{sm.summary()['achieved']:.3f}")

print("\n== a plan and a story ==")
from run_orthosteer_experiment import make_plan  # noqa: E402
LAYERS = [2, 3]
names = ["dialogue", "present_tense"]
base = dict(beta={n: 1.0 for n in names}, rms_scale=1.0, noise_mode="none", noise_alpha=0.0,
            steer_budget=1.0, offset_gamma=0.15, offset_mode="orth", offset_norm="energy",
            offset_basis_kind="random", offset_random_rank=8, noise_beta=2.0,
            offset_prefill=True, steer_prefill=True, prompt_tail_clear=2,
            offset_prefill_gain=1.5, offset_online=1.0)
q = make_plan(vs, LAYERS, names, output_tilt=0.5, **base)
check("a plan that tilts carries the rules' profile", q.output_profile is not None
      and q.output_profile.shape == (V,))
rid = _ortho_tag("M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=q))
check("the run name records the tilt", "__otilt0p5" in rid, rid[-60:])
check("and the label says so", "output tilted toward the rules at 0.5" in label_run(rid).text,
      label_run(rid).text)
none = SteeringVectorSet(vectors=vs.vectors)
try:
    make_plan(none, LAYERS, names, output_tilt=0.5, **base)
    check("asked to tilt without profiles, it refuses", False)
except ValueError:
    check("asked to tilt without profiles, it refuses", True)
PROMPT = [{"role": "user", "content": "write a story"}]
outs = [egra.generate_with_orthogonal_steering(PROMPT, q, max_new_tokens=12, seed=s) for s in range(2)]
check("stories generate with the tilt and the noise sized while writing",
      all(isinstance(o, str) and o for o in outs))
logs = getattr(q, "tilt_log", None) or []
check("each reports its tilt", len(logs) == 2 and all(x.get("steps", 0) >= 10 for x in logs))
check("the noise was still sized while writing", len(getattr(q, "online_log", None) or []) == 2)
check("and the copy still wrote the story's words", getattr(q, "shadow_drift", None) == 0)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all passed")
