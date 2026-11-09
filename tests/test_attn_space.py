"""Per-story noise on queries and prompt keys.

    python tests/test_attn_space.py
"""
import sys, warnings
from pathlib import Path
import torch
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import noiseegra.attn_space as AS  # noqa: E402
from noiseegra.setup_experiment import ExperimentSpec, _ortho_tag  # noqa: E402
from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402
from tiny_model import Tiny  # noqa: E402

FAILURES = []
def check(name, cond, extra=""):
    if not cond: FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")

egra = Tiny()
ids = torch.randint(2, 200, (1, 12))
blocks = egra._get_transformer_blocks()

def run(spec, seed=3):
    h, st = AS.attach(egra, [2, 3], 12, seed, spec, fade=2)
    grabbed = {}
    hk = blocks[2].self_attn.k_proj.register_forward_hook(lambda m, i, o: grabbed.__setitem__("k", o.detach().clone()))
    hq = blocks[2].self_attn.q_proj.register_forward_hook(lambda m, i, o: grabbed.__setitem__("q", o.detach().clone()))
    try:
        with torch.no_grad():
            out = egra.model(input_ids=ids, use_cache=True)
            step = egra.model(input_ids=ids[:, -1:], past_key_values=out.past_key_values, use_cache=True)
    finally:
        for x in h + [hk, hq]: x.remove()
    return out.logits[0, -1], step.logits[0, -1], grabbed

c0, c1, _ = run({})
q0, q1, _ = run({"q": 2.0})
check("query noise changes the prediction after the prompt", not torch.allclose(c0, q0, atol=1e-5))
check("and the next written word's", not torch.allclose(c1, q1, atol=1e-5))
k0, k1, _ = run({"k": 2.0})
check("prompt-key noise changes what later words read", not torch.allclose(c1, k1, atol=1e-5))
qa, _, _ = run({"q": 2.0}, seed=3); qb, _, _ = run({"q": 2.0}, seed=4)
check("each story has its own direction, the same story the same one",
      torch.allclose(q0, qa) and not torch.allclose(qa, qb, atol=1e-5))
d0, d1, _ = run({})
check("nothing stays hooked", torch.allclose(c0, d0) and torch.allclose(c1, d1))
# the fade: with fade 2 the push is gone by the fourth written token
h, st = AS.attach(egra, [2], 12, 3, {"q": 2.0}, fade=2)
seen = []
hq = blocks[2].self_attn.q_proj.register_forward_hook(lambda m, i, o: seen.append(o.detach().clone()))
with torch.no_grad():
    o = egra.model(input_ids=ids, use_cache=True); pk = o.past_key_values
    for _ in range(6):
        o = egra.model(input_ids=ids[:, -1:], past_key_values=pk, use_cache=True); pk = o.past_key_values
for x in h + [hq]: x.remove()
check("the query push fades out while writing", st["t"][2] == 6)

AS.ATTN_NOISE, AS.ATTN_FADE = {"q": 1.0, "k": 0.5}, 32
torch.manual_seed(0)
V = {n: {l: torch.randn(64) for l in [2, 3]} for n in ["a", "b"]}
p = SteeringPlan.build(V, [2, 3], [ConstraintSpec(n, beta=1.0) for n in V], rms_scale=1.0,
                       noise_mode="none", noise_alpha=0.0, steer_budget=1.0, steer_prefill=True)
P = [{"role": "user", "content": "write a story"}]
s1 = egra.generate_with_orthogonal_steering(P, p, max_new_tokens=8, seed=1, story_index=0)
check("the steered generator runs with it", isinstance(s1, str))
check("the run id records it", "__attq1k0p5f32" in _ortho_tag(
      "M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=p)))
AS.ATTN_NOISE = None
check("and not when off", "__att" not in _ortho_tag("M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=p)))
print(); print("all passed" if not FAILURES else f"FAILED: {FAILURES}"); sys.exit(1 if FAILURES else 0)
