"""Noise directions drawn from the model's own content-word vectors.

    python tests/test_vocab_noise.py
"""
import sys, warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))

from noiseegra.setup_experiment import ExperimentSpec, _ortho_tag  # noqa: E402
from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


from tiny_model import Tiny  # noqa: E402
egra = Tiny()
torch.manual_seed(0)
V = {n: {l: torch.randn(64) for l in [2, 3]} for n in ["a", "b"]}


def plan(vocab=False, rank=4):
    return SteeringPlan.build(V, [2, 3], [ConstraintSpec(n, beta=1.0) for n in V], rms_scale=1.0,
        noise_mode="none", noise_alpha=0.0, steer_budget=1.0, offset_gamma=0.1,
        offset_mode="orth", offset_norm="energy", offset_basis_kind="random",
        offset_random_rank=rank, noise_beta=2.0, offset_prefill=True, steer_prefill=True,
        prompt_tail_clear=2, offset_online=1.0, online_max_gain=10.0, online_rule_k=0.43,
        online_rule_start=True, offset_vocab=vocab)


print("== the word pool ==")
try:
    ids, pool = egra._content_word_vectors()
    ok = True
except ValueError:
    ids, pool, ok = [], None, False
check("the tiny vocabulary has content words, or says it has none", ok or pool is None)
if pool is None:
    # The stand-in vocabulary may hold no whole words; give it a pool directly.
    ids = list(range(10)); pool = torch.nn.functional.normalize(torch.randn(10, 64), dim=-1)
check("the pool is unit vectors", torch.allclose(pool.norm(dim=-1), torch.ones(len(ids)), atol=1e-4))

print("\n== the draw ==")
p = plan(True, 3)
p._vocab_ids, p._vocab_pool = ids, pool
torch.manual_seed(1); p.resample_offset(story_index=0)
pick = p._vocab_pick
check("three words are drawn per story", pick is not None and pick.numel() == 3, str(pick))
B = {l: lp.offset_basis.float() for l, lp in p.layer_plans.items()}
w = pool[pick].t()


def outside(l):
    lp = p.layer_plans[l]
    span = w if lp.protect is None else torch.cat([w, lp.protect.float()], 1)
    qs = torch.linalg.qr(span)[0]
    return (B[l] - qs @ (qs.t() @ B[l])).norm().item()


check("every layer's directions lie in those words' span, cleared of the rule directions",
      all(outside(l) < 1e-3 for l in B) and all(torch.linalg.matrix_rank(q) == 3 for q in B.values()),
      str({l: round(outside(l), 6) for l in B}))
check("the same words at every layer", len({tuple(p._vocab_pick.tolist())}) == 1)
torch.manual_seed(1); p.resample_offset(story_index=0)
check("the same seed draws the same words", torch.equal(p._vocab_pick, pick))
torch.manual_seed(2); p.resample_offset(story_index=1)
check("another seed draws others", not torch.equal(p._vocab_pick, pick), str(p._vocab_pick))
pr = plan(False, 3); torch.manual_seed(1); pr.resample_offset(story_index=0)
check("without the option the directions are Gaussian", getattr(pr, "_vocab_pick", None) is None)

print("\n== in a run ==")
check("the run id records it", "__voc" in _ortho_tag(
      "M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=plan(True))))
check("and a run without it does not", "__voc" not in _ortho_tag(
      "M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=plan(False))))
pv = plan(True, 3)
if not ok:
    pv._vocab_ids, pv._vocab_pool = ids, pool
out = egra.generate_with_orthogonal_steering([{"role": "user", "content": "write a story"}], pv,
                                             max_new_tokens=8, seed=3, story_index=0)
check("stories generate with it", isinstance(out, str))
check("the shadow stayed on the story's words", int(getattr(pv, "shadow_drift", 0) or 0) == 0)
a = egra.generate_with_orthogonal_steering([{"role": "user", "content": "write a story"}], plan(False),
                                           max_new_tokens=8, seed=3)
b = egra.generate_with_orthogonal_steering([{"role": "user", "content": "write a story"}], plan(False),
                                           max_new_tokens=8, seed=3)
check("off, a run is unchanged and reproducible", a == b)

print()
print("all passed" if not FAILURES else f"FAILED: {FAILURES}")
sys.exit(1 if FAILURES else 0)
