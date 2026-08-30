"""End-to-end smoke test on a tiny locally-built Llama (no downloads, no GPU).

Exercises the real code paths: contrastive extraction, plan construction, the
generation hook, and the experiment-spec plumbing.

    python tests/test_orthosteer_smoke.py
"""

import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import BatchEncoding, LlamaConfig, LlamaForCausalLM  # noqa: E402

from noiseegra.EGRA_functions import EGRA  # noqa: E402
from noiseegra.setup_experiment import _spec_to_run_id, make_specs  # noqa: E402
from noiseegra.steering_vectors import (  # noqa: E402
    SteeringVectorExtractor,
    SteeringVectorSet,
    load_pairs,
)
from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


class ByteTokenizer:
    """Minimal stand-in: enough surface for EGRA's generation and extraction paths."""

    pad_token_id, eos_token_id, bos_token_id = 0, 1, 1

    def _ids(self, text):
        return [2 + (b % 250) for b in text.encode("utf-8")][:400]

    def __call__(self, text, return_tensors=None, add_special_tokens=True, **kw):
        ids = self._ids(text)
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        if return_tensors == "pt":
            return BatchEncoding(
                {
                    "input_ids": torch.tensor([ids], dtype=torch.long),
                    "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                }
            )
        return {"input_ids": ids}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return text + "<assistant>" if add_generation_prompt else text

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(65 + int(i) % 26) for i in ids)


class TinyEGRA(EGRA):
    def __init__(self, n_layers=8, hidden=64):
        cfg = LlamaConfig(
            vocab_size=256, hidden_size=hidden, intermediate_size=2 * hidden,
            num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=1024, pad_token_id=0, bos_token_id=1, eos_token_id=1,
        )
        torch.manual_seed(0)
        self.model = LlamaForCausalLM(cfg).eval()
        self.model.generation_config.pad_token_id = 0
        # No early stopping, so decode-step counts in the test are deterministic.
        self.model.generation_config.eos_token_id = None
        self.tokenizer = ByteTokenizer()
        self.device = "cpu"


print("== building a tiny model ==")
egra = TinyEGRA()
blocks = egra._get_transformer_blocks()
check("transformer blocks are discovered", len(blocks) == 8, f"n={len(blocks)}")
HID = egra.model.config.hidden_size
LAYERS = [2, 3, 4]

print("\n== contrastive extraction ==")
pairs = load_pairs()
check("bundled pair file has the three constraints",
      sorted(pairs) == ["closure", "present_tense", "simple_register"], f"{sorted(pairs)}")
for name, block in pairs.items():
    n = len(block["pairs"])
    keys_ok = all({"prefix", "positive", "negative"} <= set(p) for p in block["pairs"])
    check(f"'{name}' has usable pairs", n >= 10 and keys_ok, f"n={n}")

small = {k: {"pairs": v["pairs"][:4]} for k, v in pairs.items()}
vecs = SteeringVectorExtractor(egra).extract(small, LAYERS, pca_rank=3, verbose=False)
check("a direction per constraint per layer",
      all(sorted(vecs.vectors[c]) == LAYERS for c in small))
check("directions have the residual-stream shape",
      all(tuple(v.shape) == (HID,) for c in small for v in vecs.vectors[c].values()))
check("directions are non-degenerate",
      all(float(v.norm()) > 1e-6 for c in small for v in vecs.vectors[c].values()))
check("principal components stored",
      all(vecs.components[c][LAYERS[0]].shape == (HID, 3) for c in small))
check("diagnostics report consistency and coverage",
      all("consistency" in vecs.diagnostics[c][LAYERS[0]] for c in small))

tmp = Path("/tmp/_orthosteer_vecs.pt")
vecs.save(tmp)
reloaded = SteeringVectorSet.load(tmp)
check("save/load round-trips",
      torch.allclose(reloaded.vectors["closure"][LAYERS[0]], vecs.vectors["closure"][LAYERS[0]]))
tmp.unlink()

print("\n== plan + generation hook ==")
names = list(small)
specs = [ConstraintSpec(n, beta=1.0, schedule="constant") for n in names]
plan = SteeringPlan.build(
    vecs.vectors, LAYERS, specs, rms_scale=1.0,
    orthogonalize="lowdin", noise_mode="orth", noise_alpha=0.175,
    protect_extra=vecs.protect_extra(names, 3),
)
# 3 steering directions + 3 constraints x 3 principal components = 12 columns,
# all independent here because the hidden size (64) leaves plenty of room.
check("protect_extra enlarges the protected subspace to 12 dims",
      plan.protect_rank == 12, f"rank={plan.protect_rank}")

prompt = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
N_NEW = 6
# generate() runs one prefill forward (which already emits the first new token)
# followed by N_NEW-1 decode forwards, so only N_NEW-1 steps are perturbed. This
# matches the published L-Res bookkeeping exactly -- the final token is never
# perturbed under any of the noise methods in this repo.
DECODE_STEPS = N_NEW - 1

calls = []
real_delta_for = plan.delta_for


def counting_delta_for(layer, t, **kw):
    calls.append((layer, t, kw.get("with_noise", True)))
    return real_delta_for(layer, t, **kw)


plan.delta_for = counting_delta_for
out = egra.generate_with_orthogonal_steering(
    prompt, plan, max_new_tokens=N_NEW, temperature=1.0, seed=123
)
plan.delta_for = real_delta_for

check("generation returns text", isinstance(out, str) and len(out) > 0, f"{out!r}")
check("exactly one perturbation per targeted layer per decode step",
      len(calls) == len(LAYERS) * DECODE_STEPS,
      f"{len(calls)} calls, expected {len(LAYERS) * DECODE_STEPS}")
check("only the targeted layers are perturbed",
      {c[0] for c in calls} == set(LAYERS), f"{sorted({c[0] for c in calls})}")
check("decode steps run 0..DECODE_STEPS-1 (prefill is skipped)",
      sorted({c[1] for c in calls}) == list(range(DECODE_STEPS)))
check("noise is enabled on every decode step", all(c[2] for c in calls))

check("no hooks are left registered on the blocks",
      all(len(blocks[i]._forward_hooks) == 0 for i in range(len(blocks))))
check("no model-level pre-hook is left registered",
      len(egra.model._forward_pre_hooks) == 0)

a = egra.generate_with_orthogonal_steering(prompt, plan, max_new_tokens=N_NEW, seed=7)
b = egra.generate_with_orthogonal_steering(prompt, plan, max_new_tokens=N_NEW, seed=7)
check("same seed -> same story", a == b, f"{a!r}")

zero = SteeringPlan.build(vecs.vectors, LAYERS,
                          [ConstraintSpec(n, beta=0.0) for n in names],
                          rms_scale=1.0, noise_mode="none", noise_alpha=0.0)
plain = egra.generate(prompt, max_new_tokens=N_NEW, temperature=1.0, seed=7)
neutral = egra.generate_with_orthogonal_steering(prompt, zero, max_new_tokens=N_NEW, seed=7)
check("beta=0 with no noise reproduces plain generation exactly", neutral == plain,
      f"{neutral!r} vs {plain!r}")

strong = SteeringPlan.build(vecs.vectors, LAYERS,
                            [ConstraintSpec(n, beta=25.0) for n in names],
                            rms_scale=1.0, noise_mode="none", noise_alpha=0.0)
steered = egra.generate_with_orthogonal_steering(prompt, strong, max_new_tokens=N_NEW, seed=7)
check("strong steering changes the output", steered != plain, f"{steered!r} vs {plain!r}")

prefill = SteeringPlan.build(vecs.vectors, LAYERS,
                             [ConstraintSpec(n, beta=25.0) for n in names],
                             rms_scale=1.0, noise_mode="none", noise_alpha=0.0,
                             steer_prefill=True)
pcalls = []
rp = prefill.delta_for
prefill.delta_for = lambda layer, t, **kw: (pcalls.append(kw.get("with_noise", True)) or rp(layer, t, **kw))
egra.generate_with_orthogonal_steering(prompt, prefill, max_new_tokens=N_NEW, seed=7)
prefill.delta_for = rp
check("steer_prefill adds one noise-free pass per layer over the prompt",
      len(pcalls) == len(LAYERS) * (DECODE_STEPS + 1)
      and pcalls.count(False) == len(LAYERS),
      f"calls={len(pcalls)} noise_free={pcalls.count(False)}")

print("\n== experiment plumbing ==")
built = make_specs({"plan": plan, "temperature": 1.0})
check("make_specs detects the mode from a bare `plan` key",
      len(built) == 1 and built[0].use_orthogonal_steering)
rid = _spec_to_run_id("Tiny", built[0])
check("run id encodes the ablation axes",
      all(tok in rid for tok in ("ORTHO", "L2-4", "lowdin", "nzorth", "a0p175", "k")), rid)

variants = [
    {"plan": plan},
    {"plan": SteeringPlan.build(vecs.vectors, LAYERS, specs, rms_scale=1.0,
                                orthogonalize="gram_schmidt", noise_mode="orth")},
    {"plan": SteeringPlan.build(vecs.vectors, LAYERS, specs, rms_scale=1.0,
                                noise_mode="iso")},
    {"plan": SteeringPlan.build(vecs.vectors, LAYERS, specs, rms_scale=1.0,
                                noise_mode="para")},
    {"plan": SteeringPlan.build(vecs.vectors, LAYERS,
                                [ConstraintSpec(n, beta=2.0) for n in names],
                                rms_scale=1.0, noise_mode="orth")},
    {"plan": SteeringPlan.build(vecs.vectors, LAYERS, specs[:2], rms_scale=1.0,
                                noise_mode="orth")},
]
rids = [_spec_to_run_id("Tiny", s) for s in make_specs(*variants)]
check("every ablation arm gets a distinct run id", len(set(rids)) == len(rids))
for r in rids:
    print(f"      {r}")

try:
    from noiseegra.setup_experiment import ExperimentSpec

    _spec_to_run_id("Tiny", ExperimentSpec(use_orthogonal_steering=True))
    check("a steering spec without a plan is rejected", False)
except ValueError:
    check("a steering spec without a plan is rejected", True)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("smoke test passed")
