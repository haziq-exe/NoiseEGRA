"""Sizing the noise by how far it moves the plan state, not the next-word odds.

    python tests/test_plan_size.py
"""
import contextlib, io, sys, warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))

import noiseegra.online_calibration as OC  # noqa: E402
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


def plan(prefill=True):
    return SteeringPlan.build(V, [2, 3], [ConstraintSpec(n, beta=1.0) for n in V], rms_scale=1.0,
        noise_mode="none", noise_alpha=0.0, steer_budget=1.0, offset_gamma=0.1,
        offset_mode="orth", offset_norm="energy", offset_basis_kind="random",
        offset_random_rank=8, noise_beta=2.0, offset_prefill=prefill, steer_prefill=True,
        prompt_tail_clear=2, offset_online=1.0, online_rule_k=0.43, online_rule_start=True)


PROMPT = [{"role": "user", "content": "write a story"}]
ALT = [{"role": "user", "content": "write a poem about the sea"}]

print("== the bisection ==")
p = plan()
ids = egra.tokenizer(egra.apply_chat_template(PROMPT, tokenize=False, add_generation_prompt=True),
                     return_tensors="pt")["input_ids"]
alt = egra.tokenizer(egra.apply_chat_template(ALT, tokenize=False, add_generation_prompt=True),
                     return_tensors="pt")["input_ids"]
check("the plan layer is two thirds up", OC._plan_layer(egra) == round(
      2 * len(egra._get_transformer_blocks()) / 3), str(OC._plan_layer(egra)))
small = OC.plan_space_start(egra, p, ids, alt, 1.0, n_tokens=8, iters=6)
big = OC.plan_space_start(egra, p, ids, alt, 3.0, n_tokens=8, iters=6)
check("two requests sit apart", small["ref"] > 0, f"{small['ref']:.3f}")
check("a bigger share asks for a longer noise", big["start"] >= small["start"],
      f"{small['start']:.3f} -> {big['start']:.3f}")
check("and the length found moves the plan state about as asked",
      not small["reached"] or abs(small["moves"] - small["goal"]) < 0.25 * small["goal"],
      f"{small['moves']:.3f} vs {small['goal']:.3f}")

print("\n== in a run ==")
OC.PLAN_KAPPA, OC.PLAN_ALT_PROMPT = 0.1, ALT
q = plan()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    egra.generate_with_orthogonal_steering(PROMPT, q, max_new_tokens=12, seed=1)
out = buf.getvalue()
check("the run reports the plan sizing", "[plan size]" in out, out.strip().splitlines()[-1][:120]
      if out.strip() else "")
check("and holds the size fixed", q.online_min_gain == q.online_max_gain == 1.0)
logs = getattr(q, "online_log", None) or []
check("the controller leaves it at 1x", logs and abs(logs[-1]["final_gain"] - 1.0) < 1e-6,
      f"{logs[-1]['final_gain']:.3f}" if logs else "no log")
check("the run id records it", "__plansz0p1" in _ortho_tag(
      "M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=q)),
      _ortho_tag("M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=q))[-40:])
nq = plan(prefill=False)
with contextlib.redirect_stdout(io.StringIO()):
    egra.generate_with_orthogonal_steering(PROMPT, nq, max_new_tokens=12, seed=1)
check("it runs with the prompt left clean", nq.online_min_gain == 1.0)
OC.PLAN_KAPPA, OC.PLAN_ALT_PROMPT = 0.0, None
check("and a run without it has no tag", "__plansz" not in _ortho_tag(
      "M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=plan())))

print("\n== forgetting the prompt's noise once the fade ends ==")
from noiseegra.EGRA_functions import _cache_tensors  # noqa: E402


def fplan(mode):
    p = plan()
    p.offset_envelope, p.offset_envelope_steps = "plateau", 3
    p.prompt_forget = mode
    return p


def last_cache(p):
    box = {}

    def grab(module, args, kwargs):
        if kwargs.get("past_key_values") is not None:
            box["pkv"] = kwargs["past_key_values"]

    h = egra.model.register_forward_pre_hook(grab, with_kwargs=True)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            egra.generate_with_orthogonal_steering(PROMPT, p, max_new_tokens=12, seed=1)
    finally:
        h.remove()
    return _cache_tensors(box["pkv"])


n_p = ids.shape[-1]
keep = fplan("")
kv = last_cache(keep)
check("without it the prompt's memory keeps the noise",
      keep._forgot_at is None and not torch.equal(kv[-1][0][0, ..., :n_p, :], kv[-1][0][1, ..., :n_p, :]))
fp = fplan("prompt")
kv = last_cache(fp)
check("it fires when the fade reaches nothing", fp._forgot_at == 6, str(fp._forgot_at))
check("after which the prompt's memory is the noise-free copy's",
      all(torch.equal(k[0, ..., :n_p, :], k[1, ..., :n_p, :]) for k, _ in kv))
check("and the words' memory still holds the noise they were written with",
      not all(torch.equal(k[0, ..., n_p:n_p + 4, :], k[1, ..., n_p:n_p + 4, :]) for k, _ in kv))
fa = fplan("all")
kv = last_cache(fa)
check("with all, everything read before the fade's end is the copy's too",
      all(torch.equal(k[0, ..., :n_p + 6, :], k[1, ..., :n_p + 6, :]) for k, _ in kv))
check("the run id records it", "__forgetprompt" in _ortho_tag(
      "M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=fp)))

print("\n== the amplified direction under the steering ==")
import noiseegra.hidden_context as HCA  # noqa: E402
HCA.AMPLIFY = (0.3, 3)
qa = plan()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    a = egra.generate_with_orthogonal_steering(PROMPT, qa, max_new_tokens=8, seed=2, story_index=0)
check("the steered story carries it", "[amplified 0.3x, 3 steps]" in buf.getvalue()
      and qa._amplify_log[1] > qa._amplify_log[0], str(getattr(qa, "_amplify_log", None)))
check("the run id records it", "__amp0p3s3" in _ortho_tag(
      "M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=qa)))
HCA.AMPLIFY = None
with contextlib.redirect_stdout(io.StringIO()):
    b = egra.generate_with_orthogonal_steering(PROMPT, plan(), max_new_tokens=8, seed=2)
check("and is gone when switched off", "__amp" not in _ortho_tag(
      "M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=plan())))

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all passed")
