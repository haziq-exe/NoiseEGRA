"""Random noise at other places in the transformer, sized alike.

    python tests/test_arch_noise.py
"""
import math, sys, warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))

from noiseegra.arch_noise import MECHANISMS, ArchNoise, calibrate  # noqa: E402
from noiseegra.fisher_calibration import _logits  # noqa: E402
from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402
from tiny_model import Tiny  # noqa: E402

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


DIM, LAYERS = 64, [2, 3]
NAMES = ["present_tense", "mature_register", "dialogue"]
torch.manual_seed(0)
VECS = {n: {l: torch.randn(DIM) for l in LAYERS} for n in NAMES}


def plan(mech="", size=0.0):
    return SteeringPlan.build(
        VECS, LAYERS, [ConstraintSpec(n, beta=1.0) for n in NAMES], rms_scale=1.0,
        noise_mode="none", noise_alpha=0.0, steer_budget=1.0, steer_prefill=True,
        prompt_tail_clear=2, arch_mechanism=mech, arch_size=size)


egra = Tiny()
ids = torch.tensor([[1] + [2 + (b % 250) for b in b"<s>write a story<a> once there"]])
n_prompt = ids.shape[-1] - 3
p = plan()
clean = _logits(egra, p, ids, n_prompt, with_offset=False)

for m in MECHANISMS:
    print(f"== {m} ==")
    a = ArchNoise(egra, p, m, seed=7, n_prompt=n_prompt)
    a.knob = {"rotate": 0.4, "headtemp": 0.8, "rope": 0.3, "value": 1.0, "mlpmask": 0.3}[m]
    a.attach()
    moved = _logits(egra, p, ids, n_prompt, with_offset=False)
    again = _logits(egra, p, ids, n_prompt, with_offset=False)
    a.detach()
    back = _logits(egra, p, ids, n_prompt, with_offset=False)
    check(f"{m}: changes the output", float((moved - clean).abs().max()) > 1e-4,
          f"{float((moved - clean).abs().max()):.4f}")
    check(f"{m}: is the same noise on every pass", torch.allclose(moved, again, atol=1e-5))
    check(f"{m}: removing it restores the output exactly", torch.allclose(back, clean, atol=1e-6))
    b = ArchNoise(egra, p, m, seed=7, n_prompt=n_prompt); b.knob = a.knob; b.attach()
    same = _logits(egra, p, ids, n_prompt, with_offset=False); b.detach()
    c = ArchNoise(egra, p, m, seed=8, n_prompt=n_prompt); c.knob = a.knob; c.attach()
    other = _logits(egra, p, ids, n_prompt, with_offset=False); c.detach()
    check(f"{m}: the same seed draws the same noise", torch.allclose(same, moved, atol=1e-5))
    check(f"{m}: another seed draws other noise", float((other - moved).abs().max()) > 1e-4)
    d = ArchNoise(egra, p, m, seed=7, n_prompt=n_prompt)
    got = calibrate(d, ids[:, :n_prompt], 0.1, iters=10)
    if got["reached"]:
        check(f"{m}: can be sized to a target", abs(got["distance"] - 0.1) < 0.03,
              f"asked 0.10, got {got['distance']:.3f} at knob {got['knob']:.4f}")
    else:
        check(f"{m}: an unreachable size is reported", got["knob"] > 0,
              f"reached {got['distance']:.3f} at the largest knob")

print("== the rotation keeps lengths and the rule components ==")
caught, pre = {}, {}
blocks = egra._get_transformer_blocks()


def _grab_before(m_, i, o):
    pre["before"] = (o[0] if isinstance(o, tuple) else o).detach().clone()
    return None


def _grab_after(m_, i, o):
    caught["after"] = (o[0] if isinstance(o, tuple) else o).detach().clone()
    return None


# Hooks run in the order they are registered: before, the rotation, after.
h0 = blocks[LAYERS[0]].register_forward_hook(_grab_before)
r = ArchNoise(egra, p, "rotate", seed=3, n_prompt=n_prompt); r.knob = 0.5
r.attach()
h1 = blocks[LAYERS[0]].register_forward_hook(_grab_after)
egra.model(input_ids=ids, use_cache=False)
r.detach(); h0.remove(); h1.remove()
x, y = pre["before"][0].float(), caught["after"][0].float()
check("the hidden state keeps its length", torch.allclose(x.norm(dim=-1), y.norm(dim=-1), rtol=1e-4))
P = p.layer_plans[LAYERS[0]].protect.float()
check("its component along the rule directions is untouched",
      torch.allclose(x @ P, y @ P, atol=1e-4))
check("and it has actually turned", float((x - y).norm()) > 1e-3)

print("== end to end ==")
PROMPT = [{"role": "system", "content": "s"}, {"role": "user", "content": "write a story"}]
egra.tokenizer.decode = lambda ids_, skip_special_tokens=True: " ".join(str(int(i)) for i in ids_)
q = plan("headtemp", 0.1)
outs = [egra.generate_with_orthogonal_steering(PROMPT, q, max_new_tokens=8, seed=s) for s in (1, 2)]
check("a story generates with the noise attached", all(isinstance(o, str) and o for o in outs))
check("each story was sized", len(getattr(q, "arch_log", []) or []) == 2)
check("nothing is left attached afterwards",
      torch.allclose(_logits(egra, p, ids, n_prompt, with_offset=False), clean, atol=1e-6))

print("== per sentence ==")
from noiseegra.arch_noise import SentenceCounter  # noqa: E402


class _Tok:
    def __init__(self, text): self.text = text
    def decode(self, ids, skip_special_tokens=True): return self.text


ps = ArchNoise(egra, p, "rotate", seed=4, n_prompt=n_prompt); ps.per_sentence = True
SentenceCounter(_Tok("word"), ps)(torch.tensor([[1, 2]]), torch.zeros(1, 3))
check("a word that ends nothing keeps the sentence", ps.sentence == 0)
cnt = SentenceCounter(_Tok("end."), ps)
cnt(torch.tensor([[1, 2]]), torch.zeros(1, 3))
check("a full stop moves to the next sentence", ps.sentence == 1, str(ps.sentence))
ps.knob = 0.4
ps.sentence = 0; ps.attach(); first = _logits(egra, p, ids, n_prompt, with_offset=False); ps.detach()
ps.sentence = 1; ps.attach(); second = _logits(egra, p, ids, n_prompt, with_offset=False); ps.detach()
ps.sentence = 0; ps.attach(); again = _logits(egra, p, ids, n_prompt, with_offset=False); ps.detach()
check("each sentence draws its own noise", float((first - second).abs().max()) > 1e-4)
check("and a sentence's draw is the same whenever it is asked for", torch.allclose(first, again, atol=1e-5))
flat_n = ArchNoise(egra, p, "rotate", seed=4, n_prompt=n_prompt); flat_n.knob = 0.4
flat_n.sentence = 0; flat_n.attach(); at0 = _logits(egra, p, ids, n_prompt, with_offset=False); flat_n.detach()
flat_n.sentence = 3; flat_n.attach(); at3 = _logits(egra, p, ids, n_prompt, with_offset=False); flat_n.detach()
check("without it the sentence makes no difference", torch.allclose(at0, at3, atol=1e-6))
qs = plan("rotate", 0.1); qs.arch_per_sentence = True
outs = [egra.generate_with_orthogonal_steering(PROMPT, qs, max_new_tokens=8, seed=s) for s in (1, 2)]
check("a story generates with per-sentence noise", all(isinstance(o, str) and o for o in outs))

sys.argv = ["x"]
from noiseegra.setup_experiment import ExperimentSpec, _ortho_tag  # noqa: E402
check("the run name records the mechanism and size",
      "__archheadtemp0p1" in _ortho_tag("M", ExperimentSpec(use_orthogonal_steering=True, steering_plan=q)))

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all passed")
