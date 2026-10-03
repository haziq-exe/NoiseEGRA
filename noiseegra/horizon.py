"""Directions that change what the story is about, not how the next word reads.

The per-story noise is sized by how far it moves the next-word distribution
against a noise-free copy of the model. That is the same currency top-p spends,
and on a model that is unsure of each word but sure of the story (OLMo 3 7B: top
word 0.55, yet 90% of its untouched stories judged the same) a random direction
spends it on wording: raising it doubled the noise and left the plots alone.

What should move instead is the part of the hidden state the model reads when it
decides what comes later. A displacement that changed the plan shows up as
different expectations about distant text *with the text in between held
fixed*: teacher-forced on the model's own continuation, the predictions many
tokens ahead move, while the next word barely does. So the directions are the
generalised eigenvectors of two Fisher matrices of the same displacement --
one for the predictions far ahead, one for the first few -- the directions that
move the far-ahead predictions most per unit of next-word change.

Nothing here is learned from stories. Both matrices are the model's own
Fisher-Rao geometry, pulled back to a vector added at the noise layers, along
the greedy continuation of this prompt that the noise is already sized on. The
noise is still a random draw, now inside this subspace, and still sized by the
same rule; only where it points changes.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch

from .fisher_calibration import _layers, _prompt_range


def _forward_with_probes(egra, plan, ids: torch.Tensor, n_prompt: int,
                         probes: Dict[int, torch.Tensor]) -> torch.Tensor:
    """Logits of ``ids`` with the rule steering applied as generation applies
    it, and ``probes[layer]`` added at every position of each noise layer.

    Out of place throughout, so gradients reach the probes. No noise: the
    directions are measured on the steered model the noise is added to.
    """
    blocks = egra._get_transformer_blocks()
    n = int(ids.shape[-1])
    lo, hi = _prompt_range(plan, n_prompt)

    def make_hook(li):
        def hook(module, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if not isinstance(t, torch.Tensor) or t.dim() != 3:
                return None
            rows = []
            dev, dt = t.device, t.dtype
            zero = torch.zeros(t.shape[-1], device=dev, dtype=dt)
            pre = None
            if plan.steer_prefill and hi > lo:
                d = plan.steering_only(li, 0, device=dev)
                pre = None if d is None else d.to(dt)
            for i in range(n):
                if i < n_prompt:
                    rows.append(pre if (pre is not None and lo <= i < hi) else zero)
                else:
                    d = plan.delta_for(li, i - n_prompt, with_noise=False,
                                       with_offset=False, device=dev)
                    rows.append(zero if d is None else d.to(dt))
            delta = torch.stack(rows).unsqueeze(0)
            new = t + delta + probes[li].to(device=dev, dtype=dt).view(1, 1, -1)
            if isinstance(out, (tuple, list)):
                return (new, *out[1:])
            return new
        return hook

    handles = [blocks[li].register_forward_hook(make_hook(li)) for li in probes]
    try:
        res = egra.model(input_ids=ids, use_cache=False, return_dict=True)
    finally:
        for h in handles:
            h.remove()
    return res.logits[0].float()


def horizon_basis(egra, plan, prompt_ids: torch.Tensor, passage: torch.Tensor, *,
                  rank: int = 8, near: Tuple[int, int] = (0, 4),
                  far: Tuple[int, int] = (16, 48), samples: int = 16,
                  ridge: float = 0.05, seed: int = 0
                  ) -> Tuple[Dict[int, torch.Tensor], Dict[str, float]]:
    """Per noise layer, an orthonormal (dim, rank) basis of the directions that
    move the predictions ``far`` tokens ahead most per unit of movement in the
    first ``near`` ones, both teacher-forced on ``passage``.

    Each Fisher matrix is estimated from ``samples`` score vectors: a token is
    drawn at every position of the window from the model's own distribution
    there, and the gradient of their summed log-probability with respect to a
    vector added at the layer is one sample. The generalised eigenproblem is
    solved in the span of those samples (rank at most 2 x samples); ``ridge``
    keeps the near-window matrix invertible in that span. The rule directions
    are removed, as for every other noise basis.

    Returns the bases and a diagnostic: how many times more far-ahead movement
    per unit of next-word movement the basis buys than a random direction does.
    """
    layers = _layers(egra, plan)
    if getattr(plan, "offset_layers", None):
        layers = [l for l in layers if l in plan.offset_layers]
    n_prompt = int(prompt_ids.shape[-1])
    m = int(passage.shape[-1]) - n_prompt
    near_rows = list(range(max(0, near[0]), min(near[1], m)))
    far_rows = list(range(max(0, far[0]), min(far[1], m)))
    if not near_rows or not far_rows:
        raise ValueError(f"passage of {m} tokens is too short for windows {near} / {far}")
    p0 = next(egra.model.parameters())
    probes = {l: torch.zeros(int(plan.dim), device=p0.device, dtype=torch.float32,
                             requires_grad=True) for l in layers}
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.enable_grad():
        logits = _forward_with_probes(egra, plan, passage, n_prompt, probes)
        # Row j predicts passage token j: the next-word distribution at the end
        # of the prompt is row 0.
        logp = torch.log_softmax(logits[n_prompt - 1: n_prompt - 1 + m], dim=-1)
        probs = logp.detach().exp().cpu()
        grads = {"near": {l: [] for l in layers}, "far": {l: [] for l in layers}}
        for _ in range(int(samples)):
            for key, rows in (("near", near_rows), ("far", far_rows)):
                ys = torch.multinomial(probs[rows], 1, generator=gen).view(-1)
                obj = logp[torch.tensor(rows, device=logp.device),
                           ys.to(logp.device)].sum()
                g = torch.autograd.grad(obj, [probes[l] for l in layers],
                                        retain_graph=True)
                for l, gl in zip(layers, g):
                    grads[key][l].append(gl.detach().float().cpu())
    del logits, logp

    bases: Dict[int, torch.Tensor] = {}
    gains = []
    for l in layers:
        gn = torch.stack(grads["near"][l])   # (s, d)
        gf = torch.stack(grads["far"][l])
        q = torch.linalg.qr(torch.cat([gf, gn]).t())[0]          # (d, k)
        a, b = gf @ q, gn @ q
        ff = a.t() @ a / a.shape[0]
        fn = b.t() @ b / b.shape[0]
        k = fn.shape[0]
        fn = fn + ridge * (torch.trace(fn) / k + 1e-12) * torch.eye(k)
        chol = torch.linalg.cholesky(fn)
        inv = torch.linalg.inv(chol)
        vals, vecs = torch.linalg.eigh(inv @ ff @ inv.t())
        top = vecs[:, torch.argsort(vals, descending=True)[: int(rank)]]
        u = inv.t() @ top                                        # back to the span
        v = torch.linalg.qr(q @ u)[0]                            # (d, rank)
        lp = plan.layer_plans.get(l)
        if lp is not None and lp.protect is not None:
            from .subspace import complement_basis
            v = complement_basis(v, lp.protect.detach().cpu().float())
        bases[l] = v
        # Far-per-near movement of the basis against random directions, both
        # measured with the same score samples.
        def ratio(dirs):
            dirs = dirs / dirs.norm(dim=0, keepdim=True).clamp_min(1e-12)
            num = (gf @ dirs).pow(2).mean(0)
            den = (gn @ dirs).pow(2).mean(0).clamp_min(1e-20)
            return (num / den).mean().item()
        rnd = torch.randn(gf.shape[1], 32, generator=gen)
        gains.append(ratio(v) / max(ratio(rnd), 1e-20))
    return bases, {"gain": float(sum(gains) / len(gains)),
                   "gain_min": float(min(gains)), "layers": float(len(layers))}
