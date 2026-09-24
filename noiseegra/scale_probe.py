"""What a model is like under the per-story noise, measured before any story.

The noise sized while writing aims at a target in top-p's own units: 1.0 moves
each next-token prediction as far, in Fisher-Rao distance, as top-p 0.95 at
temperature 1.8 does at that step. The target that keeps the writing coherent
was 1.0 on Qwen3-1.7B and below that on Llama-3.2-3B, so the unit alone does
not carry across models. This measures, on the task's own prompt and nothing
sampled, the quantities a rule for the target could be built from:

- the unit itself: how far top-p at 1.8 moves the model's next-token
  distribution per step (mean Fisher-Rao distance), and how uncertain that
  distribution is (entropy, top token's probability);
- sensitivity: how far random per-story noise, drawn exactly as the method
  draws it, moves the predictions at a range of lengths (as fractions of the
  residual norm);
- fluency cost: at each length, the greedy continuation written with the noise
  on, scored by the same steered model without it -- mean negative
  log-likelihood per token above the clean greedy passage's -- the cost in
  fluency the noise's shift in the predictions buys.

All along the model's greedy continuation of the prompt under the rule steering
alone (the calibration's reference passage), so nothing depends on sampling.
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch

from .fisher import fisher_rao_distance
from .fisher_calibration import (_logits, _probs, nucleus_unit, offset_distance,
                                 reference_passage, set_offset_length)

FRACTIONS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8)


def _entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p.clamp_min(1e-30).log() * p).sum(-1)


@torch.no_grad()
def _greedy(egra, plan, ids: torch.Tensor, n_prompt: int, n: int, with_offset: bool):
    for _ in range(n):
        lg = _logits(egra, plan, ids, n_prompt, with_offset=with_offset)
        ids = torch.cat([ids, lg[-1].argmax().view(1, 1).to(ids.device)], dim=-1)
    return ids


@torch.no_grad()
def _nll(egra, plan, ids: torch.Tensor, n_prompt: int) -> float:
    """Mean negative log-likelihood per continuation token under the clean plan."""
    lg = _logits(egra, plan, ids, n_prompt, with_offset=False)
    logp = torch.log_softmax(lg[n_prompt - 1: ids.shape[-1] - 1].float(), dim=-1)
    tgt = ids[0, n_prompt:].to(logp.device)
    return float(-logp.gather(-1, tgt.view(-1, 1)).mean())


def probe(egra, plan, prompt_ids: torch.Tensor, *, seeds: List[int], n_tokens: int = 48,
          fractions=FRACTIONS) -> Dict:
    n_prompt = int(prompt_ids.shape[-1])
    norm = float(plan.rms_scale) * math.sqrt(plan.dim)
    passage = reference_passage(egra, plan, prompt_ids, n_tokens)
    clean = _probs(_logits(egra, plan, passage, n_prompt, with_offset=False), n_prompt)
    unit = nucleus_unit(egra, plan, passage, n_prompt)
    ent = _entropy(clean)
    out = {
        "norm": norm, "rms_scale": float(plan.rms_scale), "dim": int(plan.dim),
        "unit": unit,
        "unit_t15": float(fisher_rao_distance(
            clean, _probs(_logits(egra, plan, passage, n_prompt, with_offset=False),
                          n_prompt, temperature=1.5, top_p=None)).mean()),
        "entropy": float(ent.mean()),
        "top_prob": float(clean.max(-1).values.mean()),
        "clean_nll": _nll(egra, plan, passage, n_prompt),
        "by_fraction": {},
    }
    for f in fractions:
        moved, nll = [], []
        for s in seeds:
            torch.manual_seed(int(s))
            plan.resample_offset(story_index=0)
            set_offset_length(plan, f * norm)
            moved.append(offset_distance(egra, plan, passage, n_prompt, clean))
            noisy = _greedy(egra, plan, prompt_ids, n_prompt, n_tokens, with_offset=True)
            nll.append(_nll(egra, plan, noisy, n_prompt))
        m = sum(moved) / len(moved)
        out["by_fraction"][f] = {"moved": m, "units": m / unit if unit > 0 else math.nan,
                                 "nll": sum(nll) / len(nll),
                                 "nll_rise": sum(nll) / len(nll) - out["clean_nll"]}
    return out
