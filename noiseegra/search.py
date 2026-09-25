"""Searching over several decode paths of the rule-steered model.

Beam search keeps the W most likely continuations of one model and returns the
best; the paths differ only in which tokens they chose. This module compares
that with paths that differ in how the model itself was perturbed. Every kind
produces W candidate texts for one prompt, all under the same rule steering:

- ``beam``   beam search of width W (the W final beams);
- ``dbs``    Diverse Beam Search (Vijayakumar et al., AAAI 2018): W groups of
             one beam, each penalised for repeating the tokens the groups
             before it chose at the same step;
- ``sample`` W independent samples at the run's temperature (best-of-W);
- ``noise``  one greedy path of the steered model and W - 1 greedy paths, each
             with its own per-story noise sized while writing (the full
             method: the rule's target, the measured starting length, the
             controller). Greedy, so each path is fixed by its noise alone.
- ``npad``   the same paths with noisy parallel approximate decoding's noise
             (Cho, 2016): isotropic, redrawn at every step, annealed as 1/t.
- ``fixed``  the same paths with one constant random vector each at the
             measured length and no controller (the per-sample vector of Liu et
             al., ICLR 2026, sized as ours).

Each candidate is scored by its mean log-probability per token under the
steered model with no noise -- the quantity beam search itself maximises -- and
the highest-scoring one is returned. The whole set, with scores, is left on
``egra.last_search`` for a caller that measures the set.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

KINDS = ("beam", "dbs", "sample", "noise", "npad", "fixed")
# Kinds whose paths are one greedy steered path plus width - 1 greedy paths
# under a perturbed plan: the full method ("noise"), noisy parallel approximate
# decoding ("npad": isotropic noise redrawn every step and annealed as 1/t, Cho
# 2016), and one fixed random vector per path at the measured length with no
# controller ("fixed").
PATH_KINDS = ("noise", "npad", "fixed")


@torch.no_grad()
def candidate_logprob(egra, plan, prompt, text: str) -> float:
    """Mean log-probability per token of ``text`` after ``prompt``, steered, no noise."""
    from .fisher_calibration import _logits

    chat = egra.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    device = egra._input_device()
    p_ids = egra.tokenizer(chat, return_tensors="pt")["input_ids"].to(device)
    t_ids = egra.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    if t_ids.shape[-1] == 0:
        return float("-inf")
    ids = torch.cat([p_ids, t_ids], dim=-1)
    n_prompt = int(p_ids.shape[-1])
    logits = _logits(egra, plan, ids, n_prompt, with_offset=False)
    logp = torch.log_softmax(logits[n_prompt - 1: ids.shape[-1] - 1].float(), dim=-1)
    return float(logp.gather(-1, t_ids[0].view(-1, 1).to(logp.device)).mean())


def search(egra, kind: str, width: int, steer_plan, prompt, *, seed: int,
           noise_plan=None, max_new_tokens: int = 600, max_words: Optional[int] = None,
           diversity_penalty: float = 0.5, temperature: float = 1.0) -> str:
    """W candidates by ``kind``, scored; returns the best and keeps the set."""
    if kind not in KINDS:
        raise ValueError(f"unknown search kind {kind!r}; choose from {KINDS}")
    width = int(width)
    gen = egra.generate_with_orthogonal_steering
    common = dict(max_new_tokens=max_new_tokens, max_words=max_words)
    texts: List[Tuple[str, int]] = []
    if kind in ("beam", "dbs"):
        extra = (dict(num_beam_groups=width, diversity_penalty=float(diversity_penalty))
                 if kind == "dbs" else {})
        gen(prompt, steer_plan, do_sample=False, num_beams=width,
            num_return_sequences=width, seed=seed, **extra, **common)
        texts = [(t, j) for j, t in enumerate(egra.last_candidates)]
    elif kind == "sample":
        for j in range(width):
            texts.append((gen(prompt, steer_plan, do_sample=True, temperature=temperature,
                              seed=seed * 1000 + j, **common), j))
    else:
        if noise_plan is None:
            raise ValueError(f"search '{kind}' needs the perturbed paths' plan")
        texts.append((gen(prompt, steer_plan, do_sample=False, seed=seed, **common), 0))
        for j in range(1, width):
            texts.append((gen(prompt, noise_plan, do_sample=False,
                              seed=seed * 1000 + j, story_index=j, **common), j))
    cands: List[Dict] = [
        {"path": j, "text": t, "score": candidate_logprob(egra, steer_plan, prompt, t)}
        for t, j in texts]
    egra.last_search = cands
    return max(cands, key=lambda c: c["score"])["text"]
